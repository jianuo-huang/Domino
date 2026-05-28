import argparse
import json
import time
import random
from itertools import chain
from loguru import logger
import numpy as np
import torch
from rich import print
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import is_flash_attn_2_available
from model import load_and_process_dataset
from dflash import DFlashDraftModel, is_domino_projector
import distributed as dist
from kernel.domino import DraftCorrectionGraphRunner
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--draft-name-or-path", type=str, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--bias-ablation", type=str, default=None)
    parser.add_argument("--use-graph", action="store_true")
    parser.add_argument("--use-bias", action="store_true")
    parser.add_argument("--dump-benchmark-manifest", type=str, default=None)
    parser.add_argument("--dump-only", action="store_true")
    parser.add_argument("--confidence-threshold", type=float, default=0.0, help="Only inject token info into neighbor when its softmax confidence exceeds this threshold (0.0 = always inject).")
    parser.add_argument("--answer-file", type=str, default=None, help="Output answer file (jsonl) to store generation results for both b=1 and b=k.")
    parser.add_argument("--attn-implementation", type=str, default=None, choices=["eager", "sdpa", "flash_attention_2"], help="Attention implementation for target and draft models. Default: auto-detect flash_attn.")

    args = parser.parse_args()

    if not args.dump_only and (args.model_name_or_path is None or args.draft_name_or_path is None):
        parser.error("--model-name-or-path and --draft-name-or-path are required unless --dump-only is set")

    # Fast path: dump-only mode skips model loading / CUDA init entirely
    if args.dump_only:
        dataset = load_and_process_dataset(args.dataset)
        if args.max_samples is not None and len(dataset) > args.max_samples:
            dataset = dataset.shuffle(seed=0).select(range(args.max_samples))
        if args.dump_benchmark_manifest:
            with open(args.dump_benchmark_manifest, "w", encoding="utf-8") as f:
                for selected_sample_idx, instance in enumerate(dataset):
                    record = {
                        "selected_sample_idx": selected_sample_idx,
                        "question_id": selected_sample_idx,
                        "turns": instance["turns"],
                        "num_turns": len(instance["turns"]),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"Saved benchmark manifest with {len(dataset)} samples to {args.dump_benchmark_manifest}")
        return

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dist.init()
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")

    if args.attn_implementation is not None:
        attn_impl = args.attn_implementation
        logger.info(f"Using specified attention implementation: {attn_impl}")
    else:
        def has_flash_attn():
            if is_flash_attn_2_available():
                return True
            logger.warning("FlashAttention2 is not available. Falling back to torch.sdpa. The speedup will be lower.")
            return False

        installed_flash_attn = has_flash_attn()
        attn_impl = "flash_attention_2" if installed_flash_attn else "sdpa"
        logger.info(f"Auto-detected attention implementation: {attn_impl}")

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
    ).to(device).eval()

    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_name_or_path,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
    ).to(device).eval()
    logger.info(f"[VERIFY] Target attn_implementation: {target.config._attn_implementation}")
    logger.info(f"[VERIFY] Draft attn_implementation: {draft_model.config._attn_implementation}")

    block_size = args.block_size if args.block_size is not None else draft_model.block_size

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = load_and_process_dataset(args.dataset)

    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

    if args.dump_benchmark_manifest and dist.is_main():
        with open(args.dump_benchmark_manifest, "w", encoding="utf-8") as f:
            for selected_sample_idx, instance in enumerate(dataset):
                record = {
                    "selected_sample_idx": selected_sample_idx,
                    "question_id": selected_sample_idx,
                    "turns": instance["turns"],
                    "num_turns": len(instance["turns"]),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Saved benchmark manifest with {len(dataset)} samples to {args.dump_benchmark_manifest}")

    hidden_size = int(target.lm_head.weight.shape[1])
    vocab_size = int(target.lm_head.weight.shape[0])
    prefix_len = int(getattr(draft_model, "pure_draft_prefix_len", 0))
    projector_type = getattr(draft_model, "projector_type", None)
    graph_runner = None
    if not is_domino_projector(projector_type):
        raise ValueError(
            "This reviewer package only supports Domino checkpoints; "
            f"got projector_type={projector_type!r}."
        )
    if args.use_graph:
        shift_label = bool(
            getattr(draft_model.config, "dflash_config", {}).get("shift_label", False)
        )
        K = block_size if shift_label else block_size - 1
        steps = K - prefix_len

        graph_runner = DraftCorrectionGraphRunner(
            draft_model=draft_model,
            target_model=target,
            batch_size=1,
            steps=steps,
            hidden_dim=hidden_size,
            gru_hidden_dim=draft_model.prefix_gru.hidden_size,
            vocab_size=vocab_size,
            prefix_token_count=1 + prefix_len,
            device=device,
        )
    answers = []
    indices = range(dist.rank(), len(dataset), dist.size())
    for idx in tqdm(indices, disable=not dist.is_main()):
        instance = dataset[idx]
        messages = []
        choice_b1 = {"index": 0, "block_size": 1, "turns": [], "new_tokens": [], "wall_time": [], "prefill_times": [], "decode_times": [], "acceptance_lengths": []}
        choice_bk = {"index": 1, "block_size": block_size, "turns": [], "new_tokens": [], "wall_time": [], "prefill_times": [], "decode_times": [], "acceptance_lengths": []}
        for turn_index, user_content in enumerate(instance["turns"]):
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)

            response = {}
            for bs in [1, block_size]:
                response[bs] = draft_model.spec_generate(
                    target=target,
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    block_size=bs,
                    stop_token_ids=[tokenizer.eos_token_id],
                    temperature=args.temperature,
                    graph_runner=graph_runner,
                    use_bias=args.use_bias,
                    bias_ablation=args.bias_ablation,
                    confidence_threshold=args.confidence_threshold,
                    return_dict=True,
                )

            # Record results for both b=1 and b=k
            for choice, bs in [(choice_b1, 1), (choice_bk, block_size)]:
                r = response[bs]
                generated_ids = r.output_ids[0, r.num_input_tokens:]
                output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                choice["turns"].append(output_text)
                choice["new_tokens"].append(int(r.num_output_tokens))
                prefill_t = float(r.time_to_first_token)
                decode_t = float(r.time_per_output_token) * int(r.num_output_tokens)
                choice["prefill_times"].append(prefill_t)
                choice["decode_times"].append(decode_t)
                choice["wall_time"].append(prefill_t + decode_t)
                choice["acceptance_lengths"].append([int(x) for x in r.acceptance_lengths])

            # Use b=k result as conversation history (same as original logic)
            spec_response = response[block_size]
            generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})

        answers.append({
            "question_id": idx,
            "choices": [choice_b1, choice_bk],
            "tstamp": time.time(),
        })

    if dist.size() > 1:
        answers = dist.gather(answers, dst=0)
        if not dist.is_main():
            return
        answers = list(chain(*answers))

    answers.sort(key=lambda x: x["question_id"])

    # Write answer file
    if args.answer_file and dist.is_main():
        os.makedirs(os.path.dirname(args.answer_file) or ".", exist_ok=True)
        with open(args.answer_file, "w", encoding="utf-8") as f:
            for ans in answers:
                f.write(json.dumps(ans, ensure_ascii=False) + "\n")
        print(f"Saved answer file with {len(answers)} samples to {args.answer_file}")

    # Compute and print stats
    t1 = np.mean([ans["choices"][0]["decode_times"][0] / max(1, ans["choices"][0]["new_tokens"][0]) for ans in answers if ans["choices"][0]["new_tokens"]])
    tb = np.mean([ans["choices"][1]["decode_times"][0] / max(1, ans["choices"][1]["new_tokens"][0]) for ans in answers if ans["choices"][1]["new_tokens"]])
    print(f"Decoding speedup: {t1 / tb:.2f}")

    acceptance_lengths = list(chain(*[ans["choices"][1]["acceptance_lengths"][0] for ans in answers if ans["choices"][1]["acceptance_lengths"]]))
    tau_per_step = np.mean(acceptance_lengths) if acceptance_lengths else 0
    tau_per_sample = np.mean([np.mean(ans["choices"][1]["acceptance_lengths"][0]) for ans in answers if ans["choices"][1]["acceptance_lengths"]])
    print(f"Average Acceptance length (per step):  {tau_per_step:.2f}")
    print(f"Average Acceptance length (per sample): {tau_per_sample:.2f}")

    shift_label = bool(getattr(draft_model.config, "dflash_config", {}).get("shift_label", False))
    if shift_label:
        histogram = [acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(block_size + 2)]
    else:
        histogram = [acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(block_size + 1)]
    print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")

if __name__ == "__main__":
    main()
