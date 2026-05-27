import argparse
import json
import time
import random
from itertools import chain
from types import SimpleNamespace
from loguru import logger
import numpy as np
import torch
from rich import print
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.utils import is_flash_attn_2_available
from model import sample, load_and_process_dataset, extract_context_feature
from dflash import DFlashDraftModel
import distributed as dist
from kernel.domino import DraftCorrectionGraphRunner
import os


def is_domino_projector(projector_type):
    return projector_type in {"domino", "causal" + "_v5"}


def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()

def apply_domino_bias_gate(model, z_i: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Apply scalar hidden-based gate for Domino inference.

    z_i:
        [B, 1, hidden_size]
    bias:
        [B, 1, vocab_size]

    return:
        gated bias, [B, 1, vocab_size]
    """
    if (
        getattr(model, "use_bias_gate", False)
        and hasattr(model, "bias_gate")
    ):
        gate = torch.sigmoid(model.bias_gate(z_i))  # [B, 1, 1]
        bias = gate * bias

    return bias

@torch.inference_mode()
def dflash_generate(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    graph_runner=None,
    use_bias=True,
    bias_ablation=None,
    confidence_threshold: float = 0.0,
) -> SimpleNamespace:
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    shift_label = bool(getattr(model.config, "dflash_config", {}).get("shift_label", False))
    extra_buffer = block_size + 1 if shift_label else block_size

    output_ids = torch.full(
        (1, max_length + extra_buffer),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    # Prefill stage
    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True if block_size > 1 else False,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens:num_input_tokens+1] = sample(output.logits, temperature)
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

    time_to_first_token = cuda_time() - prefill_start

    # Decode stage
    decode_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    draft_prefill = True
    projector_type = model.projector_type
    prefix_len = model.pure_draft_prefix_len
    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        K = block_size if shift_label else block_size - 1
        verify_ids = torch.full(
            (1, K + 1),
            mask_token_id,
            dtype=torch.long,
            device=model.device,
        )
        verify_ids[:, 0] = output_ids[:, start]
        verify_position_ids = position_ids[:, start : start + K + 1]
        if block_size > 1:
            noise_embedding = target.model.embed_tokens(block_output_ids)
            parallel_hiddens = model(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, past_key_values_draft.get_seq_length(): start + block_size],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )
            if shift_label:
                parallel_hiddens = parallel_hiddens
            else:
                parallel_hiddens = parallel_hiddens[:, -block_size+1:, :]
            past_key_values_draft.crop(start)

            if not is_domino_projector(projector_type):
                raise ValueError(
                    "This reviewer package only supports Domino checkpoints; "
                    f"got projector_type={projector_type!r}."
                )
            if not use_bias:
                raise ValueError("This reviewer package keeps only the Domino bias path; pass --use-bias.")

            base_logits = target.lm_head(parallel_hiddens)
            if prefix_len > 0:
                prefix_token_ids = sample(base_logits[:, :prefix_len], temperature)
                verify_ids[:, 1 : 1 + prefix_len] = prefix_token_ids

            if graph_runner is not None:
                graph_prefix_ids = verify_ids[:, :1 + prefix_len].contiguous()
                graph_parallel_hiddens = parallel_hiddens[:, prefix_len:K, :].contiguous()
                graph_base_logits = base_logits[:, prefix_len:K, :].contiguous()

                verify_ids[:, 1 + prefix_len:] = graph_runner(
                    graph_prefix_ids,
                    graph_parallel_hiddens,
                    graph_base_logits,
                )
            else:
                realized_prefix_ids = verify_ids[:, : 1 + prefix_len]
                realized_prefix_embeds = target.model.embed_tokens(realized_prefix_ids)
                _, gru_hidden = model.prefix_gru(realized_prefix_embeds)

                for i in range(prefix_len, K):
                    z_i = parallel_hiddens[:, i : i + 1, :]
                    s_i = gru_hidden.transpose(0, 1)

                    if bias_ablation == "zero_e":
                        s_i = torch.zeros_like(s_i)
                    elif bias_ablation == "zero_z":
                        z_i = torch.zeros_like(z_i)
                    if model.use_bias_norm:
                        s_i = model.bias_norm(s_i)
                    bias = model.embed_proj(torch.cat([z_i, s_i], dim=-1))
                    bias = apply_domino_bias_gate(model, z_i, bias)
                    current_logit = base_logits[:, i : i + 1, :]
                    current_token_id = sample(current_logit + bias, temperature)
                    verify_ids[:, i + 1 : i + 2] = current_token_id

                    if i + 1 < K:
                        new_embed = target.model.embed_tokens(current_token_id)
                        _, gru_hidden = model.prefix_gru(new_embed, gru_hidden)
            if draft_prefill:
                draft_prefill = False
                decode_start = cuda_time()

        output = target(
            verify_ids,
            position_ids=verify_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True if block_size > 1 else False,
        )

        posterior = sample(output.logits, temperature)
        acceptance_length = (verify_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        output_ids[:, start : start + acceptance_length + 1] = verify_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        acceptance_lengths.append(acceptance_length+1)
        start += acceptance_length + 1
        past_key_values_target.crop(start)
        if block_size > 1:
            target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, :acceptance_length + 1, :]
        
        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids is not None:
        stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / num_output_tokens

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=acceptance_lengths,
    )


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
                response[bs] = dflash_generate(
                    model=draft_model,
                    target=target,
                    input_ids=input_ids,
                    mask_token_id=draft_model.mask_token_id,
                    max_new_tokens=args.max_new_tokens,
                    block_size=bs,
                    stop_token_ids=[tokenizer.eos_token_id],
                    temperature=args.temperature,
                    graph_runner = graph_runner,
                    use_bias = args.use_bias,
                    bias_ablation=args.bias_ablation,
                    confidence_threshold=args.confidence_threshold,
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
