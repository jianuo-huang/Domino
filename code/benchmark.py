import argparse
import json
import os
import random
import statistics
import time
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from rich import print
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import is_flash_attn_2_available

import distributed as dist
from accelerator import (
    ACCELERATOR_BACKENDS,
    get_device,
    manual_seed_all,
    resolve_backend,
    set_device,
)
from dflash import (
    DFlashDraftModel,
    is_domino_projector,
    target_greedy_generate,
)
from model import load_and_process_dataset


TARGET_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def normalize_draft_config_for_benchmark(config):
    dflash_config = dict(getattr(config, "dflash_config", {}) or {})
    if dflash_config.get("projector_type") == "causal_v5":
        dflash_config["projector_type"] = "domino"
    if "emb_dim" not in dflash_config:
        emb_dim = getattr(config, "emb_dim", None)
        if emb_dim is not None:
            dflash_config["emb_dim"] = emb_dim
    if "gru_hidden_dim" not in dflash_config:
        gru_hidden_dim = getattr(config, "gru_hidden_dim", None)
        if gru_hidden_dim is not None:
            dflash_config["gru_hidden_dim"] = gru_hidden_dim
        elif "emb_dim" in dflash_config:
            dflash_config["gru_hidden_dim"] = dflash_config["emb_dim"]
    config.dflash_config = dflash_config
    return config


def load_draft_model_for_benchmark(
    model_name_or_path: str,
    attn_impl: str,
    revision: str | None,
):
    draft_config = AutoConfig.from_pretrained(
        model_name_or_path, revision=revision
    )
    draft_config = normalize_draft_config_for_benchmark(draft_config)
    return DFlashDraftModel.from_pretrained(
        model_name_or_path,
        revision=revision,
        config=draft_config,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
    )


def _load_instances(args) -> list[dict]:
    if args.benchmark_manifest:
        with open(args.benchmark_manifest, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    dataset = load_and_process_dataset(args.dataset)
    dataset = dataset.shuffle(seed=0)
    start = int(args.sample_offset)
    stop = len(dataset)
    if args.max_samples is not None:
        stop = min(stop, start + int(args.max_samples))
    return [dataset[index] for index in range(start, stop)]


def _dump_manifest(path: str, instances: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for selected_index, instance in enumerate(instances):
            record = {
                "selected_sample_idx": selected_index,
                "question_id": int(instance.get("question_id", selected_index)),
                "turns": instance["turns"],
                "num_turns": len(instance["turns"]),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _new_choice(index: int, mode: str, block_size: int) -> dict:
    return {
        "index": index,
        "mode": mode,
        "block_size": block_size,
        "turns": [],
        "output_ids": [],
        "new_tokens": [],
        "wall_time": [],
        "prefill_times": [],
        "draft_setup_times": [],
        "decode_times": [],
        "time_per_output_token": [],
        "peak_memory_mb": [],
        "acceptance_lengths": [],
    }


def _median_metric(responses, name: str) -> float:
    return float(statistics.median(float(getattr(item, name)) for item in responses))


def _record_choice(choice, responses, tokenizer) -> str:
    first = responses[0]
    generated = first.output_ids[0, first.num_input_tokens :]
    generated_list = [int(token) for token in generated.tolist()]
    for response in responses[1:]:
        other = response.output_ids[0, response.num_input_tokens :].tolist()
        if generated_list != [int(token) for token in other]:
            raise RuntimeError("Repeated deterministic runs produced different tokens.")

    text = tokenizer.decode(generated, skip_special_tokens=True)
    choice["turns"].append(text)
    choice["output_ids"].append(generated_list)
    choice["new_tokens"].append(int(first.num_output_tokens))
    choice["wall_time"].append(_median_metric(responses, "total_wall_time"))
    choice["prefill_times"].append(_median_metric(responses, "target_prefill_time"))
    choice["draft_setup_times"].append(
        _median_metric(responses, "draft_setup_time")
    )
    choice["decode_times"].append(
        _median_metric(responses, "steady_state_decode_time")
    )
    choice["time_per_output_token"].append(
        _median_metric(responses, "time_per_output_token")
    )
    choice["peak_memory_mb"].append(_median_metric(responses, "peak_memory_mb"))
    choice["acceptance_lengths"].append(
        [int(value) for value in first.acceptance_lengths]
    )
    return text


def _write_and_merge_answers(answer_file: str | None, answers: list[dict]):
    if not answer_file:
        if dist.size() == 1:
            return answers
        gathered = dist.gather(answers, dst=0)
        if not dist.is_main():
            return None
        return list(chain(*gathered))

    os.makedirs(os.path.dirname(answer_file) or ".", exist_ok=True)
    if dist.size() == 1:
        merged = answers
    else:
        rank_path = f"{answer_file}.rank{dist.rank()}.jsonl"
        with open(rank_path, "w", encoding="utf-8") as handle:
            for answer in answers:
                handle.write(json.dumps(answer, ensure_ascii=False) + "\n")
        dist.barrier()
        if not dist.is_main():
            return None
        merged = []
        for rank in range(dist.size()):
            rank_path = f"{answer_file}.rank{rank}.jsonl"
            with open(rank_path, encoding="utf-8") as handle:
                merged.extend(json.loads(line) for line in handle if line.strip())
            os.remove(rank_path)

    merged.sort(key=lambda item: item["question_id"])
    with open(answer_file, "w", encoding="utf-8") as handle:
        for answer in merged:
            handle.write(json.dumps(answer, ensure_ascii=False) + "\n")
    print(f"Saved answer file with {len(merged)} samples to {answer_file}")
    return merged


def _print_stats(answers: list[dict], block_size: int) -> None:
    choices = list(chain.from_iterable(answer["choices"] for answer in answers))
    by_mode = {}
    for choice in choices:
        by_mode.setdefault(choice["mode"], []).append(choice)

    per_mode = {}
    for mode, mode_choices in by_mode.items():
        token_times = list(
            chain.from_iterable(choice["time_per_output_token"] for choice in mode_choices)
        )
        wall_times = list(chain.from_iterable(choice["wall_time"] for choice in mode_choices))
        peaks = list(chain.from_iterable(choice["peak_memory_mb"] for choice in mode_choices))
        per_mode[mode] = statistics.median(token_times) if token_times else 0.0
        print(
            f"{mode}: median_decode_ms_per_token={per_mode[mode] * 1000:.3f} "
            f"median_wall_s={statistics.median(wall_times):.3f} "
            f"peak_memory_mib={max(peaks):.1f}"
        )

    if per_mode.get("baseline", 0.0) and per_mode.get("domino", 0.0):
        print(f"Decoding speedup: {per_mode['baseline'] / per_mode['domino']:.3f}x")

    domino_choices = by_mode.get("domino", [])
    acceptance = list(
        chain.from_iterable(
            chain.from_iterable(choice["acceptance_lengths"] for choice in domino_choices)
        )
    )
    if acceptance:
        print(f"Average acceptance length: {np.mean(acceptance):.3f}")
        histogram = [
            acceptance.count(value) / len(acceptance)
            for value in range(block_size + 2)
        ]
        print(f"Acceptance histogram: {[f'{x * 100:.1f}%' for x in histogram]}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--model-revision", type=str, default=None)
    parser.add_argument("--draft-name-or-path", type=str, default=None)
    parser.add_argument("--draft-revision", type=str, default=None)
    parser.add_argument("--mode", choices=["baseline", "domino", "compare"], default="compare")
    parser.add_argument("--device-backend", choices=ACCELERATOR_BACKENDS, default="auto")
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--benchmark-manifest", type=str, default=None)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--target-dtype",
        choices=TARGET_DTYPES,
        default="bfloat16",
        help="Target model weight dtype (default: bfloat16); draft remains bfloat16.",
    )
    parser.add_argument("--warmup-samples", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--use-graph", action="store_true")
    parser.add_argument("--use-bias", action="store_true")
    parser.add_argument("--dump-benchmark-manifest", type=str, default=None)
    parser.add_argument("--dump-only", action="store_true")
    parser.add_argument("--answer-file", type=str, default=None)
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default=None,
    )
    args = parser.parse_args()

    if not args.dataset and not args.benchmark_manifest:
        parser.error("Provide --dataset or --benchmark-manifest.")
    if not args.dump_only and args.model_name_or_path is None:
        parser.error("--model-name-or-path is required unless --dump-only is set.")
    if args.mode != "baseline" and not args.dump_only and not args.draft_name_or_path:
        parser.error("Domino mode requires --draft-name-or-path.")
    if args.repetitions < 1 or args.warmup_samples < 0:
        parser.error("--repetitions must be >=1 and --warmup-samples must be >=0.")
    instances = _load_instances(args)
    if args.dump_benchmark_manifest:
        _dump_manifest(args.dump_benchmark_manifest, instances)
    if args.dump_only:
        return

    backend = resolve_backend(args.device_backend)
    device = get_device(local_rank=dist.local_rank(), backend=backend)
    set_device(device)
    dist.init(backend=backend)
    random.seed(0)
    np.random.seed(0)
    manual_seed_all(0, backend=backend)
    if backend == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if args.use_graph and backend != "cuda":
        raise ValueError("--use-graph currently supports CUDA only; disable it on NPU.")

    if args.attn_implementation:
        attn_impl = args.attn_implementation
    elif backend == "cuda" and is_flash_attn_2_available():
        attn_impl = "flash_attention_2"
    else:
        attn_impl = "sdpa"
    target_dtype = TARGET_DTYPES[args.target_dtype]
    logger.info(
        f"backend={backend} device={device} attention={attn_impl} "
        f"target_dtype={args.target_dtype}"
    )

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        revision=args.model_revision,
        attn_implementation=attn_impl,
        torch_dtype=target_dtype,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, revision=args.model_revision
    )

    draft_model = None
    graph_runner = None
    block_size = int(args.block_size or 16)
    if args.mode != "baseline":
        draft_model = load_draft_model_for_benchmark(
            args.draft_name_or_path, attn_impl, args.draft_revision
        ).to(device).eval()
        block_size = int(args.block_size or draft_model.block_size)
        if not is_domino_projector(draft_model.projector_type):
            raise ValueError(f"Unsupported projector_type={draft_model.projector_type!r}")
        if not args.use_bias:
            raise ValueError("Domino mode requires --use-bias.")
        if args.use_graph:
            from kernel.domino import DraftCorrectionGraphRunner

            shift_label = bool(draft_model.config.dflash_config.get("shift_label", False))
            steps = (block_size if shift_label else block_size - 1) - int(
                draft_model.pure_draft_prefix_len
            )
            graph_runner = DraftCorrectionGraphRunner(
                draft_model=draft_model,
                target_model=target,
                batch_size=1,
                steps=steps,
                hidden_dim=int(target.lm_head.weight.shape[1]),
                gru_hidden_dim=draft_model.prefix_gru.hidden_size,
                vocab_size=int(target.lm_head.weight.shape[0]),
                prefix_token_count=1 + int(draft_model.pure_draft_prefix_len),
                device=device,
            )

    def run(mode, input_ids, max_new_tokens):
        kwargs = dict(
            input_ids=input_ids,
            target=target,
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
            stop_token_ids=[tokenizer.eos_token_id],
            return_dict=True,
        )
        if mode == "baseline":
            return target_greedy_generate(**kwargs)
        return draft_model.spec_generate(
            **kwargs,
            block_size=block_size,
            graph_runner=graph_runner,
            use_bias=args.use_bias,
        )

    modes = [args.mode] if args.mode != "compare" else ["baseline", "domino"]
    if args.warmup_samples:
        warmup_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Write one short sentence about inference."}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        warmup_ids = tokenizer.encode(warmup_text, return_tensors="pt").to(device)
        for _ in range(args.warmup_samples):
            for mode in modes:
                run(mode, warmup_ids, min(32, args.max_new_tokens))

    answers = []
    indices = range(dist.rank(), len(instances), dist.size())
    for index in tqdm(indices, disable=not dist.is_main()):
        instance = instances[index]
        histories = {mode: [] for mode in modes}
        choices = [
            _new_choice(position, mode, 1 if mode == "baseline" else block_size)
            for position, mode in enumerate(modes)
        ]
        for user_content in instance["turns"]:
            for mode, choice in zip(modes, choices):
                histories[mode].append({"role": "user", "content": user_content})
                prompt = tokenizer.apply_chat_template(
                    histories[mode],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                responses = [
                    run(mode, input_ids, args.max_new_tokens)
                    for _ in range(args.repetitions)
                ]
                output_text = _record_choice(choice, responses, tokenizer)
                histories[mode].append({"role": "assistant", "content": output_text})

        answers.append(
            {
                "question_id": int(instance.get("question_id", index)),
                "choices": choices,
                "tstamp": time.time(),
            }
        )

    answers = _write_and_merge_answers(args.answer_file, answers)
    if answers is None:
        return
    answers.sort(key=lambda item: item["question_id"])
    _print_stats(answers, block_size)


if __name__ == "__main__":
    main()
