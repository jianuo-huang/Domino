from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

from benchmark_sglang import (
    BenchMetrics,
    DEFAULT_DRAFT_MODEL_DFLASH,
    DEFAULT_TARGET_MODEL_DFLASH,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    _flush_cache,
    _format_table,
    _is_blackwell,
    _run_bench_requests,
    _send_generate,
    find_available_port,
    popen_launch_server,
)
from model import load_and_process_dataset
from sglang.srt.utils import get_device_sm, kill_process_tree


DEFAULT_TASKS = (
    "gsm8k:128,math500:128,humaneval:164,mbpp:128,"
    "livecodebench:128,alpaca:128"
)


def _parse_tasks(spec: str) -> list[tuple[str, int]]:
    tasks: list[tuple[str, int]] = []
    for raw in spec.replace("\n", ",").split(","):
        raw = raw.strip()
        if not raw:
            continue
        if ":" not in raw:
            raise ValueError(f"Task entry must be dataset:count, got {raw!r}")
        name, count_s = raw.split(":", 1)
        name = name.strip()
        count = int(count_s.strip())
        if not name or count <= 0:
            raise ValueError(f"Invalid task entry: {raw!r}")
        tasks.append((name, count))
    if not tasks:
        raise ValueError("No tasks specified.")
    return tasks


def _build_prompts(
    *,
    dataset_name: str,
    target_model: str,
    tokenizer: AutoTokenizer,
    required_questions: int,
) -> list[str]:
    dataset = load_and_process_dataset(dataset_name)
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset {dataset_name!r} is empty.")
    if len(dataset) < required_questions:
        print(
            f"Warning: dataset={dataset_name} has {len(dataset)} items, "
            f"but need {required_questions}; reusing items."
        )

    prompts: list[str] = []
    prompt_count = min(len(dataset), required_questions)
    for i in range(prompt_count):
        item = dataset[i]
        user_content = item["turns"][0]
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(prompt_text)
    return prompts


def _cycle_take(pool: list[str], n: int) -> list[str]:
    if n <= 0:
        return []
    if not pool:
        raise RuntimeError("Cannot take prompts from an empty pool.")
    return [pool[i % len(pool)] for i in range(n)]


def _compose_run_prompts(
    prompts: list[str],
    *,
    timed_count: int,
    warmup_count: int,
    skip_count: int,
) -> tuple[list[str], str]:
    """Return prompts in the order consumed by `_run_bench_requests`.

    Timed prompts are always the first `timed_count` prompts, so each
    concurrency measures the same task set. Warmup/skip prompts prefer the
    remaining dataset prompts and cycle within that pool. If the timed set
    consumes the whole dataset, warmup falls back to cycling timed prompts.
    """
    if timed_count <= 0:
        raise ValueError(f"timed_count must be positive, got {timed_count}.")
    if not prompts:
        raise RuntimeError("No prompts available.")

    timed = _cycle_take(prompts, timed_count)
    warmup_pool = prompts[timed_count:]
    warmup_source = "post_timed"
    if not warmup_pool:
        warmup_pool = timed
        warmup_source = "timed_reuse"

    warmup = _cycle_take(warmup_pool, warmup_count)
    skipped = _cycle_take(warmup_pool, skip_count)
    return warmup + skipped + timed, warmup_source


def _server_args(args: argparse.Namespace, backend: str) -> list[str]:
    common_server_args: list[str] = [
        "--trust-remote-code",
        "--attention-backend",
        backend,
        "--tp-size",
        str(args.tp_size),
        "--dtype",
        str(args.dtype),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--max-running-requests",
        str(args.max_running_requests),
        "--page-size",
        str(args.page_size),
    ]
    if args.disable_cuda_graph:
        common_server_args.extend(["--disable-cuda-graph", "--disable-piecewise-cuda-graph"])
    else:
        common_server_args.extend(
            [
                "--cuda-graph-bs",
                *[str(i) for i in range(1, int(args.cuda_graph_max_bs) + 1)],
                "--cuda-graph-max-bs",
                str(args.cuda_graph_max_bs),
            ]
        )
    if args.disable_radix_cache:
        common_server_args.append("--disable-radix-cache")
    if args.mode == "dflash":
        common_server_args.extend(
            [
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-draft-model-path",
                args.draft_model,
            ]
        )
    elif args.mode == "eagle":
        common_server_args.extend(
            [
                "--speculative-algorithm",
                args.speculative_algorithm,
                "--speculative-draft-model-path",
                args.draft_model,
                "--speculative-num-steps",
                str(args.speculative_num_steps),
                "--speculative-eagle-topk",
                str(args.speculative_eagle_topk),
                "--speculative-num-draft-tokens",
                str(args.speculative_num_draft_tokens),
            ]
        )
    return common_server_args


def _write_outputs(
    *,
    args: argparse.Namespace,
    tasks: list[tuple[str, int]],
    concurrencies: list[int],
    backend: str,
    device_sm: int,
    is_blackwell: bool,
    results: dict[tuple[str, int], BenchMetrics],
    output_md: Optional[str],
    output_jsonl: Optional[str],
) -> None:
    if output_jsonl:
        out_path = Path(output_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for dataset_name, count in tasks:
                for conc in concurrencies:
                    m = results.get((dataset_name, conc))
                    if m is None:
                        continue
                    rec = {
                        "mode": args.mode,
                        "backend": backend,
                        "dataset": dataset_name,
                        "num_questions": count,
                        "concurrency": conc,
                        "target_model": args.target_model,
                        "draft_model": args.draft_model if args.mode != "baseline" else None,
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "cuda_graph_max_bs": args.cuda_graph_max_bs,
                        "max_running_requests": args.max_running_requests,
                        **asdict(m),
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote jsonl report to: {out_path}")

    if not output_md:
        return

    md: list[str] = []
    title = {"baseline": "Baseline", "dflash": "DFLASH", "eagle": "EAGLE"}[args.mode]
    md.append(f"# {title} Multi-Task SGLang Bench Report")
    md.append("")
    md.append("## Settings")
    md.append(f"- mode: `{args.mode}`")
    md.append(f"- target_model: `{args.target_model}`")
    if args.mode != "baseline":
        md.append(f"- draft_model: `{args.draft_model}`")
    if args.mode == "eagle":
        md.append(f"- speculative_algorithm: `{args.speculative_algorithm}`")
        md.append(f"- speculative_num_steps: `{args.speculative_num_steps}`")
        md.append(f"- speculative_eagle_topk: `{args.speculative_eagle_topk}`")
        md.append(f"- speculative_num_draft_tokens: `{args.speculative_num_draft_tokens}`")
    md.append(f"- max_new_tokens: `{args.max_new_tokens}`")
    md.append(f"- temperature: `{args.temperature}`")
    md.append(f"- top_p: `{args.top_p}`")
    md.append(f"- top_k: `{args.top_k}`")
    md.append(f"- attention_backend: `{backend}`")
    md.append(f"- tp_size: `{args.tp_size}`")
    md.append(f"- concurrencies: `{', '.join(str(c) for c in concurrencies)}`")
    md.append(f"- device_sm: `{device_sm}`")
    md.append(f"- is_blackwell: `{is_blackwell}`")
    md.append(f"- disable_cuda_graph: `{bool(args.disable_cuda_graph)}`")
    md.append(f"- cuda_graph_max_bs: `{args.cuda_graph_max_bs}`")
    md.append(f"- max_running_requests: `{args.max_running_requests}`")
    md.append(f"- page_size: `{args.page_size}`")
    md.append(f"- warmup_requests: `{args.warmup_requests}`")
    md.append(f"- warmup_max_new_tokens: `{args.warmup_max_new_tokens}`")
    md.append(f"- skip_first_requests: `{args.skip_first_requests}`")
    md.append("- timed_prompts: `fixed_first_n_per_dataset`")
    md.append("- warmup_prompts: `post_timed_cycle_or_timed_reuse_if_no_extra_prompts`")
    md.append("")
    md.append("## Tasks")
    for dataset_name, count in tasks:
        md.append(f"- `{dataset_name}`: `{count}`")
    md.append("")

    md.append("## Output tok/s")
    md.append("")
    for dataset_name, _ in tasks:
        md.append(f"### `{dataset_name}`")
        values = {
            c: (results[(dataset_name, c)].output_toks_per_s if (dataset_name, c) in results else None)
            for c in concurrencies
        }
        md.append(_format_table(concurrencies=concurrencies, values=values, float_fmt=",.2f"))
        md.append("")

    if args.mode != "baseline":
        md.append(f"## {title} acceptance length")
        md.append("")
        for dataset_name, _ in tasks:
            md.append(f"### `{dataset_name}`")
            values = {
                c: (results[(dataset_name, c)].spec_accept_length if (dataset_name, c) in results else None)
                for c in concurrencies
            }
            md.append(_format_table(concurrencies=concurrencies, values=values, float_fmt=".3f"))
            md.append("")

    out_path = Path(output_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote markdown report to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "dflash", "eagle"], required=True)
    parser.add_argument("--tasks", type=str, default=DEFAULT_TASKS)
    parser.add_argument("--target-model", type=str, default=DEFAULT_TARGET_MODEL_DFLASH)
    parser.add_argument("--draft-model", type=str, default=DEFAULT_DRAFT_MODEL_DFLASH)
    parser.add_argument("--speculative-algorithm", type=str, default="EAGLE3")
    parser.add_argument("--speculative-num-steps", type=int, default=7)
    parser.add_argument("--speculative-eagle-topk", type=int, default=10)
    parser.add_argument("--speculative-num-draft-tokens", type=int, default=16)
    parser.add_argument("--output-md", type=str, default=None)
    parser.add_argument("--output-jsonl", type=str, default=None)
    parser.add_argument("--batch-requests", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--warmup-requests", type=int, default=3)
    parser.add_argument("--warmup-max-new-tokens", type=int, default=1024)
    parser.add_argument("--skip-first-requests", type=int, default=1)
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--disable-radix-cache", action="store_true")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max-running-requests", type=int, default=64)
    parser.add_argument("--cuda-graph-max-bs", type=int, default=64)
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--concurrencies", type=str, default="1,2,4,8,16,32,64")
    parser.add_argument("--attention-backend", type=str, default="flashinfer")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    tasks = _parse_tasks(args.tasks)
    concurrencies = [int(x) for x in args.concurrencies.split(",") if x.strip()]
    concurrencies = [c for c in concurrencies if c >= 1]
    if not concurrencies:
        raise RuntimeError("No concurrencies specified.")

    max_conc = max(concurrencies)
    extra_per_run = int(args.warmup_requests) * max_conc + int(args.skip_first_requests)
    max_count = max(count for _, count in tasks)
    required_questions = max_count + extra_per_run

    device_sm = get_device_sm()
    is_blackwell = _is_blackwell()
    backend = args.attention_backend
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)

    print("Loading prompts for all tasks...")
    prompts_by_task = {
        dataset_name: _build_prompts(
            dataset_name=dataset_name,
            target_model=args.target_model,
            tokenizer=tokenizer,
            required_questions=required_questions,
        )
        for dataset_name, _ in tasks
    }

    port = find_available_port(20000)
    url = f"http://127.0.0.1:{port}"
    server_args = _server_args(args, backend)
    print(f"Launching {args.mode} server once at {url}")
    proc = popen_launch_server(
        args.target_model,
        url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=server_args,
    )

    results: dict[tuple[str, int], BenchMetrics] = {}
    try:
        _send_generate(
            url,
            "Hello",
            max_new_tokens=8,
            stop=[],
            timeout_s=min(int(args.timeout_s), 300),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            top_k=int(args.top_k),
        )

        for dataset_name, count in tasks:
            prompts = prompts_by_task[dataset_name]
            print(f"\n=== TASK {dataset_name} n={count} ===")
            for conc in concurrencies:
                _flush_cache(url)
                warmup_n = int(args.warmup_requests) * int(conc)
                skip_n = int(args.skip_first_requests)
                prompt_slice, warmup_source = _compose_run_prompts(
                    prompts,
                    timed_count=count,
                    warmup_count=warmup_n,
                    skip_count=skip_n,
                )
                print(
                    f"[run] dataset={dataset_name} conc={conc} n={count} "
                    f"warmup_batches={args.warmup_requests} temp={args.temperature} "
                    f"timed_prompts=fixed warmup_source={warmup_source}"
                )
                metrics = _run_bench_requests(
                    url,
                    prompts=prompt_slice,
                    max_new_tokens=int(args.max_new_tokens),
                    concurrency=int(conc),
                    batch_requests=bool(args.batch_requests),
                    stop=[],
                    timeout_s=int(args.timeout_s),
                    expect_dflash=args.mode != "baseline",
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    top_k=int(args.top_k),
                    warmup_requests=int(args.warmup_requests),
                    warmup_max_new_tokens=int(args.warmup_max_new_tokens),
                    skip_first_requests=int(args.skip_first_requests),
                )
                results[(dataset_name, conc)] = metrics
                msg = (
                    f"[{args.mode}] dataset={dataset_name} conc={conc:>2} n={count:<4} "
                    f"tok/s={metrics.output_toks_per_s:,.2f} latency={metrics.latency_s:.1f}s"
                )
                if metrics.spec_accept_length is not None:
                    msg += f" accept_len={metrics.spec_accept_length:.3f}"
                print(msg)
                _write_outputs(
                    args=args,
                    tasks=tasks,
                    concurrencies=concurrencies,
                    backend=backend,
                    device_sm=device_sm,
                    is_blackwell=is_blackwell,
                    results=results,
                    output_md=args.output_md,
                    output_jsonl=args.output_jsonl,
                )
    finally:
        kill_process_tree(proc.pid)
        try:
            proc.wait(timeout=30)
        except Exception:
            pass


if __name__ == "__main__":
    main()
