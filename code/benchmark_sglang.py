from __future__ import annotations

import argparse
import os
import random
import shlex
import socket
import subprocess
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import requests
import torch
from transformers import AutoTokenizer
from model import load_and_process_dataset

from sglang.srt.utils import get_device_sm, kill_process_tree


DEFAULT_TARGET_MODEL_DFLASH = "Qwen/Qwen3-8B"
DEFAULT_DRAFT_MODEL_DFLASH = ""
DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH = 3600


def is_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", int(port))) != 0


def find_available_port(base_port: int) -> int:
    port = int(base_port) + random.randint(100, 1000)
    while True:
        if is_port_available(port):
            return port
        port = port + 42 if port < 60000 else port - 43


def popen_launch_server(model: str, base_url: str, timeout: float, other_args: Optional[list[str]] = None):
    other_args = other_args or []
    if not base_url.startswith("http://"):
        raise ValueError(f"base_url must start with http://, got {base_url!r}")
    host_port = base_url[len("http://"):]
    host, port_s = host_port.rsplit(":", 1)
    command = [
        "sglang",
        "serve",
        "--model-path",
        model,
        *[str(x) for x in other_args],
        "--host",
        host,
        "--port",
        port_s,
    ]
    print(f"command={shlex.join(command)}")
    process = subprocess.Popen(command, env=os.environ.copy())
    deadline = time.time() + float(timeout)
    last_error = None
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"SGLang server exited early with code {process.returncode}.")
        try:
            resp = requests.get(base_url + "/health", timeout=5)
            if resp.status_code == 200:
                return process
            last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(2)
    try:
        kill_process_tree(process.pid)
    except Exception:
        pass
    raise TimeoutError(f"Timed out waiting for SGLang server at {base_url}; last_error={last_error}")


def _is_blackwell() -> bool:
    return False


def _flush_cache(base_url: str) -> None:
    last_error: Exception | None = None
    for i in range(5):
        try:
            resp = requests.get(base_url + "/flush_cache", timeout=60)
            resp.raise_for_status()
            return
        except requests.HTTPError as exc:
            last_error = exc
            if exc.response is None or exc.response.status_code != 400:
                raise
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(1.0 + i)
    print(f"Warning: /flush_cache failed after retries. Continuing. error={last_error}")


def _send_generate(
    base_url: str,
    prompt: str,
    *,
    max_new_tokens: int,
    stop: list[str],
    timeout_s: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 1,
) -> dict:
    sampling_params: dict = {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "max_new_tokens": int(max_new_tokens),
    }
    if stop:
        sampling_params["stop"] = stop
    resp = requests.post(
        base_url + "/generate",
        json={
            "text": prompt,
            "sampling_params": sampling_params,
        },
        timeout=int(timeout_s),
    )
    resp.raise_for_status()
    return resp.json()


def _send_generate_batch(
    base_url: str,
    prompts: list[str],
    *,
    max_new_tokens: int,
    stop: list[str],
    timeout_s: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 1,
) -> list[dict]:
    if not prompts:
        return []
    sampling_params: dict = {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "max_new_tokens": int(max_new_tokens),
    }
    if stop:
        sampling_params["stop"] = stop
    resp = requests.post(
        base_url + "/generate",
        json={
            "text": prompts,
            "sampling_params": sampling_params,
        },
        timeout=int(timeout_s),
    )
    resp.raise_for_status()
    out = resp.json()
    if not isinstance(out, list):
        raise RuntimeError(
            "Expected a list response for batched /generate, but got "
            f"type={type(out).__name__}."
        )
    return out


@dataclass(frozen=True)
class BenchMetrics:
    latency_s: float
    output_tokens: int
    output_toks_per_s: float
    spec_accept_length: Optional[float]
    spec_verify_ct_sum: int


def _run_bench_requests(
    base_url: str,
    *,
    prompts: list[str],
    max_new_tokens: int,
    concurrency: int,
    batch_requests: bool,
    stop: list[str],
    timeout_s: int,
    expect_dflash: bool,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 1,
    warmup_requests: int = 0,
    warmup_max_new_tokens: int = 0,
    skip_first_requests: int = 0,
) -> BenchMetrics:
    """Run a metric collection sweep with explicit warmup + skip controls.

    The warmup phase issues `warmup_requests` requests at the same concurrency
    as the timed phase, with `warmup_max_new_tokens` so each request actually
    walks the full DFLASH decode path long enough for graphs / kernels to
    reach steady state. Warmup output is discarded.

    `skip_first_requests` further drops the first N requests of the timed
    phase from the metrics aggregation (useful when the very first batch in
    steady state still pays one-time costs like first-time draft KV pool
    growth or radix cache layout shuffles).
    """
    bs = max(int(concurrency), 1)

    warmup_n = max(int(warmup_requests), 0) * bs
    if warmup_n > 0 and len(prompts) > warmup_n:
        warmup_prompts = prompts[:warmup_n]
        warmup_mnt = int(warmup_max_new_tokens) if warmup_max_new_tokens else int(max_new_tokens)
        if batch_requests:
            for start_idx in range(0, warmup_n, bs):
                _send_generate_batch(
                    base_url,
                    warmup_prompts[start_idx : start_idx + bs],
                    max_new_tokens=warmup_mnt,
                    stop=stop,
                    timeout_s=timeout_s,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
        else:
            with ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
                futures = [
                    pool.submit(
                        _send_generate,
                        base_url,
                        prompt,
                        max_new_tokens=warmup_mnt,
                        stop=stop,
                        timeout_s=timeout_s,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                    )
                    for prompt in warmup_prompts
                ]
                for fut in as_completed(futures):
                    fut.result()
        prompts = prompts[warmup_n:]

    # Optionally drop additional first-N timed requests after warmup. We still
    # issue them (so the server's request stream looks identical) but exclude
    # them from the timed window and metric aggregation.
    skip_n = max(int(skip_first_requests), 0)
    if skip_n > 0 and len(prompts) > skip_n:
        skip_prompts = prompts[:skip_n]
        if batch_requests:
            for start_idx in range(0, skip_n, bs):
                _send_generate_batch(
                    base_url,
                    skip_prompts[start_idx : start_idx + bs],
                    max_new_tokens=max_new_tokens,
                    stop=stop,
                    timeout_s=timeout_s,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
        else:
            with ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
                futures = [
                    pool.submit(
                        _send_generate,
                        base_url,
                        prompt,
                        max_new_tokens=max_new_tokens,
                        stop=stop,
                        timeout_s=timeout_s,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                    )
                    for prompt in skip_prompts
                ]
                for fut in as_completed(futures):
                    fut.result()
        prompts = prompts[skip_n:]

    start = time.perf_counter()
    total_tokens = 0
    spec_verify_ct_sum = 0
    spec_accept_lengths: list[float] = []

    if batch_requests:
        bs = max(int(concurrency), 1)
        for start_idx in range(0, len(prompts), bs):
            chunk_prompts = prompts[start_idx : start_idx + bs]
            outs = _send_generate_batch(
                base_url,
                chunk_prompts,
                max_new_tokens=max_new_tokens,
                stop=stop,
                timeout_s=timeout_s,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            if len(outs) != len(chunk_prompts):
                raise RuntimeError(
                    "Batched /generate output length mismatch: "
                    f"got {len(outs)} outputs for {len(chunk_prompts)} prompts."
                )

            for j, out in enumerate(outs):
                meta = out.get("meta_info", {}) or {}
                total_tokens += int(meta.get("completion_tokens", 0))
                spec_verify_ct_sum += int(meta.get("spec_verify_ct", 0))
                if "spec_accept_length" in meta:
                    try:
                        spec_accept_lengths.append(float(meta["spec_accept_length"]))
                    except (TypeError, ValueError):
                        pass
    else:
        with ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
            futures = {
                pool.submit(
                    _send_generate,
                    base_url,
                    prompt,
                    max_new_tokens=max_new_tokens,
                    stop=stop,
                    timeout_s=timeout_s,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                ): i
                for i, prompt in enumerate(prompts)
            }
            for fut in as_completed(futures):
                out = fut.result()
                meta = out.get("meta_info", {}) or {}
                total_tokens += int(meta.get("completion_tokens", 0))
                spec_verify_ct_sum += int(meta.get("spec_verify_ct", 0))
                if "spec_accept_length" in meta:
                    try:
                        spec_accept_lengths.append(float(meta["spec_accept_length"]))
                    except (TypeError, ValueError):
                        pass

    latency = time.perf_counter() - start
    toks_per_s = total_tokens / max(latency, 1e-6)

    if expect_dflash and spec_verify_ct_sum <= 0:
        raise RuntimeError(
            "DFLASH sanity check failed: did not observe any `spec_verify_ct` in responses "
            "(DFLASH may not have been enabled)."
        )

    spec_accept_length = (
        float(statistics.mean(spec_accept_lengths)) if spec_accept_lengths else None
    )

    return BenchMetrics(
        latency_s=float(latency),
        output_tokens=int(total_tokens),
        output_toks_per_s=float(toks_per_s),
        spec_accept_length=spec_accept_length,
        spec_verify_ct_sum=int(spec_verify_ct_sum),
    )


def _format_table(
    *,
    concurrencies: list[int],
    values: dict[int, Optional[float]],
    float_fmt: str,
) -> str:
    header = ["conc"] + [str(c) for c in concurrencies]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    row = ["value"]
    for c in concurrencies:
        v = values.get(c, None)
        row.append("N/A" if v is None else format(v, float_fmt))
    lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Write a markdown report to this file (disabled by default).",
    )
    parser.add_argument("--dataset-name", type=str, default="gsm8k")
    parser.add_argument("--target-model", type=str, default=DEFAULT_TARGET_MODEL_DFLASH)
    parser.add_argument("--draft-model", type=str, default=DEFAULT_DRAFT_MODEL_DFLASH)
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip running the baseline (target-only) sweep; only run DFLASH and report N/A for baseline/speedup.",
    )
    parser.add_argument(
        "--batch-requests",
        action="store_true",
        help="Send prompts as server-side batched /generate requests (batch size = concurrency) instead of client-side concurrent requests.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=3,
        help="Number of warmup *batches* (each of size=concurrency) to send "
             "after /flush_cache. Each warmup request walks the full DFLASH "
             "decode path. Output is discarded.",
    )
    parser.add_argument(
        "--warmup-max-new-tokens",
        type=int,
        default=1024,
        help="max_new_tokens used during warmup. Should be large enough that "
             "warmup requests actually exercise sustained decode (not just "
             "prefill / first few tokens).",
    )
    parser.add_argument(
        "--skip-first-requests",
        type=int,
        default=1,
        help="After warmup, drop the first N *timed* requests from metrics "
             "(they are still issued so the request stream is unchanged).",
    )
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--disable-radix-cache", action="store_true")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max-running-requests", type=int, default=128)
    parser.add_argument(
        "--cuda-graph-max-bs",
        type=int,
        default=64,
        help="Capture CUDA graph batch sizes from 1..N when CUDA graph is enabled.",
    )
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size (single value, no sweep).",
    )
    parser.add_argument(
        "--concurrencies",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma-separated list of client concurrency levels.",
    )
    parser.add_argument(
        "--questions-per-concurrency-base",
        type=int,
        default=128,
        help="num_questions = base * concurrency (default matches the sweep plan).",
    )
    parser.add_argument(
        "--max-questions-per-config",
        type=int,
        default=1024,
        help="Cap num_questions per (tp, concurrency) run (default: 1024).",
    )
    parser.add_argument(
        "--attention-backends",
        type=str,
        default="flashinfer,fa3,fa4",
        help="Comma-separated list. Will auto-skip fa3 unless SM90 (Hopper), and fa4 unless SM100+ (Blackwell).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this sweep.")

    concurrencies = [int(x) for x in args.concurrencies.split(",") if x.strip()]
    concurrencies = [c for c in concurrencies if c >= 1]
    if not concurrencies:
        raise RuntimeError("No concurrencies specified.")

    num_questions_by_conc = {
        c: min(int(args.questions_per_concurrency_base) * int(c), int(args.max_questions_per_config))
        for c in concurrencies
    }
    max_questions = max(num_questions_by_conc.values())
    max_concurrency = max(concurrencies)

    attention_backends = [s.strip() for s in args.attention_backends.split(",") if s.strip()]
    is_blackwell = _is_blackwell()
    device_sm = get_device_sm()
    if device_sm != 90:
        attention_backends = [b for b in attention_backends if b != "fa3"]
    if device_sm < 100:
        attention_backends = [b for b in attention_backends if b != "fa4"]
    attention_backends = attention_backends or ["flashinfer"]

    # --- Load Data using the new function ---
    print(f"Loading dataset: {args.dataset_name}...")
    dataset = load_and_process_dataset(args.dataset_name)
    # Each conc run consumes:  warmup_requests*conc (warmup, dropped) +
    # skip_first_requests (issued+dropped) + n (timed). Size for max_concurrency.
    extra_per_run = (
        max(int(args.warmup_requests), 0) * max_concurrency
        + max(int(args.skip_first_requests), 0)
    )
    required_questions = max_questions + max_concurrency + extra_per_run
    
    if len(dataset) < required_questions:
        print(f"Warning: Dataset has {len(dataset)} items, but need up to {required_questions}. Reusing items.")

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)

    prompts: list[str] = []
    # Build prompts list
    for i in range(max(len(dataset), required_questions)):
        item = dataset[i % len(dataset)]
        user_content = item["turns"][0] # Extract the formatted turn
        
        # Apply chat template
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(prompt_text)
        if len(prompts) >= required_questions:
            break

    # Results indexed by (backend, concurrency) for baseline + dflash.
    # Removed TP dimension from keys since we aren't sweeping it.
    baseline_toks: dict[tuple[str, int], Optional[float]] = {}
    dflash_toks: dict[tuple[str, int], Optional[float]] = {}
    dflash_accept_len: dict[tuple[str, int], Optional[float]] = {}
    
    tp = args.tp_size  # Fixed TP size

    for backend in attention_backends:
        port_base = find_available_port(20000)

        common_server_args: list[str] = [
            "--trust-remote-code",
            "--attention-backend",
            backend,
            "--tp-size",
            str(tp),
            "--dtype",
            str(args.dtype),
            "--mem-fraction-static",
            str(args.mem_fraction_static),
            "--max-running-requests",
            str(args.max_running_requests),
            "--page-size",
            str(args.page_size),
        ]
        if not args.disable_cuda_graph:
            common_server_args.extend(
                [
                    "--cuda-graph-bs",
                    *[str(i) for i in range(1, int(args.cuda_graph_max_bs) + 1)],
                    "--cuda-graph-max-bs",
                    str(args.cuda_graph_max_bs),
                ]
            )
        else:
            common_server_args.append("--disable-cuda-graph")
            common_server_args.append("--disable-piecewise-cuda-graph")
        if args.disable_radix_cache:
            common_server_args.append("--disable-radix-cache")

        if not args.skip_baseline:
            print(f"\n=== backend={backend} tp={tp} (baseline) ===")
            baseline_port = port_base
            baseline_url = f"http://127.0.0.1:{baseline_port}"
            baseline_proc = popen_launch_server(
                args.target_model,
                baseline_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=common_server_args,
            )
            try:
                # Warm up.
                _send_generate(
                    baseline_url,
                    "Hello",
                    max_new_tokens=8,
                    stop=[],
                    timeout_s=min(int(args.timeout_s), 300),
                )

                for conc in concurrencies:
                    n = num_questions_by_conc[conc]
                    _flush_cache(baseline_url)
                    extra = args.warmup_requests * conc + args.skip_first_requests
                    print(
                        f"[warmup] run {args.warmup_requests} warmup batch(es) of size {conc} "
                        f"(max_new_tokens={args.warmup_max_new_tokens}) + skip first "
                        f"{args.skip_first_requests} timed request(s)."
                    )
                    metrics = _run_bench_requests(
                        baseline_url,
                        prompts=prompts[: n + extra],
                        max_new_tokens=int(args.max_new_tokens),
                        concurrency=int(conc),
                        batch_requests=bool(args.batch_requests),
                        stop=[],
                        timeout_s=int(args.timeout_s),
                        expect_dflash=False,
                        warmup_requests=int(args.warmup_requests),
                        warmup_max_new_tokens=int(args.warmup_max_new_tokens),
                        skip_first_requests=int(args.skip_first_requests),
                    )
                    baseline_toks[(backend, conc)] = metrics.output_toks_per_s
                    print(
                        f"[baseline] conc={conc:>2} n={n:<4} "
                        f"toks/s={metrics.output_toks_per_s:,.2f} "
                        f"latency={metrics.latency_s:.1f}s "
                    )
            finally:
                kill_process_tree(baseline_proc.pid)
                try:
                    baseline_proc.wait(timeout=30)
                except Exception:
                    pass

        print(f"\n=== backend={backend} tp={tp} (DFLASH) ===")
        dflash_port = find_available_port(port_base + 1)
        dflash_url = f"http://127.0.0.1:{dflash_port}"
        dflash_proc = popen_launch_server(
            args.target_model,
            dflash_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                *common_server_args,
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-draft-model-path",
                args.draft_model,
            ],
        )
        try:
            _send_generate(
                dflash_url,
                "Hello",
                max_new_tokens=8,
                stop=[],
                timeout_s=min(int(args.timeout_s), 300),
            )
            for conc in concurrencies:
                n = num_questions_by_conc[conc]
                _flush_cache(dflash_url)
                extra = args.warmup_requests * conc + args.skip_first_requests
                print(
                    f"[warmup] run {args.warmup_requests} warmup batch(es) of size {conc} "
                    f"(max_new_tokens={args.warmup_max_new_tokens}) + skip first "
                    f"{args.skip_first_requests} timed request(s)."
                )
                metrics = _run_bench_requests(
                    dflash_url,
                    prompts=prompts[: n + extra],
                    max_new_tokens=int(args.max_new_tokens),
                    concurrency=int(conc),
                    batch_requests=bool(args.batch_requests),
                    stop=[],
                    timeout_s=int(args.timeout_s),
                    expect_dflash=True,
                    warmup_requests=int(args.warmup_requests),
                    warmup_max_new_tokens=int(args.warmup_max_new_tokens),
                    skip_first_requests=int(args.skip_first_requests),
                )
                dflash_toks[(backend, conc)] = metrics.output_toks_per_s
                dflash_accept_len[(backend, conc)] = metrics.spec_accept_length
                print(
                    f"[DFLASH]   conc={conc:>2} n={n:<4} "
                    f"toks/s={metrics.output_toks_per_s:,.2f} "
                    f"latency={metrics.latency_s:.1f}s "
                    f"accept_len={metrics.spec_accept_length:.3f} "
                    f"spec_verify_ct_sum={metrics.spec_verify_ct_sum}"
                )
        finally:
            kill_process_tree(dflash_proc.pid)
            try:
                dflash_proc.wait(timeout=30)
            except Exception:
                pass

    # Render markdown.
    md_lines: list[str] = []
    md_lines.append("# DFLASH Bench Report")
    md_lines.append("")
    md_lines.append("## Settings")
    md_lines.append(f"- dataset: `{args.dataset_name}`")
    md_lines.append(f"- target_model: `{args.target_model}`")
    md_lines.append(f"- draft_model: `{args.draft_model}`")
    md_lines.append(f"- max_new_tokens: `{args.max_new_tokens}`")
    md_lines.append(f"- attention_backends: `{', '.join(attention_backends)}`")
    md_lines.append(f"- tp_size: `{tp}`")
    md_lines.append(f"- concurrencies: `{', '.join(str(x) for x in concurrencies)}`")
    md_lines.append(f"- questions_per_concurrency: `base={args.questions_per_concurrency_base}`")
    md_lines.append(f"- device_sm: `{device_sm}`")
    md_lines.append(f"- is_blackwell: `{is_blackwell}`")
    md_lines.append(f"- skip_baseline: `{bool(args.skip_baseline)}`")
    md_lines.append(f"- disable_cuda_graph: `{bool(args.disable_cuda_graph)}`")
    md_lines.append(f"- cuda_graph_max_bs: `{args.cuda_graph_max_bs}`")
    md_lines.append(f"- max_running_requests: `{args.max_running_requests}`")
    md_lines.append(f"- page_size: `{args.page_size}`")
    md_lines.append(f"- warmup_requests: `{int(args.warmup_requests)}`")
    md_lines.append(f"- warmup_max_new_tokens: `{int(args.warmup_max_new_tokens)}`")
    md_lines.append(f"- skip_first_requests: `{int(args.skip_first_requests)}`")
    md_lines.append("")

    for backend in attention_backends:
        md_lines.append(f"## Backend: `{backend}`")
        md_lines.append("")

        baseline_values = {
            c: baseline_toks.get((backend, c), None) for c in concurrencies
        }
        dflash_values = {
            c: dflash_toks.get((backend, c), None) for c in concurrencies
        }
        speedup_values: dict[int, Optional[float]] = {}
        for c in concurrencies:
            b = baseline_values.get(c, None)
            d = dflash_values.get(c, None)
            speedup_values[c] = None if (b is None or d is None or b <= 0) else (d / b)

        md_lines.append("### Baseline output tok/s")
        md_lines.append(
            _format_table(
                concurrencies=concurrencies,
                values=baseline_values,
                float_fmt=",.2f",
            )
        )
        md_lines.append("")
        
        md_lines.append("### DFLASH output tok/s")
        md_lines.append(
            _format_table(
                concurrencies=concurrencies,
                values=dflash_values,
                float_fmt=",.2f",
            )
        )
        md_lines.append("")

        md_lines.append("### Speedup (DFLASH / baseline)")
        md_lines.append(
            _format_table(
                concurrencies=concurrencies,
                values=speedup_values,
                float_fmt=".3f",
            )
        )
        md_lines.append("")

        md_lines.append("### DFLASH acceptance length")
        md_lines.append(
            _format_table(
                concurrencies=concurrencies,
                values={
                    c: dflash_accept_len.get((backend, c), None)
                    for c in concurrencies
                },
                float_fmt=".3f",
            )
        )
        md_lines.append("")

    if args.output_md:
        with open(args.output_md, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
            f.write("\n")
        print(f"\nWrote markdown report to: {args.output_md}")
    else:
        print("\nMarkdown report disabled (pass --output-md to write one).")


if __name__ == "__main__":
    main()