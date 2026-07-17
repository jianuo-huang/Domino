import argparse
import json
import math
import statistics
from pathlib import Path


def parse_summary_spec(value: str) -> tuple[int, Path]:
    try:
        raw_block_size, raw_path = value.split("=", 1)
        block_size = int(raw_block_size)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(
            f"expected BLOCK_SIZE=PATH, got {value!r}"
        ) from error
    if block_size < 1 or not raw_path:
        raise argparse.ArgumentTypeError(
            f"expected a positive BLOCK_SIZE and non-empty PATH, got {value!r}"
        )
    return block_size, Path(raw_path)


def positive_finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select the fastest block size whose benchmark token IDs match the "
            "baseline on every expected task."
        )
    )
    parser.add_argument(
        "--summary",
        action="append",
        default=[],
        type=parse_summary_spec,
        metavar="BLOCK_SIZE=PATH",
        help="Comparison summary; repeat once per task and candidate.",
    )
    parser.add_argument("--block-sizes", default="4,8,12,16")
    parser.add_argument("--expected-summaries-per-block", type=int, default=1)
    parser.add_argument("--min-speedup", type=float, default=1.05)
    parser.add_argument("--require-min-speedup", action="store_true")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    try:
        block_sizes = [int(value) for value in args.block_sizes.split(",")]
    except ValueError as error:
        parser.error(f"invalid --block-sizes: {error}")
    if not block_sizes or any(value < 1 for value in block_sizes):
        parser.error("--block-sizes must contain positive integers.")
    if len(set(block_sizes)) != len(block_sizes):
        parser.error("--block-sizes must not contain duplicates.")
    if args.expected_summaries_per_block < 1:
        parser.error("--expected-summaries-per-block must be at least 1.")
    if not math.isfinite(args.min_speedup) or args.min_speedup < 0:
        parser.error("--min-speedup must be a finite, non-negative number.")

    grouped: dict[int, list[tuple[Path, dict]]] = {
        block_size: [] for block_size in block_sizes
    }
    for block_size, path in args.summary:
        if block_size not in grouped:
            parser.error(f"summary supplied for unlisted block size {block_size}.")
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f"cannot read summary {path}: {error}")
        grouped[block_size].append((path, summary))

    candidates = []
    for block_size in block_sizes:
        entries = grouped[block_size]
        reasons = []
        if len(entries) != args.expected_summaries_per_block:
            reasons.append(
                "expected "
                f"{args.expected_summaries_per_block} summaries, found {len(entries)}"
            )

        baseline_times = []
        domino_times = []
        compared_turns = 0
        summary_paths = []
        for path, summary in entries:
            summary_paths.append(str(path))
            mismatches = summary.get("token_mismatches")
            if mismatches is None:
                reasons.append(f"{path}: missing token_mismatches")
            elif mismatches:
                reasons.append(f"{path}: {len(mismatches)} token mismatch(es)")

            turns = summary.get("compared_turns")
            if not isinstance(turns, int) or turns < 1:
                reasons.append(f"{path}: compared_turns must be positive")
            else:
                compared_turns += turns

            baseline_time = summary.get("median_baseline_decode_ms_per_token")
            domino_time = summary.get("median_domino_decode_ms_per_token")
            if not positive_finite(baseline_time):
                reasons.append(f"{path}: invalid baseline decode time")
            else:
                baseline_times.append(float(baseline_time))
            if not positive_finite(domino_time):
                reasons.append(f"{path}: invalid Domino decode time")
            else:
                domino_times.append(float(domino_time))

        legal = not reasons
        baseline_median = statistics.median(baseline_times) if legal else None
        domino_median = statistics.median(domino_times) if legal else None
        speedup = baseline_median / domino_median if legal else None
        candidates.append(
            {
                "block_size": block_size,
                "legal": legal,
                "reasons": reasons,
                "summary_count": len(entries),
                "summary_paths": summary_paths,
                "compared_turns": compared_turns,
                "median_baseline_decode_ms_per_token": baseline_median,
                "median_domino_decode_ms_per_token": domino_median,
                "decode_speedup": speedup,
                "meets_min_speedup": legal and speedup >= args.min_speedup,
            }
        )

    legal_candidates = [candidate for candidate in candidates if candidate["legal"]]
    selected = min(
        legal_candidates,
        key=lambda candidate: (
            candidate["median_domino_decode_ms_per_token"],
            candidate["block_size"],
        ),
        default=None,
    )
    result = {
        "selected_block_size": selected["block_size"] if selected else None,
        "selection_rule": (
            "lowest median Domino decode time among candidates with exact token "
            "matches on every expected task"
        ),
        "min_speedup": args.min_speedup,
        "selected_meets_min_speedup": (
            selected["meets_min_speedup"] if selected else False
        ),
        "candidates": candidates,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output_json:
        Path(args.output_json).write_text(rendered + "\n", encoding="utf-8")

    if selected is None:
        raise SystemExit("No legal block-size candidate completed all tasks.")
    if args.require_min_speedup and not selected["meets_min_speedup"]:
        raise SystemExit(
            f"Selected block size {selected['block_size']} achieved "
            f"{selected['decode_speedup']:.3f}x, below {args.min_speedup:.3f}x."
        )


if __name__ == "__main__":
    main()
