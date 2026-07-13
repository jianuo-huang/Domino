import argparse
import json
import statistics
from pathlib import Path


def load_answers(path: str) -> dict[int, dict]:
    with open(path, encoding="utf-8") as handle:
        return {
            int(item["question_id"]): item
            for item in (json.loads(line) for line in handle if line.strip())
        }


def only_choice(answer: dict, expected_mode: str) -> dict:
    choices = [choice for choice in answer["choices"] if choice["mode"] == expected_mode]
    if len(choices) != 1:
        raise ValueError(
            f"Expected one {expected_mode!r} choice for question "
            f"{answer['question_id']}, found {len(choices)}."
        )
    return choices[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--domino", required=True)
    parser.add_argument("--min-speedup", type=float, default=1.05)
    parser.add_argument("--min-wall-tokens", type=int, default=128)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()

    baseline = load_answers(args.baseline)
    domino = load_answers(args.domino)
    if baseline.keys() != domino.keys():
        raise RuntimeError("Baseline and Domino answer files contain different questions.")

    baseline_token_times = []
    domino_token_times = []
    baseline_wall_times = []
    domino_wall_times = []
    mismatches = []
    compared_turns = 0
    for question_id in sorted(baseline):
        base_choice = only_choice(baseline[question_id], "baseline")
        domino_choice = only_choice(domino[question_id], "domino")
        if len(base_choice["output_ids"]) != len(domino_choice["output_ids"]):
            raise RuntimeError(f"Turn count mismatch for question {question_id}.")
        for turn, (base_ids, domino_ids) in enumerate(
            zip(base_choice["output_ids"], domino_choice["output_ids"])
        ):
            compared_turns += 1
            if base_ids != domino_ids:
                mismatches.append({"question_id": question_id, "turn": turn})
            baseline_token_times.append(base_choice["time_per_output_token"][turn])
            domino_token_times.append(domino_choice["time_per_output_token"][turn])
            if base_choice["new_tokens"][turn] >= args.min_wall_tokens:
                baseline_wall_times.append(base_choice["wall_time"][turn])
                domino_wall_times.append(domino_choice["wall_time"][turn])

    baseline_decode = statistics.median(baseline_token_times)
    domino_decode = statistics.median(domino_token_times)
    decode_speedup = baseline_decode / domino_decode
    wall_speedup = None
    if baseline_wall_times:
        wall_speedup = statistics.median(baseline_wall_times) / statistics.median(
            domino_wall_times
        )

    passed = not mismatches and decode_speedup >= args.min_speedup
    if wall_speedup is not None:
        passed = passed and wall_speedup >= 1.0
    summary = {
        "passed": passed,
        "compared_turns": compared_turns,
        "token_mismatches": mismatches,
        "median_baseline_decode_ms_per_token": baseline_decode * 1000,
        "median_domino_decode_ms_per_token": domino_decode * 1000,
        "decode_speedup": decode_speedup,
        "wall_speedup_for_long_outputs": wall_speedup,
        "min_speedup": args.min_speedup,
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output_json:
        Path(args.output_json).write_text(rendered + "\n", encoding="utf-8")
    if args.enforce and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
