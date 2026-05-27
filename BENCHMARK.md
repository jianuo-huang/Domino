# Benchmark Instructions

This file describes how to run the bundled benchmarks for the `qwen8B-Domino` checkpoint. It does not contain precomputed results.

## HF Default Smoke Run

```bash
cd /path/to/qwen8B-Domino-release
DRAFT_MODEL=/path/to/qwen8B-Domino \
TARGET_MODEL=Qwen/Qwen3-8B \
PYTHON=python \
./run_hf_benchmark.sh
```

Defaults:

- Datasets: `gsm8k:4,math500:4`
- Max new tokens: `2048`
- Temperature: `0.0`
- Draft block size: `16`
- GPUs: `1`
- Output directory: `outputs/hf_<timestamp>/`

## HF Faster Reviewer Smoke Run

For a quick load-and-decode check, reduce samples:

```bash
TASKS="gsm8k:2,math500:2" \
DRAFT_MODEL=/path/to/qwen8B-Domino \
TARGET_MODEL=Qwen/Qwen3-8B \
PYTHON=python \
./run_hf_benchmark.sh
```

## HF Larger Runs

Increase samples by editing `TASKS`:

```bash
TASKS="gsm8k:128,math500:128" \
DRAFT_MODEL=/path/to/qwen8B-Domino \
TARGET_MODEL=Qwen/Qwen3-8B \
./run_hf_benchmark.sh
```

Use `NUM_GPUS` only when the target environment has a working multi-GPU PyTorch distributed setup:

```bash
NUM_GPUS=4 MASTER_PORT=29601 ./run_hf_benchmark.sh
```

## HF Outputs

For each dataset, the script writes:

- `<dataset>_t<temperature>.log`: console log with timing, speedup, and acceptance length.
- `<dataset>_t<temperature>_answers.jsonl`: generated outputs from the block-size-1 baseline and Domino/DFlash path.

The benchmark itself prints the measured values; reviewers should report the numbers generated in their own environment.

## SGLang Branch Setup

The SGLang benchmark requires the Domino-compatible SGLang branch:

```bash
export SGLANG_REPO_URL="https://github.com/jianuo-huang/Domino.git"
export SGLANG_BRANCH="sglang-feat/dflash-domino"

git clone --branch "${SGLANG_BRANCH}" "${SGLANG_REPO_URL}" sglang-domino
cd sglang-domino
python -m pip install -e ./python
```

Then install the release-side dependencies in the same environment:

```bash
cd /path/to/qwen8B-Domino-release
python -m pip install -r requirements-hf.txt
```

## SGLang Smoke Run

```bash
cd /path/to/qwen8B-Domino-release
DRAFT_MODEL=/path/to/qwen8B-Domino \
TARGET_MODEL=Qwen/Qwen3-8B \
PYTHON=python \
./run_sglang_benchmark.sh
```

Defaults:

- Datasets: `gsm8k:2,math500:2,aime25:2,humaneval:2,mbpp:2,livecodebench:2,mt-bench:2,alpaca:2`
- Max new tokens: `2048`
- Temperature: `0.0`
- Attention backend: `triton`
- Concurrency: `1`
- Output directory: `outputs/sglang_<timestamp>/`

For a shorter check, reduce `TASKS`:

```bash
TASKS="gsm8k:2,math500:2" \
DRAFT_MODEL=/path/to/qwen8B-Domino \
TARGET_MODEL=Qwen/Qwen3-8B \
./run_sglang_benchmark.sh
```

The SGLang script writes `sglang_domino_tasks.md` and `sglang_domino_tasks.jsonl`. The Markdown report contains measured output tokens per second and DFlash acceptance length for the tasks run locally.
