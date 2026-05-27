# qwen8B-Domino Benchmark Release

`qwen8B-Domino` is a Domino/DFlash speculative draft checkpoint for `Qwen/Qwen3-8B`. It is not a standalone chat model: run it together with the Qwen3-8B target model and the bundled benchmark code in this directory.

Checkpoint download:

<https://drive.google.com/drive/folders/1C3eOPxnXnyAnBWmJytp6f2-QWARbztOL?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto>

## Package Contents

- `code/`: bundled Domino benchmark client code.
- `run_hf_benchmark.sh`: reviewer-facing HF launch script.
- `run_sglang_benchmark.sh`: reviewer-facing SGLang launch script.
- `requirements-hf.txt`: Python package requirements for the HF benchmark, excluding the CUDA-specific PyTorch wheel choice.
- `BENCHMARK.md`: exact run commands and output descriptions.
- `CODE_MANIFEST.md`: source file manifest.

No result logs or precomputed benchmark numbers are included. Reviewers should run the scripts and inspect the generated outputs.

## HF Setup

Use Python 3.10 or newer on a CUDA GPU machine. Install a PyTorch build that matches the local CUDA driver first, then install the remaining dependencies from this release directory:

```bash
cd /path/to/qwen8B-Domino-release
python -m pip install --upgrade pip
python -m pip install -r requirements-hf.txt
```

If PyTorch is not installed yet, install the CUDA wheel recommended for your system from the PyTorch installation guide, then rerun the requirements command above.

Download the checkpoint with the Google Drive UI or `gdown`:

```bash
python -m pip install gdown
gdown --folder "https://drive.google.com/drive/folders/1C3eOPxnXnyAnBWmJytp6f2-QWARbztOL" -O ./qwen8B-Domino
```

Set the target and draft model paths:

```bash
export TARGET_MODEL="Qwen/Qwen3-8B"
export DRAFT_MODEL="/path/to/qwen8B-Domino"
```

`DRAFT_MODEL` must point to the downloaded checkpoint directory. `TARGET_MODEL` may be the Hugging Face model id above or a local Qwen3-8B snapshot directory.

## Run HF Benchmark

A small smoke benchmark runs GSM8K and MATH-500 with four samples each by default:

```bash
DRAFT_MODEL=/path/to/qwen8B-Domino \
TARGET_MODEL=Qwen/Qwen3-8B \
PYTHON=python \
./run_hf_benchmark.sh
```

The default settings are `max_new_tokens=2048`, `temperature=0.0`, `block_size=16`, `--use-bias`, and `--use-graph`. Override sample counts through `TASKS`, for example:

```bash
TASKS="gsm8k:2,math500:2" ./run_hf_benchmark.sh
```

Outputs are written under `outputs/hf_<timestamp>/`. Each task gets a log file and an answer JSONL file containing both the block-size-1 baseline and Domino/DFlash generations.

## SGLang Setup

The SGLang server path is kept outside this release package. Install the Domino-compatible SGLang branch before running the SGLang benchmark:

```bash
export SGLANG_REPO_URL="https://github.com/jianuo-huang/Domino.git"
export SGLANG_BRANCH="sglang-feat/dflash-domino"

git clone --branch "${SGLANG_BRANCH}" "${SGLANG_REPO_URL}" sglang-domino
cd sglang-domino
python -m pip install -e ./python
```

Use the same Python environment for the release dependencies:

```bash
cd /path/to/qwen8B-Domino-release
python -m pip install -r requirements-hf.txt
```

Do not use upstream SGLang `main` for this benchmark unless the DFlash speculative backend has been merged there; the stock server does not include the server-side DFlash implementation used by `qwen8B-Domino`. The small Domino alias patch is also included under `sglang_patch/` for reference.

## Run SGLang Benchmark

The default SGLang run uses two samples from each listed task and one client concurrency:

```bash
DRAFT_MODEL=/path/to/qwen8B-Domino \
TARGET_MODEL=Qwen/Qwen3-8B \
PYTHON=python \
./run_sglang_benchmark.sh
```

Override task counts or concurrency through environment variables:

```bash
TASKS="gsm8k:2,math500:2" CONCURRENCIES="1" ./run_sglang_benchmark.sh
```

Outputs are written under `outputs/sglang_<timestamp>/` as a Markdown report and JSONL records.

## Expected Run Signal

A successful HF run should print the baseline time, Domino/DFlash time, speedup, and average acceptance length. A successful SGLang run should print output tokens per second and DFlash acceptance length. The exact numbers depend on GPU type, CUDA/PyTorch/SGLang versions, local model cache, and sample count, so this package intentionally does not include fixed benchmark results.

## Notes

- Pairing matters: this checkpoint was trained for Qwen3-8B hidden states.
- The HF benchmark code is bundled under `code/` and is trimmed to the Domino path used by qwen8B-Domino; no source code from the original internal repositories is required.
- The SGLang benchmark client is bundled, while the SGLang server implementation comes from the public Domino branch described above.
- The benchmark downloads public datasets through the Hugging Face `datasets` package unless they are already cached locally.
