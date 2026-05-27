# Domino: Decoupling Causal Modeling from Autoregressive Drafting in Speculative Decoding

Domino accelerates large language model inference with speculative decoding. Standard autoregressive decoding is sequential and often memory-bound, leaving GPU parallelism underused. Existing autoregressive draft models improve draft quality by modeling dependencies between draft tokens, but they also introduce sequential drafting overhead.

Domino keeps drafting block-parallel while adding a lightweight causal correction head. The parallel draft backbone proposes a full block at once, and the Domino head injects causal information from previously drafted tokens to refine the draft distributions. This preserves the low cost of parallel drafting while improving acceptance length and end-to-end speedup.

![Domino pipeline](asset/pipeline.png)

## Supported Models

| Target model | Draft model |
| --- | --- |
| `Qwen/Qwen3-4B` | [`Huang2020/Qwen3-4B-Domino-b16`](https://huggingface.co/Huang2020/Qwen3-4B-Domino-b16) |
| `Qwen/Qwen3-8B` | [`Huang2020/Qwen3-8B-Domino-b16`](https://huggingface.co/Huang2020/Qwen3-8B-Domino-b16) |

## Installation

Use Python 3.10 or newer on a CUDA GPU machine. Install a PyTorch build that matches your CUDA driver, then install the remaining dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-hf.txt
```

For the SGLang benchmark, install the Domino-compatible SGLang branch in the same environment:

```bash
git clone --branch sglang-feat/dflash-domino https://github.com/jianuo-huang/Domino.git sglang-domino
cd sglang-domino
python -m pip install -e ./python
cd -
```

## Hugging Face Benchmark

```bash
DRAFT_MODEL=Huang2020/Qwen3-8B-Domino-b16 \
TARGET_MODEL=Qwen/Qwen3-8B \
PYTHON=python \
./run_hf_benchmark.sh
```

Defaults:

- `TASKS=gsm8k:128`
- `MAX_NEW_TOKENS=2048`
- `TEMPERATURE=0.0`
- `BLOCK_SIZE=16`
- `NUM_GPUS=8`

Override tasks or runtime settings with environment variables:

```bash
TASKS="gsm8k:128,math500:128" NUM_GPUS=4 ./run_hf_benchmark.sh
```

## SGLang Benchmark

```bash
DRAFT_MODEL=Huang2020/Qwen3-8B-Domino-b16 \
TARGET_MODEL=Qwen/Qwen3-8B \
PYTHON=python \
./run_sglang_benchmark.sh
```

Defaults:

- `TASKS=gsm8k:128`
- `MAX_NEW_TOKENS=2048`
- `TEMPERATURE=0.0`
- `CONCURRENCIES=1`

Use these sample counts to reproduce the paper settings:

```bash
TASKS="gsm8k:128,math500:128,aime24:30,aime25:30,humaneval:164,mbpp:128,livecodebench:128,swe-bench:128,mt-bench:80,alpaca:128"
```

Override tasks or runtime settings with environment variables:

```bash
TASKS="mt-bench:80,alpaca:128" CONCURRENCIES=1 ./run_sglang_benchmark.sh
```
