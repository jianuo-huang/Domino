# Domino: Decoupling Causal Modeling from Autoregressive Drafting in Speculative Decoding
<p align="center">
  <a href="https://arxiv.org/abs/2605.29707"><img src="https://img.shields.io/badge/Paper-arXiv%3A2605.29707-blue" alt="Paper"></a>
  <a href="https://huggingface.co/collections/Huang2020/domino"><img src="https://img.shields.io/badge/Hugging%20Face-Models-yellow" alt="Hugging Face Models"></a>
</p>
Domino is a speculative decoding method that keeps draft generation block-parallel while adding a lightweight causal correction head to improve draft-token acceptance.

![Domino pipeline](asset/pipeline.png)

## News

- [2026-05-30] 🔥🔥 Domino training code is now available in [SpecForge](https://github.com/sgl-project/SpecForge).
- [2026-05-29] 🔥 Domino paper released! Read the paper on [arXiv](https://arxiv.org/abs/2605.29707).

## Demo

![Domino throughput demo](asset/DFlash_demo.gif)

## Supported Models

| Target model | Draft model |
| --- | --- |
| `Qwen3-4B` | [`Qwen3-4B-Domino-b16`](https://huggingface.co/Huang2020/Qwen3-4B-Domino-b16) |
| `Qwen3-8B` | [`Qwen3-8B-Domino-b16`](https://huggingface.co/Huang2020/Qwen3-8B-Domino-b16) |
| `Qwen3.6-35B-A3B` | Coming soon |
| `Qwen3.6-27B` | [`Qwen3.6-27B-Domino`](https://huggingface.co/Huang2020/Qwen3.6-27B-Domino) |

## Installation

```bash
uv pip install "git+https://github.com/jianuo-huang/sglang.git@feat/domino-tensor-parallel#subdirectory=python"
```

## Quick Usage

Domino draft checkpoints provide `spec_generate` for direct speculative decoding with a target model. We currently recommend running this path on one GPU.

```python
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

draft_model = AutoModel.from_pretrained(
    "Huang2020/Qwen3-8B-Domino-b16",
    trust_remote_code=True,
    dtype="auto",
    device_map="cuda:0",
).eval()

target_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    dtype="auto",
    device_map="cuda:0",
).eval()

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
prompt = "How many positive whole-number divisors does 196 have?"
messages = [{"role": "user", "content": prompt}]

# The Domino draft model is trained for Qwen3 with thinking mode disabled.
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
model_inputs = tokenizer([text], return_tensors="pt").to(draft_model.device)

output_ids = draft_model.spec_generate(
    input_ids=model_inputs["input_ids"],
    target=target_model,
    max_new_tokens=2048,
    temperature=0.0,
    stop_token_ids=[tokenizer.eos_token_id],
)

generated_ids = output_ids[:, model_inputs["input_ids"].shape[1]:]
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
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
- `CONCURRENCIES=1,2,4,8,16,32`

Use these sample counts to reproduce the paper settings:

```bash
TASKS="gsm8k:128,math500:128,aime24:30,aime25:30,humaneval:164,mbpp:128,livecodebench:128,swe-bench:128,mt-bench:80,alpaca:128"
```

Override tasks or runtime settings with environment variables:

```bash
TASKS="mt-bench:80,alpaca:128" CONCURRENCIES=1 ./run_sglang_benchmark.sh
```

## Acknowledgements

We thank the authors and maintainers of [DFlash](https://github.com/z-lab/dflash), [SpecForge](https://github.com/sgl-project/SpecForge), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), and [SGLang](https://github.com/sgl-project/sglang). Their open-source work on block-parallel speculative decoding, speculative-decoding training infrastructure, high-performance attention kernels, and LLM serving helped shape this project and its benchmarking setup.

## Citation

If you use Domino in your research, please cite:

```bibtex
@article{huang2026domino,
  title={Domino: Decoupling Causal Modeling from Autoregressive Drafting in Speculative Decoding},
  author={Huang, Jianuo and Zhang, Yaojie and Zhang, Qituan and Lin, Hao and Xu, Hanlin and Zhang, Linfeng},
  journal={arXiv preprint arXiv:2605.29707},
  year={2026},
  eprint={2605.29707},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  doi={10.48550/arXiv.2605.29707},
  url={https://arxiv.org/abs/2605.29707}
}
```
