# Code Manifest

This release contains only the files needed for the Domino `qwen8B-Domino` HF benchmark and the SGLang benchmark client. The SGLang server implementation is not vendored; install it from the public Domino branch.

| Path | Role |
| --- | --- |
| `run_hf_benchmark.sh` | Reviewer-facing HF launch script. |
| `run_sglang_benchmark.sh` | Reviewer-facing SGLang launch script. |
| `requirements-hf.txt` | Python dependency list for the HF benchmark, excluding CUDA-specific PyTorch wheel selection. |
| `code/benchmark.py` | Block-size-1 baseline vs Domino/DFlash speculative decoding benchmark. |
| `code/benchmark_sglang.py` | Shared SGLang request, timing, and report helpers. |
| `code/benchmark_sglang_tasks.py` | Multi-task SGLang DFlash benchmark client. |
| `code/model/utils.py` | Dataset loading, sampling, and target hidden-state extraction helpers. |
| `code/specforge/modeling/draft/dflash.py` | Domino `qwen8B-Domino` draft model definition. |
| `code/kernel/domino.py` | Domino CUDA Graph/Triton rollout runner. |
| `code/distributed.py` | Minimal `torch.distributed` helper used by `benchmark.py`. |
