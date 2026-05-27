#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_CODE_ROOT="${BENCHMARK_CODE_ROOT:-${SCRIPT_DIR}/code}"
PYTHON="${PYTHON:-python}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-Huang2020/Qwen3-8B-Domino-b16}"
TASKS="${TASKS:-gsm8k:128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-1}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
CONCURRENCIES="${CONCURRENCIES:-1}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-4}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.60}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-1}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/outputs/sglang_$(date +%Y%m%d_%H%M%S)}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${SCRIPT_DIR}/.cache/huggingface}"

mkdir -p "${OUT_DIR}"

export HF_HOME="${HF_CACHE_DIR}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export HF_HUB_DISABLE_SYMLINKS_WARNING="${HF_HUB_DISABLE_SYMLINKS_WARNING:-1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${OUT_DIR}/cache/xdg}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${OUT_DIR}/cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${OUT_DIR}/cache/triton}"
export SGLANG_CACHE_DIR="${SGLANG_CACHE_DIR:-${OUT_DIR}/cache/sglang}"
export TVM_FFI_CACHE_DIR="${TVM_FFI_CACHE_DIR:-${OUT_DIR}/cache/tvm-ffi}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${OUT_DIR}/cache/flashinfer}"
export TMPDIR="${TMPDIR:-/tmp/sglang_domino_${USER:-user}_$$}"
mkdir -p   "${HF_HOME}"   "${HF_HUB_CACHE}"   "${TRANSFORMERS_CACHE}"   "${XDG_CACHE_HOME}"   "${TORCHINDUCTOR_CACHE_DIR}"   "${TRITON_CACHE_DIR}"   "${SGLANG_CACHE_DIR}"   "${TVM_FFI_CACHE_DIR}"   "${FLASHINFER_WORKSPACE_BASE}"   "${TMPDIR}"

CUDA_GRAPH_ARGS=()
if [ "${DISABLE_CUDA_GRAPH}" != "0" ]; then
  CUDA_GRAPH_ARGS+=(--disable-cuda-graph)
fi

"${PYTHON}" "${BENCHMARK_CODE_ROOT}/benchmark_sglang_tasks.py" \
  --mode dflash \
  --target-model "${TARGET_MODEL}" \
  --draft-model "${DRAFT_MODEL}" \
  --tasks "${TASKS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --top-k "${TOP_K}" \
  --attention-backend "${ATTENTION_BACKEND}" \
  --concurrencies "${CONCURRENCIES}" \
  --warmup-requests 0 \
  --warmup-max-new-tokens 8 \
  --skip-first-requests 0 \
  --timeout-s 3600 \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  "${CUDA_GRAPH_ARGS[@]}" \
  --output-md "${OUT_DIR}/sglang_domino_tasks.md" \
  --output-jsonl "${OUT_DIR}/sglang_domino_tasks.jsonl"

echo "Wrote outputs to ${OUT_DIR}"
