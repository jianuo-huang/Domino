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
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
CONCURRENCIES="${CONCURRENCIES:-1,2,4,8,16,32}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-64}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.75}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/outputs/sglang_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${OUT_DIR}"

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
  --timeout-s 3600 \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --output-md "${OUT_DIR}/sglang_domino_tasks.md" \
  --output-jsonl "${OUT_DIR}/sglang_domino_tasks.jsonl"

echo "Wrote outputs to ${OUT_DIR}"
