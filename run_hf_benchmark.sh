#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_CODE_ROOT="${BENCHMARK_CODE_ROOT:-${SCRIPT_DIR}/code}"
PYTHON="${PYTHON:-python}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-}"
TASKS="${TASKS:-gsm8k:128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.0}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29601}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/outputs/hf_$(date +%Y%m%d_%H%M%S)}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${SCRIPT_DIR}/.cache/huggingface}"

mkdir -p "${OUT_DIR}"

export HF_HOME="${HF_CACHE_DIR}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export HF_HUB_DISABLE_SYMLINKS_WARNING="${HF_HUB_DISABLE_SYMLINKS_WARNING:-1}"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

IFS=',' read -r -a TASK_ARRAY <<< "${TASKS}"
for task in "${TASK_ARRAY[@]}"; do
  IFS=':' read -r DATASET MAX_SAMPLES <<< "${task}"
  LOG_FILE="${OUT_DIR}/${DATASET}_t${TEMPERATURE}.log"
  ANSWER_FILE="${OUT_DIR}/${DATASET}_t${TEMPERATURE}_answers.jsonl"

  echo "dataset=${DATASET} max_samples=${MAX_SAMPLES} temperature=${TEMPERATURE}" | tee "${LOG_FILE}"

  "${PYTHON}" -m torch.distributed.run \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    "${BENCHMARK_CODE_ROOT}/benchmark.py" \
    --dataset "${DATASET}" \
    --max-samples "${MAX_SAMPLES}" \
    --model-name-or-path "${TARGET_MODEL}" \
    --draft-name-or-path "${DRAFT_MODEL}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --block-size "${BLOCK_SIZE}" \
    --use-bias \
    --use-graph \
    --answer-file "${ANSWER_FILE}" 2>&1 | tee -a "${LOG_FILE}"
done

echo "Wrote outputs to ${OUT_DIR}"
