#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_CODE_ROOT="${BENCHMARK_CODE_ROOT:-${SCRIPT_DIR}/code}"
PYTHON="${PYTHON:-python}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-}"
TASKS="${TASKS:-gsm8k:4,math500:4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.0}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29601}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/outputs/hf_$(date +%Y%m%d_%H%M%S)}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

if [ -z "${DRAFT_MODEL}" ]; then
  fail "DRAFT_MODEL is required. Example: DRAFT_MODEL=/path/to/qwen8B-Domino ./run_hf_benchmark.sh"
fi
if [ ! -d "${DRAFT_MODEL}" ]; then
  fail "DRAFT_MODEL does not exist or is not a directory: ${DRAFT_MODEL}"
fi
if [ ! -f "${BENCHMARK_CODE_ROOT}/benchmark.py" ]; then
  fail "BENCHMARK_CODE_ROOT must point to bundled benchmark code containing benchmark.py. Current: ${BENCHMARK_CODE_ROOT}"
fi
if [[ "${PYTHON}" == */* ]]; then
  [ -x "${PYTHON}" ] || fail "PYTHON is not executable: ${PYTHON}"
else
  command -v "${PYTHON}" >/dev/null 2>&1 || fail "PYTHON not found on PATH: ${PYTHON}"
fi

mkdir -p "${OUT_DIR}"
cd "${BENCHMARK_CODE_ROOT}"

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
