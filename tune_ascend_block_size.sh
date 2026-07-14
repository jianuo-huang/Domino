#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Run a short Ascend Domino block-size sweep and select the fastest correct result.

Usage:
  ./tune_ascend_block_size.sh

Configuration is supplied through environment variables:
  BLOCK_SIZES       Comma-separated candidates (default: 4,8,12,16)
  TASKS             Comma-separated DATASET:SAMPLES pairs (default: alpaca:4)
  MAX_NEW_TOKENS    Generation limit per turn (default: 128)
  WARMUP_SAMPLES    Warmup generations (default: 1)
  REPETITIONS       Timed repetitions per sample (default: 2)
  NUM_NPUS          Number of NPU worker processes (default: 1)
  TARGET_MODEL      Target model (default: Qwen/Qwen3-4B)
  TARGET_REVISION   Pinned target revision
  TARGET_DTYPE      Target weight dtype (default: float16)
  DRAFT_MODEL       Domino draft model (default: Huang2020/Qwen3-4B-Domino-b16)
  DRAFT_REVISION    Pinned draft revision
  MIN_SPEEDUP       Speedup reported as the target gate (default: 1.05)
  ENFORCE            Exit nonzero unless the selected result meets MIN_SPEEDUP
                     (default: 0)
  MASTER_PORT       First distributed launcher port (default: 29661)
  OUT_DIR           Output directory (default: timestamped outputs directory)
  PYTHON             Python executable (default: python)

Each candidate is compared with one shared baseline per task. A candidate is
eligible only when all generated token IDs match the baseline on every task.
The eligible candidate with the lowest median decode time is selected.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -ne 0 ]]; then
  echo "Unknown argument: $1" >&2
  usage >&2
  exit 2
fi

source "${SCRIPT_DIR}/activate_ascend.sh"

PYTHON="${PYTHON:-python}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-4B}"
if [[ ! -v TARGET_REVISION ]]; then
  TARGET_REVISION="1cfa9a7208912126459214e8b04321603b3df60c"
fi
TARGET_DTYPE="${TARGET_DTYPE:-float16}"
DRAFT_MODEL="${DRAFT_MODEL:-Huang2020/Qwen3-4B-Domino-b16}"
if [[ ! -v DRAFT_REVISION ]]; then
  DRAFT_REVISION="12f52f165aea6e57e56373b2cb0d7f93bf41d4c1"
fi
TASKS="${TASKS:-alpaca:4}"
BLOCK_SIZES="${BLOCK_SIZES:-4,8,12,16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
NUM_NPUS="${NUM_NPUS:-1}"
WARMUP_SAMPLES="${WARMUP_SAMPLES:-1}"
REPETITIONS="${REPETITIONS:-2}"
MASTER_PORT="${MASTER_PORT:-29661}"
MIN_SPEEDUP="${MIN_SPEEDUP:-1.05}"
ENFORCE="${ENFORCE:-0}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/outputs/ascend_tuning_$(date +%Y%m%d_%H%M%S)}"

TARGET_REVISION_ARGS=()
if [[ -n "${TARGET_REVISION}" ]]; then
  TARGET_REVISION_ARGS=(--model-revision "${TARGET_REVISION}")
fi
DRAFT_REVISION_ARGS=()
if [[ -n "${DRAFT_REVISION}" ]]; then
  DRAFT_REVISION_ARGS=(--draft-revision "${DRAFT_REVISION}")
fi

if [[ "${ENFORCE}" != "0" && "${ENFORCE}" != "1" ]]; then
  echo "ENFORCE must be 0 or 1, got: ${ENFORCE}" >&2
  exit 2
fi
for integer_value in "${MAX_NEW_TOKENS}" "${NUM_NPUS}" "${REPETITIONS}"; do
  if [[ ! "${integer_value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Expected a positive integer, got: ${integer_value}" >&2
    exit 2
  fi
done
if [[ ! "${WARMUP_SAMPLES}" =~ ^[0-9]+$ ]]; then
  echo "WARMUP_SAMPLES must be a non-negative integer, got: ${WARMUP_SAMPLES}" >&2
  exit 2
fi
if [[ ! "${MASTER_PORT}" =~ ^[0-9]+$ ]] || (( MASTER_PORT < 1 || MASTER_PORT > 65535 )); then
  echo "MASTER_PORT must be between 1 and 65535, got: ${MASTER_PORT}" >&2
  exit 2
fi

IFS=',' read -r -a BLOCK_ARRAY <<< "${BLOCK_SIZES}"
if (( ${#BLOCK_ARRAY[@]} == 0 )); then
  echo "BLOCK_SIZES must contain at least one size." >&2
  exit 2
fi
for block_size in "${BLOCK_ARRAY[@]}"; do
  if [[ ! "${block_size}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid block size: ${block_size}" >&2
    exit 2
  fi
done

IFS=',' read -r -a TASK_ARRAY <<< "${TASKS}"
if (( ${#TASK_ARRAY[@]} == 0 )); then
  echo "TASKS must contain at least one DATASET:SAMPLES pair." >&2
  exit 2
fi
for task in "${TASK_ARRAY[@]}"; do
  IFS=':' read -r dataset max_samples extra <<< "${task}"
  if [[ -z "${dataset}" || -n "${extra:-}" || ! "${max_samples:-}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid task ${task@Q}; expected DATASET:SAMPLES." >&2
    exit 2
  fi
done

mkdir -p "${OUT_DIR}"
STATUS_FILE="${OUT_DIR}/run_status.tsv"
printf 'dataset\tblock_size\tstatus\tanswer_file\tsummary_file\n' > "${STATUS_FILE}"

echo "Target: ${TARGET_MODEL}@${TARGET_REVISION}"
echo "Target dtype: ${TARGET_DTYPE}"
echo "Draft: ${DRAFT_MODEL}@${DRAFT_REVISION}"
echo "Tasks: ${TASKS}; block sizes: ${BLOCK_SIZES}"
echo "Outputs: ${OUT_DIR}"

PORT_CURSOR="${MASTER_PORT}"
SUMMARY_ARGS=()
for task in "${TASK_ARRAY[@]}"; do
  IFS=':' read -r DATASET MAX_SAMPLES <<< "${task}"
  SAFE_DATASET="${DATASET//\//_}"
  MANIFEST="${OUT_DIR}/${SAFE_DATASET}_manifest.jsonl"
  BASELINE="${OUT_DIR}/${SAFE_DATASET}_baseline.jsonl"
  BASELINE_LOG="${OUT_DIR}/${SAFE_DATASET}_baseline.log"

  "${PYTHON}" "${SCRIPT_DIR}/code/benchmark_portable.py" \
    --dataset "${DATASET}" --max-samples "${MAX_SAMPLES}" \
    --dump-benchmark-manifest "${MANIFEST}" --dump-only

  if (( PORT_CURSOR > 65535 )); then
    echo "Distributed launcher port range exceeded 65535." >&2
    exit 2
  fi
  if ! "${PYTHON}" -m torch.distributed.run \
    --nproc_per_node="${NUM_NPUS}" --master_port="${PORT_CURSOR}" \
    "${SCRIPT_DIR}/code/benchmark_portable.py" \
    --mode baseline --device-backend npu --attn-implementation sdpa \
    --benchmark-manifest "${MANIFEST}" \
    --model-name-or-path "${TARGET_MODEL}" "${TARGET_REVISION_ARGS[@]}" \
    --target-dtype "${TARGET_DTYPE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --warmup-samples "${WARMUP_SAMPLES}" --repetitions "${REPETITIONS}" \
    --answer-file "${BASELINE}" 2>&1 | tee "${BASELINE_LOG}"; then
    printf '%s\t-\tbaseline_failed\t%s\t-\n' "${DATASET}" "${BASELINE}" >> "${STATUS_FILE}"
    echo "Baseline failed for ${DATASET}; stopping because no candidate can be compared." >&2
    exit 1
  fi
  printf '%s\t-\tbaseline_ok\t%s\t-\n' "${DATASET}" "${BASELINE}" >> "${STATUS_FILE}"
  PORT_CURSOR=$((PORT_CURSOR + 1))

  for block_size in "${BLOCK_ARRAY[@]}"; do
    DOMINO="${OUT_DIR}/${SAFE_DATASET}_domino_b${block_size}.jsonl"
    DOMINO_LOG="${OUT_DIR}/${SAFE_DATASET}_domino_b${block_size}.log"
    SUMMARY="${OUT_DIR}/${SAFE_DATASET}_summary_b${block_size}.json"

    if (( PORT_CURSOR > 65535 )); then
      echo "Distributed launcher port range exceeded 65535." >&2
      exit 2
    fi
    if ! "${PYTHON}" -m torch.distributed.run \
      --nproc_per_node="${NUM_NPUS}" --master_port="${PORT_CURSOR}" \
      "${SCRIPT_DIR}/code/benchmark_portable.py" \
      --mode domino --device-backend npu --attn-implementation sdpa \
      --benchmark-manifest "${MANIFEST}" \
      --model-name-or-path "${TARGET_MODEL}" "${TARGET_REVISION_ARGS[@]}" \
      --target-dtype "${TARGET_DTYPE}" \
      --draft-name-or-path "${DRAFT_MODEL}" "${DRAFT_REVISION_ARGS[@]}" \
      --block-size "${block_size}" --use-bias \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --warmup-samples "${WARMUP_SAMPLES}" --repetitions "${REPETITIONS}" \
      --answer-file "${DOMINO}" 2>&1 | tee "${DOMINO_LOG}"; then
      printf '%s\t%s\tdomino_failed\t%s\t-\n' \
        "${DATASET}" "${block_size}" "${DOMINO}" >> "${STATUS_FILE}"
      echo "Block size ${block_size} failed on ${DATASET}; continuing the sweep." >&2
      PORT_CURSOR=$((PORT_CURSOR + 1))
      continue
    fi
    PORT_CURSOR=$((PORT_CURSOR + 1))

    if "${PYTHON}" "${SCRIPT_DIR}/code/compare_benchmarks.py" \
      --baseline "${BASELINE}" --domino "${DOMINO}" \
      --min-speedup 0 --min-wall-tokens 1000000000 \
      --output-json "${SUMMARY}"; then
      printf '%s\t%s\tcompared\t%s\t%s\n' \
        "${DATASET}" "${block_size}" "${DOMINO}" "${SUMMARY}" >> "${STATUS_FILE}"
      SUMMARY_ARGS+=(--summary "${block_size}=${SUMMARY}")
    else
      printf '%s\t%s\tcompare_failed\t%s\t%s\n' \
        "${DATASET}" "${block_size}" "${DOMINO}" "${SUMMARY}" >> "${STATUS_FILE}"
      echo "Comparison failed for block size ${block_size} on ${DATASET}; continuing." >&2
    fi
  done
done

SELECT_ARGS=(
  --block-sizes "${BLOCK_SIZES}"
  --expected-summaries-per-block "${#TASK_ARRAY[@]}"
  --min-speedup "${MIN_SPEEDUP}"
  --output-json "${OUT_DIR}/selected_block_size.json"
)
if [[ "${ENFORCE}" == "1" ]]; then
  SELECT_ARGS+=(--require-min-speedup)
fi

"${PYTHON}" "${SCRIPT_DIR}/code/select_block_size.py" \
  "${SELECT_ARGS[@]}" "${SUMMARY_ARGS[@]}"

echo "Block-size tuning outputs written to ${OUT_DIR}"
