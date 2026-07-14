#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/activate_ascend.sh"

PYTHON="${PYTHON:-python}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-8B}"
if [[ ! -v TARGET_REVISION ]]; then
  TARGET_REVISION="b968826d9c46dd6066d109eabc6255188de91218"
fi
TARGET_DTYPE="${TARGET_DTYPE:-float16}"
DRAFT_MODEL="${DRAFT_MODEL:-Huang2020/Qwen3-8B-Domino-b16}"
if [[ ! -v DRAFT_REVISION ]]; then
  DRAFT_REVISION="b2b249e3429fedbcb17c2166d3ac2161a047157b"
fi
TASKS="${TASKS:-gsm8k:32,humaneval:32,alpaca:32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
NUM_NPUS="${NUM_NPUS:-8}"
WARMUP_SAMPLES="${WARMUP_SAMPLES:-3}"
REPETITIONS="${REPETITIONS:-3}"
MASTER_PORT="${MASTER_PORT:-29641}"
MIN_SPEEDUP="${MIN_SPEEDUP:-1.05}"
ENFORCE="${ENFORCE:-0}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/outputs/ascend_$(date +%Y%m%d_%H%M%S)}"

TARGET_REVISION_ARGS=()
if [[ -n "${TARGET_REVISION}" ]]; then
  TARGET_REVISION_ARGS=(--model-revision "${TARGET_REVISION}")
fi
DRAFT_REVISION_ARGS=()
if [[ -n "${DRAFT_REVISION}" ]]; then
  DRAFT_REVISION_ARGS=(--draft-revision "${DRAFT_REVISION}")
fi

mkdir -p "${OUT_DIR}"
IFS=',' read -r -a TASK_ARRAY <<< "${TASKS}"
for task in "${TASK_ARRAY[@]}"; do
  IFS=':' read -r DATASET MAX_SAMPLES <<< "${task}"
  MANIFEST="${OUT_DIR}/${DATASET}_manifest.jsonl"
  BASELINE="${OUT_DIR}/${DATASET}_baseline.jsonl"
  DOMINO="${OUT_DIR}/${DATASET}_domino_b${BLOCK_SIZE}.jsonl"
  SUMMARY="${OUT_DIR}/${DATASET}_summary_b${BLOCK_SIZE}.json"

  "${PYTHON}" "${SCRIPT_DIR}/code/benchmark_portable.py" \
    --dataset "${DATASET}" --max-samples "${MAX_SAMPLES}" \
    --dump-benchmark-manifest "${MANIFEST}" --dump-only

  "${PYTHON}" -m torch.distributed.run \
    --nproc_per_node="${NUM_NPUS}" --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/code/benchmark_portable.py" \
    --mode baseline --device-backend npu --attn-implementation sdpa \
    --benchmark-manifest "${MANIFEST}" \
    --model-name-or-path "${TARGET_MODEL}" "${TARGET_REVISION_ARGS[@]}" \
    --target-dtype "${TARGET_DTYPE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --warmup-samples "${WARMUP_SAMPLES}" --repetitions "${REPETITIONS}" \
    --answer-file "${BASELINE}"

  "${PYTHON}" -m torch.distributed.run \
    --nproc_per_node="${NUM_NPUS}" --master_port="$((MASTER_PORT + 1))" \
    "${SCRIPT_DIR}/code/benchmark_portable.py" \
    --mode domino --device-backend npu --attn-implementation sdpa \
    --benchmark-manifest "${MANIFEST}" \
    --model-name-or-path "${TARGET_MODEL}" "${TARGET_REVISION_ARGS[@]}" \
    --target-dtype "${TARGET_DTYPE}" \
    --draft-name-or-path "${DRAFT_MODEL}" "${DRAFT_REVISION_ARGS[@]}" \
    --block-size "${BLOCK_SIZE}" --use-bias \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --warmup-samples "${WARMUP_SAMPLES}" --repetitions "${REPETITIONS}" \
    --answer-file "${DOMINO}"

  COMPARE_ARGS=(
    --baseline "${BASELINE}" --domino "${DOMINO}"
    --min-speedup "${MIN_SPEEDUP}" --output-json "${SUMMARY}"
  )
  if [[ "${ENFORCE}" == "1" ]]; then
    COMPARE_ARGS+=(--enforce)
  fi
  "${PYTHON}" "${SCRIPT_DIR}/code/compare_benchmarks.py" "${COMPARE_ARGS[@]}"
done

echo "Wrote Ascend benchmark outputs to ${OUT_DIR}"
