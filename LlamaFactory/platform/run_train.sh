#!/bin/bash
# Discovery training-task smoke entry for LlamaFactory on a2-cann NPU.
set -euo pipefail

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

export PLATFORM_TRAIN_SMOKE=1
export HF_HOME="${HF_HOME:-/data/.cache/huggingface}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

# Per-run output dirs: /data/sft_output/{RUN_ID}, /data/data/{RUN_ID}/train_result.json
RUN_ID="${TRAIN_RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
export TRAIN_RUN_ID="${RUN_ID}"
SFT_OUTPUT_BASE="${SFT_OUTPUT_BASE:-/data/sft_output}"
RESULTS_BASE="${RESULTS_BASE:-/data/data}"
OUTPUT_DIR="${SFT_OUTPUT_BASE}/${RUN_ID}"
export RESULTS_DIR="${RESULTS_DIR:-${RESULTS_BASE}/${RUN_ID}}"
mkdir -p "${RESULTS_DIR}" "${OUTPUT_DIR}"

# Use platform-injected TB path when present; fallback alias is /var/log/training
export TB_LOGDIR="${TB_LOGDIR:-/var/log/training}"

# Resolve dataset_dir: CLI dataset_dir= (last wins) > $DATASET_DIR > /data/datasets
resolve_requested_dataset_dir() {
  local dir="${DATASET_DIR:-/data/datasets}"
  local arg
  for arg in "$@"; do
    if [[ "$arg" == dataset_dir=* ]]; then
      dir="${arg#dataset_dir=}"
    fi
  done
  printf '%s' "$dir"
}

BUNDLED_DATA_DIR="/app/data"
REQUESTED_DATA_DIR="$(resolve_requested_dataset_dir "$@")"

if [[ -f "${REQUESTED_DATA_DIR}/dataset_info.json" ]]; then
  FINAL_DATASET_DIR="${REQUESTED_DATA_DIR}"
  if [[ "${REQUESTED_DATA_DIR}" == "/data/datasets" ]]; then
    echo "Dataset: using platform mount ${FINAL_DATASET_DIR}"
  elif [[ -n "${DATASET_DIR:-}" && "${REQUESTED_DATA_DIR}" == "${DATASET_DIR}" ]]; then
    echo "Dataset: using DATASET_DIR=${FINAL_DATASET_DIR}"
  else
    echo "Dataset: using ${FINAL_DATASET_DIR} (dataset_info.json found)"
  fi
elif [[ -f "${BUNDLED_DATA_DIR}/dataset_info.json" ]]; then
  FINAL_DATASET_DIR="${BUNDLED_DATA_DIR}"
  echo "Dataset: ${REQUESTED_DATA_DIR}/dataset_info.json not found, using bundled ${FINAL_DATASET_DIR}"
else
  echo "ERROR: dataset_info.json missing under ${REQUESTED_DATA_DIR} and ${BUNDLED_DATA_DIR}" >&2
  exit 1
fi

export DATASET_DIR="${FINAL_DATASET_DIR}"

# Drop duplicate dataset_dir= from user args; we pass a single resolved value to llamafactory-cli.
USER_ARGS=()
for arg in "$@"; do
  if [[ "$arg" == dataset_dir=* ]]; then
    continue
  fi
  USER_ARGS+=("$arg")
done

cd /app

echo "PLATFORM_TRAIN_SMOKE=1"
echo "TRAIN_RUN_ID=${TRAIN_RUN_ID}"
echo "TENSORBOARD_LOGDIR=${TENSORBOARD_LOGDIR:-<unset, using TB_LOGDIR>}"
echo "TB_LOGDIR=${TB_LOGDIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "RESULTS_DIR=${RESULTS_DIR}"
echo "HF_HOME=${HF_HOME}"
echo "DATASET_DIR=${DATASET_DIR}"
echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"

exec llamafactory-cli train platform/train_npu_platform.yaml \
  "dataset_dir=${FINAL_DATASET_DIR}" \
  "output_dir=${OUTPUT_DIR}" \
  "${USER_ARGS[@]}"
