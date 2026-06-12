#!/usr/bin/env bash
# Discovery training-task smoke entry for Open-Assistant SFT (GPU / P 集群).
# galactica-125m + webgpt_dataset_only；周期性写 TB events，结果带时间戳目录。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PLATFORM_TRAIN_SMOKE=1
export PLATFORM_CALLBACK_DIR="${SCRIPT_DIR}"

OA_ROOT="${OPEN_ASSISTANT_ROOT:-/workspace/Open-Assistant}"
MT_DIR="${OA_ROOT}/model/model_training"

# Per-run dirs: /data/result/{RUN_ID}/checkpoints, /data/result/{RUN_ID}/train_result.json
RUN_ID="${TRAIN_RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
export TRAIN_RUN_ID="${RUN_ID}"
RESULTS_BASE="${RESULTS_BASE:-/data/result}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_BASE}/${RUN_ID}/checkpoints}"
export RESULTS_DIR="${RESULTS_DIR:-${RESULTS_BASE}/${RUN_ID}}"
mkdir -p "${RESULTS_DIR}" "${OUTPUT_DIR}"

# Platform injects TENSORBOARD_LOGDIR on train-events PVC; alias for scripts / docs.
export TB_LOGDIR="${TB_LOGDIR:-/var/log/training}"
if [[ -n "${TENSORBOARD_LOGDIR:-}" ]]; then
  mkdir -p "${TENSORBOARD_LOGDIR}"
  if [[ ! -e "${TB_LOGDIR}" ]]; then
    ln -sfn "${TENSORBOARD_LOGDIR}" "${TB_LOGDIR}"
  fi
else
  mkdir -p "${TB_LOGDIR}"
  echo "WARN: TENSORBOARD_LOGDIR unset; writing TB events to ${TB_LOGDIR} (local only, TB ksvc may not see them)" >&2
fi

export HF_HOME="${HF_HOME:-/data/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CLEAN_HF_CACHE_ON_EXIT="${CLEAN_HF_CACHE_ON_EXIT:-1}"

LOGGING_STEPS="${PLATFORM_LOGGING_STEPS:-5}"
NUM_EPOCHS="${OA_NUM_EPOCHS:-1}"
MAX_LENGTH="${OA_MAX_LENGTH:-128}"

echo "PLATFORM_TRAIN_SMOKE=1"
echo "TRAIN_RUN_ID=${TRAIN_RUN_ID}"
echo "TENSORBOARD_LOGDIR=${TENSORBOARD_LOGDIR:-<unset>}"
echo "TB_LOGDIR=${TB_LOGDIR} -> $(readlink -f "${TB_LOGDIR}" 2>/dev/null || echo "${TB_LOGDIR}")"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "RESULTS_DIR=${RESULTS_DIR}"
echo "PLATFORM_LOGGING_STEPS=${LOGGING_STEPS}"
echo "OPEN_ASSISTANT_ROOT=${OA_ROOT}"
echo "HF_HOME=${HF_HOME}"
echo "HF_ENDPOINT=${HF_ENDPOINT}"

cleanup_hf_cache() {
  local ec=$?
  if [[ "${CLEAN_HF_CACHE_ON_EXIT}" == "1" && -n "${HF_HOME}" && -d "${HF_HOME}" ]]; then
    echo "[cleanup] Removing Hugging Face cache under ${HF_HOME} (train exit=${ec})"
    rm -rf "${HF_HOME:?}"/*
  fi
  exit "${ec}"
}
trap cleanup_hf_cache EXIT

if [[ ! -f "${MT_DIR}/trainer_sft.py" ]]; then
  echo "ERROR: trainer_sft.py not found under ${MT_DIR}" >&2
  exit 1
fi

cd "${MT_DIR}"

# datasets>=4 不再支持 Hub loading script；webgpt 走 parquet revision。
QA_PY="${MT_DIR}/custom_datasets/qa_datasets.py"
if [[ -f "${QA_PY}" ]]; then
  sed -i 's/load_dataset("openai\/webgpt_comparisons")/load_dataset("openai\/webgpt_comparisons", revision="refs\/convert\/parquet")/' "${QA_PY}" || true
fi
python -m pip install -q 'datasets>=2.13.1,<4.0' 2>/dev/null || true

# galactica-125m (OPT) 无 rope_scaling / max_position_embeddings。
TRAINER_PY="${MT_DIR}/trainer_sft.py"
sed -i "s/model.config.rope_scaling/getattr(model.config, 'rope_scaling', None)/" "${TRAINER_PY}" || true
sed -i "s/model.config.max_position_embeddings/getattr(model.config, 'max_position_embeddings', None)/" "${TRAINER_PY}" || true
if ! grep -q 'PLATFORM_TRAIN_SMOKE' "${TRAINER_PY}"; then
  echo "Applying platform callback patch to trainer_sft.py (image 未内置时可离线生效)"
  patch -p0 -d "${MT_DIR}" < "${SCRIPT_DIR}/trainer_sft_platform.patch" || {
    echo "ERROR: failed to patch trainer_sft.py; rebuild training image with PLATFORM_TRAIN_SMOKE support" >&2
    exit 1
  }
fi
python3 -m py_compile "${TRAINER_PY}"

python3 trainer_sft.py \
  --configs galactica-125m webgpt_dataset_only \
  --log_wandb false \
  --num_train_epochs "${NUM_EPOCHS}" \
  --max_length "${MAX_LENGTH}" \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --logging_steps "${LOGGING_STEPS}" \
  --eval_steps 50 \
  --save_steps 999999 \
  --cache_dir "${HF_HOME}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
