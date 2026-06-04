#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SSC_EXPLORATION_DIR="${WORKSPACE_DIR}/ssc_exploration"

CHECKPOINT_DIR="${WORKSPACE_DIR}/checkpoints/full_train"
LOG_DIR="${WORKSPACE_DIR}/logs/full_train"
MODEL_NAME="${MODEL_NAME:-SSCNet_NYU_full_train}"
LOG_FILE="${LOG_DIR}/${MODEL_NAME}.log"

mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}"

export PYTHONPATH="${SSC_EXPLORATION_DIR}:${PYTHONPATH:-}"

cd "${SCRIPT_DIR}"

python ./train.py \
  --model=sscnet \
  --dataset=nyu \
  --epochs="${EPOCHS:-50}" \
  --batch_size="${BATCH_SIZE:-1}" \
  --workers="${WORKERS:-1}" \
  --lr="${LR:-0.01}" \
  --lr_adj_n="${LR_ADJ_N:-10}" \
  --lr_adj_rate="${LR_ADJ_RATE:-0.1}" \
  --checkpoint="${CHECKPOINT_DIR}/" \
  --model_name="${MODEL_NAME}" \
  2>&1 | tee "${LOG_FILE}"
