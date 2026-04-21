#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

cd "${DINO_ROOT}"

run_stage_script() {
  local model="$1"
  local script_path="$2"

  env \
    MODEL="${model}" \
    CONDA_ENV_NAME="${CONDA_ENV_NAME}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    DATA_ROOT="${DATA_ROOT}" \
    OUT_ROOT="${OUT_ROOT}" \
    TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS}" \
    TRAIN_PIN_MEMORY="${TRAIN_PIN_MEMORY}" \
    TRAIN_FP8="${TRAIN_FP8}" \
    TRAIN_COMPILE="${TRAIN_COMPILE}" \
    DRY_RUN="${DRY_RUN}" \
    STAGE1_BATCH_PER_GPU="${STAGE1_BATCH_PER_GPU}" \
    STAGE2_BATCH_PER_GPU="${STAGE2_BATCH_PER_GPU}" \
    STAGE3_BATCH_PER_GPU="${STAGE3_BATCH_PER_GPU}" \
    STAGE1_NUM_WORKERS="${STAGE1_NUM_WORKERS}" \
    STAGE2_NUM_WORKERS="${STAGE2_NUM_WORKERS}" \
    STAGE3_NUM_WORKERS="${STAGE3_NUM_WORKERS}" \
    STAGE1_PIN_MEMORY="${STAGE1_PIN_MEMORY}" \
    STAGE2_PIN_MEMORY="${STAGE2_PIN_MEMORY}" \
    STAGE3_PIN_MEMORY="${STAGE3_PIN_MEMORY}" \
    STAGE1_FP8="${STAGE1_FP8}" \
    STAGE2_FP8="${STAGE2_FP8}" \
    STAGE3_FP8="${STAGE3_FP8}" \
    STAGE1_COMPILE="${STAGE1_COMPILE}" \
    STAGE2_COMPILE="${STAGE2_COMPILE}" \
    STAGE3_COMPILE="${STAGE3_COMPILE}" \
    STAGE1_TEACHER_CKPT="${STAGE1_TEACHER_CKPT:-}" \
    STAGE2_TEACHER_CKPT="${STAGE2_TEACHER_CKPT:-}" \
    "${script_path}"
}

for model in ${MODELS}; do
  set_model_spec "${model}"
  echo "=================================================="
  echo "B200 content-state pipeline | model=${MODEL_NAME}"
  echo "=================================================="
  run_stage_script "${model}" "${SCRIPT_DIR}/run_1_pretraining.sh"
  run_stage_script "${model}" "${SCRIPT_DIR}/run_2_gram_anchor.sh"
  run_stage_script "${model}" "${SCRIPT_DIR}/run_3_high-res-adapt_content_state.sh"
done
