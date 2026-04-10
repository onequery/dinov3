#!/bin/bash

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DINO_ROOT="$(cd "${COMMON_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${DINO_ROOT}/.." && pwd)"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-dinov3_stack}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
DATA_ROOT="${DATA_ROOT:-/mnt/30TB_SSD/heesu/cag_vision_fm/images}"
OUT_ROOT="${OUT_ROOT:-output/h100/1_pretrain}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-8}"
DRY_RUN="${DRY_RUN:-0}"
MODELS="${MODELS:-vits vitb vitl}"

die() {
  echo "Error: $*" >&2
  exit 1
}

resolve_output_root() {
  if [[ "${OUT_ROOT}" = /* ]]; then
    printf '%s\n' "${OUT_ROOT}"
  else
    printf '%s\n' "${DINO_ROOT}/${OUT_ROOT}"
  fi
}

ABS_OUT_ROOT="$(resolve_output_root)"

set_model_spec() {
  local model="$1"
  case "${model}" in
    vits)
      MODEL_NAME="dinov3_vits16"
      STUDENT_ARCH="vit_small"
      CONTENT_STATE_HEAD_HIDDEN_DIM="384"
      CONTENT_STATE_HEAD_OUT_DIM="384"
      CONTENT_STATE_DECODER_HIDDEN_DIM="768"
      ;;
    vitb)
      MODEL_NAME="dinov3_vitb16"
      STUDENT_ARCH="vit_base"
      CONTENT_STATE_HEAD_HIDDEN_DIM="768"
      CONTENT_STATE_HEAD_OUT_DIM="768"
      CONTENT_STATE_DECODER_HIDDEN_DIM="1536"
      ;;
    vitl)
      MODEL_NAME="dinov3_vitl16"
      STUDENT_ARCH="vit_large"
      CONTENT_STATE_HEAD_HIDDEN_DIM="1024"
      CONTENT_STATE_HEAD_OUT_DIM="1024"
      CONTENT_STATE_DECODER_HIDDEN_DIM="2048"
      ;;
    *)
      die "MODEL must be one of: vits, vitb, vitl"
      ;;
  esac

  export MODEL_NAME
  export STUDENT_ARCH
  export CONTENT_STATE_HEAD_HIDDEN_DIM
  export CONTENT_STATE_HEAD_OUT_DIM
  export CONTENT_STATE_DECODER_HIDDEN_DIM
}

require_model() {
  [[ -n "${MODEL:-}" ]] || die "MODEL is required for this launcher."
  set_model_spec "${MODEL}"
}

stage_config_file() {
  local stage="$1"
  case "${stage}" in
    stage1)
      printf '%s\n' "${DINO_ROOT}/dinov3/configs/train/dinov3_h100_hybrid_cag_pretrain.yaml"
      ;;
    stage2)
      printf '%s\n' "${DINO_ROOT}/dinov3/configs/train/dinov3_h100_hybrid_cag_gram_anchor.yaml"
      ;;
    stage3)
      printf '%s\n' "${DINO_ROOT}/dinov3/configs/train/dinov3_h100_hybrid_cag_high_res_adapt_content_state.yaml"
      ;;
    *)
      die "Unknown stage: ${stage}"
      ;;
  esac
}

stage_dataset_path() {
  local stage="$1"
  case "${stage}" in
    stage1|stage2)
      printf 'CAGContrastFM3M:split=TRAIN:root=%s\n' "${DATA_ROOT}"
      ;;
    stage3)
      printf 'CAGContrastFMContinuityV1:split=TRAIN:root=%s\n' "${DATA_ROOT}"
      ;;
    *)
      die "Unknown stage: ${stage}"
      ;;
  esac
}

stage_batch_per_gpu() {
  local stage="$1"
  case "${stage}" in
    stage1|stage2)
      printf '72\n'
      ;;
    stage3)
      printf '24\n'
      ;;
    *)
      die "Unknown stage: ${stage}"
      ;;
  esac
}

stage_output_dir() {
  local stage="$1"
  local suffix=""
  case "${stage}" in
    stage1)
      suffix="1_stage1_pretrain"
      ;;
    stage2)
      suffix="2_stage2_gram_anchor"
      ;;
    stage3)
      suffix="3_stage3_high_res_adapt_content_state"
      ;;
    *)
      die "Unknown stage: ${stage}"
      ;;
  esac
  printf '%s\n' "${ABS_OUT_ROOT}/${MODEL_NAME}/3_cagcontfm3m/${suffix}"
}

default_stage1_teacher_ckpt() {
  printf '%s\n' "$(stage_output_dir stage1)/eval/training_124999/teacher_checkpoint.pth"
}

default_stage2_teacher_ckpt() {
  printf '%s\n' "$(stage_output_dir stage2)/eval/training_29999/teacher_checkpoint.pth"
}

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

run_or_print() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    print_command "$@"
  else
    "$@"
  fi
}

require_file_unless_dry_run() {
  local path="$1"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  [[ -f "${path}" ]] || die "Required checkpoint not found: ${path}"
}

launch_h100_stage() {
  local stage="$1"
  shift

  local output_dir
  output_dir="$(stage_output_dir "${stage}")"

  local -a cmd=(
    env
    "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    "PYTHONPATH=${DINO_ROOT}"
    conda run -n "${CONDA_ENV_NAME}" python
    dinov3/train/train.py
    --config-file "$(stage_config_file "${stage}")"
    --output-dir "${output_dir}"
    "train.dataset_path=$(stage_dataset_path "${stage}")"
    "train.batch_size_per_gpu=$(stage_batch_per_gpu "${stage}")"
    "train.num_workers=${TRAIN_NUM_WORKERS}"
    "train.sharded_eval_checkpoint=false"
    "student.arch=${STUDENT_ARCH}"
    "student.fp8_enabled=false"
    "student.fp8_filter=blocks"
    "train.compile=false"
  )
  cmd+=("$@")

  echo "==> ${stage} | ${MODEL_NAME}"
  echo "    output_dir=${output_dir}"
  run_or_print "${cmd[@]}"
}

