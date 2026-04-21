#!/bin/bash

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DINO_ROOT="$(cd "${COMMON_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${DINO_ROOT}/.." && pwd)"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-dinov3_stack}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6}"
DATA_ROOT="${DATA_ROOT:-/mnt/30TB_SSD/heesu/dataset/cag_vision_fm/images}"
OUT_ROOT="${OUT_ROOT:-output/h100/1_pretrain}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-8}"
TRAIN_PIN_MEMORY="${TRAIN_PIN_MEMORY:-false}"
DRY_RUN="${DRY_RUN:-0}"
MODELS="${MODELS:-vits vitb vitl}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
ENABLE_DISTRIBUTED_DIAGNOSTICS="${ENABLE_DISTRIBUTED_DIAGNOSTICS:-1}"
TORCH_DISTRIBUTED_DEBUG_LEVEL="${TORCH_DISTRIBUTED_DEBUG_LEVEL:-DETAIL}"
NCCL_DEBUG_LEVEL="${NCCL_DEBUG_LEVEL:-INFO}"
NCCL_DEBUG_SUBSYS_LIST="${NCCL_DEBUG_SUBSYS_LIST:-INIT,COLL}"
ENABLE_NCCL_DEBUG_FILE="${ENABLE_NCCL_DEBUG_FILE:-1}"
TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"
PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"

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

cuda_device_count() {
  local count=0
  local device
  for device in ${CUDA_VISIBLE_DEVICES//,/ }; do
    [[ -n "${device}" ]] || continue
    count=$((count + 1))
  done
  if [[ "${count}" -eq 0 ]]; then
    count=1
  fi
  printf '%s\n' "${count}"
}

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
    stage1)
      printf '80\n'
      ;;
    stage2)
      printf '80\n'
      ;;
    stage3)
      printf '28\n'
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
  printf '%s\n' "$(stage_output_dir stage1)/eval/training_124999/sharded_teacher_checkpoint"
}

default_stage2_teacher_ckpt() {
  printf '%s\n' "$(stage_output_dir stage2)/eval/training_99999/sharded_teacher_checkpoint"
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
  [[ -f "${path}" || -d "${path}" ]] || die "Required checkpoint not found: ${path}"
}

normalize_interrupted_checkpoints() {
  local output_dir="$1"
  local ckpt_root="${output_dir}/ckpt"
  local ckpt_dir
  local ckpt_name
  local canonical_name
  local canonical_path
  local incomplete_path

  [[ -d "${ckpt_root}" ]] || return 0

  shopt -s nullglob
  for ckpt_dir in "${ckpt_root}"/*; do
    [[ -d "${ckpt_dir}" ]] || continue
    ckpt_name="$(basename "${ckpt_dir}")"

    # Normal checkpoints use pure integer directory names and must contain
    # a complete distributed checkpoint payload, including .metadata.
    if [[ "${ckpt_name}" =~ ^[0-9]+$ ]]; then
      if [[ ! -f "${ckpt_dir}/.metadata" ]] && compgen -G "${ckpt_dir}/*.distcp" > /dev/null; then
        incomplete_path="${ckpt_dir}.incomplete"
        if [[ ! -e "${incomplete_path}" ]]; then
          echo "    quarantining incomplete checkpoint ${ckpt_dir} -> ${incomplete_path}"
          if [[ "${DRY_RUN}" != "1" ]]; then
            mv "${ckpt_dir}" "${incomplete_path}"
          fi
        fi
      fi
      continue
    fi

    # Interrupted runs can leave a fully written distcp directory with a tempfile suffix
    # such as "4999somt6kv0". Canonicalize it back to "4999" so resume can detect it.
    if [[ "${ckpt_name}" =~ ^([0-9]+)[A-Za-z0-9._-]+$ ]] \
      && [[ -f "${ckpt_dir}/.metadata" ]] \
      && compgen -G "${ckpt_dir}/*.distcp" > /dev/null; then
      canonical_name="${BASH_REMATCH[1]}"
      canonical_path="${ckpt_root}/${canonical_name}"
      if [[ ! -e "${canonical_path}" ]]; then
        echo "    normalizing interrupted checkpoint ${ckpt_dir} -> ${canonical_path}"
        if [[ "${DRY_RUN}" != "1" ]]; then
          mv "${ckpt_dir}" "${canonical_path}"
        fi
      fi
    fi
  done
  shopt -u nullglob
}

launch_h100_stage() {
  local stage="$1"
  shift

  local output_dir
  output_dir="$(stage_output_dir "${stage}")"
  if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "${output_dir}/logs"
  fi
  normalize_interrupted_checkpoints "${output_dir}"
  local nproc_per_node
  nproc_per_node="${NPROC_PER_NODE:-$(cuda_device_count)}"
  local -a launcher
  if [[ "${nproc_per_node}" -gt 1 ]]; then
    launcher=(
      conda run -n "${CONDA_ENV_NAME}"
      torchrun
      --standalone
      --nproc_per_node="${nproc_per_node}"
      -m
      dinov3.train.train
    )
  else
    launcher=(
      conda run -n "${CONDA_ENV_NAME}"
      python
      dinov3/train/train.py
    )
  fi

  local -a env_args=(
    "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    "PYTHONPATH=${DINO_ROOT}"
    "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
  )
  if [[ "${ENABLE_DISTRIBUTED_DIAGNOSTICS}" == "1" ]]; then
    env_args+=(
      "TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG_LEVEL}"
      "NCCL_DEBUG=${NCCL_DEBUG_LEVEL}"
      "NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS_LIST}"
      "TORCH_SHOW_CPP_STACKTRACES=${TORCH_SHOW_CPP_STACKTRACES}"
      "PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER}"
    )
    if [[ "${ENABLE_NCCL_DEBUG_FILE}" == "1" ]]; then
      env_args+=("NCCL_DEBUG_FILE=${output_dir}/logs/nccl.%h.%p.log")
    fi
  fi

  local -a cmd=(
    env
    "${env_args[@]}"
    "${launcher[@]}"
    --config-file "$(stage_config_file "${stage}")"
    --output-dir "${output_dir}"
    "train.dataset_path=$(stage_dataset_path "${stage}")"
    "train.batch_size_per_gpu=$(stage_batch_per_gpu "${stage}")"
    "train.num_workers=${TRAIN_NUM_WORKERS}"
    "train.pin_memory=${TRAIN_PIN_MEMORY}"
    "train.sharded_eval_checkpoint=true"
    "student.arch=${STUDENT_ARCH}"
    "student.fp8_enabled=false"
    "student.fp8_filter=blocks"
    "train.compile=false"
  )
  cmd+=("$@")

  echo "==> ${stage} | ${MODEL_NAME}"
  echo "    output_dir=${output_dir}"
  echo "    cuda_visible_devices=${CUDA_VISIBLE_DEVICES} | nproc_per_node=${nproc_per_node}"
  if [[ "${ENABLE_DISTRIBUTED_DIAGNOSTICS}" == "1" ]]; then
    echo "    distributed diagnostics: TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG_LEVEL}, NCCL_DEBUG=${NCCL_DEBUG_LEVEL}, NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS_LIST}"
    if [[ "${ENABLE_NCCL_DEBUG_FILE}" == "1" ]]; then
      echo "    nccl debug files: ${output_dir}/logs/nccl.%h.%p.log"
    fi
  fi
  run_or_print "${cmd[@]}"
}
