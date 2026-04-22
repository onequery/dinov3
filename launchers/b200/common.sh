#!/bin/bash

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DINO_ROOT="$(cd "${COMMON_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${DINO_ROOT}/.." && pwd)"

RECIPE_ENV="${RECIPE_ENV:-${COMMON_DIR}/recipe.env}"
if [[ -f "${RECIPE_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${RECIPE_ENV}"
fi

CONDA_ENV_NAME="${CONDA_ENV_NAME:-dinov3_stack}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/home/gmail_asse/anaconda3/envs/${CONDA_ENV_NAME}}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PREFIX}/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
DATA_ROOT="${DATA_ROOT:-/NHNHOME/WORKSPACE/0526040018_A/heesu/cag_vision_fm/images}"
OUT_ROOT="${OUT_ROOT:-output/b200/1_pretrain}"
MODEL="${MODEL:-vit7b}"
MODELS="${MODELS:-vit7b}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-8}"
TRAIN_PIN_MEMORY="${TRAIN_PIN_MEMORY:-false}"
TRAIN_FP8="${TRAIN_FP8:-true}"
TRAIN_COMPILE="${TRAIN_COMPILE:-true}"
DRY_RUN="${DRY_RUN:-0}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${DINO_ROOT}/output/torchinductor_cache}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TORCHINDUCTOR_CACHE_DIR}/triton}"
TORCHINDUCTOR_FX_GRAPH_CACHE="${TORCHINDUCTOR_FX_GRAPH_CACHE:-0}"
TORCHINDUCTOR_AUTOGRAD_CACHE="${TORCHINDUCTOR_AUTOGRAD_CACHE:-0}"
TORCHINDUCTOR_DISABLE_MULTI_KERNEL_CACHE="${TORCHINDUCTOR_DISABLE_MULTI_KERNEL_CACHE:-1}"

STAGE1_BATCH_PER_GPU="${STAGE1_BATCH_PER_GPU:-16}"
STAGE2_BATCH_PER_GPU="${STAGE2_BATCH_PER_GPU:-16}"
STAGE3_BATCH_PER_GPU="${STAGE3_BATCH_PER_GPU:-8}"
STAGE1_NUM_WORKERS="${STAGE1_NUM_WORKERS:-${TRAIN_NUM_WORKERS}}"
STAGE2_NUM_WORKERS="${STAGE2_NUM_WORKERS:-${TRAIN_NUM_WORKERS}}"
STAGE3_NUM_WORKERS="${STAGE3_NUM_WORKERS:-${TRAIN_NUM_WORKERS}}"
STAGE1_PIN_MEMORY="${STAGE1_PIN_MEMORY:-${TRAIN_PIN_MEMORY}}"
STAGE2_PIN_MEMORY="${STAGE2_PIN_MEMORY:-${TRAIN_PIN_MEMORY}}"
STAGE3_PIN_MEMORY="${STAGE3_PIN_MEMORY:-${TRAIN_PIN_MEMORY}}"
STAGE1_FP8="${STAGE1_FP8:-${TRAIN_FP8}}"
STAGE2_FP8="${STAGE2_FP8:-${TRAIN_FP8}}"
STAGE3_FP8="${STAGE3_FP8:-${TRAIN_FP8}}"
STAGE1_COMPILE="${STAGE1_COMPILE:-${TRAIN_COMPILE}}"
STAGE2_COMPILE="${STAGE2_COMPILE:-${TRAIN_COMPILE}}"
STAGE3_COMPILE="${STAGE3_COMPILE:-${TRAIN_COMPILE}}"
STAGE1_CHECKPOINTING_FULL="${STAGE1_CHECKPOINTING_FULL:-}"
STAGE2_CHECKPOINTING_FULL="${STAGE2_CHECKPOINTING_FULL:-}"
STAGE3_CHECKPOINTING_FULL="${STAGE3_CHECKPOINTING_FULL:-}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-100}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-100}"
STAGE3_EPOCHS="${STAGE3_EPOCHS:-30}"
STAGE1_EPOCH_LENGTH="${STAGE1_EPOCH_LENGTH:-1250}"
STAGE2_EPOCH_LENGTH="${STAGE2_EPOCH_LENGTH:-1000}"
STAGE3_EPOCH_LENGTH="${STAGE3_EPOCH_LENGTH:-1000}"
STAGE1_EVAL_PERIOD="${STAGE1_EVAL_PERIOD:-5000}"
STAGE2_EVAL_PERIOD="${STAGE2_EVAL_PERIOD:-5000}"
STAGE3_EVAL_PERIOD="${STAGE3_EVAL_PERIOD:-5000}"

CHECKPOINT_PERIOD="${CHECKPOINT_PERIOD:-2500}"
CHECKPOINT_MAX_TO_KEEP="${CHECKPOINT_MAX_TO_KEEP:-2}"
CHECKPOINT_KEEP_EVERY="${CHECKPOINT_KEEP_EVERY:-50000}"

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

bool_value() {
  case "${1}" in
    1|true|TRUE|True|yes|YES|Yes)
      printf 'true\n'
      ;;
    0|false|FALSE|False|no|NO|No)
      printf 'false\n'
      ;;
    *)
      die "Expected boolean-like value, got: ${1}"
      ;;
  esac
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
    vit7b|dinov3_vit7b16)
      MODEL_NAME="dinov3_vit7b16"
      STUDENT_ARCH="vit_7b"
      CONTENT_STATE_HEAD_HIDDEN_DIM="4096"
      CONTENT_STATE_HEAD_OUT_DIM="4096"
      CONTENT_STATE_DECODER_HIDDEN_DIM="8192"
      ;;
    *)
      die "MODEL must be vit7b for the B200 launcher, got: ${model}"
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
      printf '%s\n' "${DINO_ROOT}/dinov3/configs/train/dinov3_b200_vit7b16_cag_pretrain.yaml"
      ;;
    stage2)
      printf '%s\n' "${DINO_ROOT}/dinov3/configs/train/dinov3_b200_vit7b16_cag_gram_anchor.yaml"
      ;;
    stage3)
      printf '%s\n' "${DINO_ROOT}/dinov3/configs/train/dinov3_b200_vit7b16_cag_high_res_adapt_content_state.yaml"
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
      printf '%s\n' "${STAGE1_BATCH_PER_GPU}"
      ;;
    stage2)
      printf '%s\n' "${STAGE2_BATCH_PER_GPU}"
      ;;
    stage3)
      printf '%s\n' "${STAGE3_BATCH_PER_GPU}"
      ;;
    *)
      die "Unknown stage: ${stage}"
      ;;
  esac
}

stage_num_workers() {
  local stage="$1"
  case "${stage}" in
    stage1) printf '%s\n' "${STAGE1_NUM_WORKERS}" ;;
    stage2) printf '%s\n' "${STAGE2_NUM_WORKERS}" ;;
    stage3) printf '%s\n' "${STAGE3_NUM_WORKERS}" ;;
    *) die "Unknown stage: ${stage}" ;;
  esac
}

stage_pin_memory() {
  local stage="$1"
  case "${stage}" in
    stage1) printf '%s\n' "${STAGE1_PIN_MEMORY}" ;;
    stage2) printf '%s\n' "${STAGE2_PIN_MEMORY}" ;;
    stage3) printf '%s\n' "${STAGE3_PIN_MEMORY}" ;;
    *) die "Unknown stage: ${stage}" ;;
  esac
}

stage_fp8() {
  local stage="$1"
  case "${stage}" in
    stage1) printf '%s\n' "${STAGE1_FP8}" ;;
    stage2) printf '%s\n' "${STAGE2_FP8}" ;;
    stage3) printf '%s\n' "${STAGE3_FP8}" ;;
    *) die "Unknown stage: ${stage}" ;;
  esac
}

stage_compile() {
  local stage="$1"
  case "${stage}" in
    stage1) printf '%s\n' "${STAGE1_COMPILE}" ;;
    stage2) printf '%s\n' "${STAGE2_COMPILE}" ;;
    stage3) printf '%s\n' "${STAGE3_COMPILE}" ;;
    *) die "Unknown stage: ${stage}" ;;
  esac
}

stage_checkpointing_full() {
  local stage="$1"
  case "${stage}" in
    stage1) printf '%s\n' "${STAGE1_CHECKPOINTING_FULL}" ;;
    stage2) printf '%s\n' "${STAGE2_CHECKPOINTING_FULL}" ;;
    stage3) printf '%s\n' "${STAGE3_CHECKPOINTING_FULL}" ;;
    *) die "Unknown stage: ${stage}" ;;
  esac
}

stage_epochs() {
  local stage="$1"
  case "${stage}" in
    stage1) printf '%s\n' "${STAGE1_EPOCHS}" ;;
    stage2) printf '%s\n' "${STAGE2_EPOCHS}" ;;
    stage3) printf '%s\n' "${STAGE3_EPOCHS}" ;;
    *) die "Unknown stage: ${stage}" ;;
  esac
}

stage_epoch_length() {
  local stage="$1"
  case "${stage}" in
    stage1) printf '%s\n' "${STAGE1_EPOCH_LENGTH}" ;;
    stage2) printf '%s\n' "${STAGE2_EPOCH_LENGTH}" ;;
    stage3) printf '%s\n' "${STAGE3_EPOCH_LENGTH}" ;;
    *) die "Unknown stage: ${stage}" ;;
  esac
}

stage_eval_period() {
  local stage="$1"
  case "${stage}" in
    stage1) printf '%s\n' "${STAGE1_EVAL_PERIOD}" ;;
    stage2) printf '%s\n' "${STAGE2_EVAL_PERIOD}" ;;
    stage3) printf '%s\n' "${STAGE3_EVAL_PERIOD}" ;;
    *) die "Unknown stage: ${stage}" ;;
  esac
}

stage_final_iteration() {
  local stage="$1"
  local epochs
  local epoch_length
  epochs="$(stage_epochs "${stage}")"
  epoch_length="$(stage_epoch_length "${stage}")"
  printf '%s\n' "$((epochs * epoch_length - 1))"
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
  printf '%s\n' "$(stage_output_dir stage1)/eval/training_$(stage_final_iteration stage1)/sharded_teacher_checkpoint"
}

default_stage2_teacher_ckpt() {
  printf '%s\n' "$(stage_output_dir stage2)/eval/training_$(stage_final_iteration stage2)/sharded_teacher_checkpoint"
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

    if [[ "${ckpt_name}" =~ ^([0-9]+)[A-Za-z0-9]+$ ]] \
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

launch_b200_stage() {
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
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    die "PYTHON_BIN is not executable: ${PYTHON_BIN}"
  fi
  local -a launcher
  if [[ "${nproc_per_node}" -gt 1 ]]; then
    launcher=(
      "${PYTHON_BIN}"
      -m
      torch.distributed.run
      --standalone
      --nproc_per_node="${nproc_per_node}"
      -m
      dinov3.train.train
    )
  else
    launcher=(
      "${PYTHON_BIN}"
      -m
      dinov3.train.train
    )
  fi

  local -a env_args=(
    "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    "PYTHONPATH=${DINO_ROOT}"
    "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
    "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
    "TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR}"
    "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
    "TORCHINDUCTOR_FX_GRAPH_CACHE=${TORCHINDUCTOR_FX_GRAPH_CACHE}"
    "TORCHINDUCTOR_AUTOGRAD_CACHE=${TORCHINDUCTOR_AUTOGRAD_CACHE}"
    "TORCHINDUCTOR_DISABLE_MULTI_KERNEL_CACHE=${TORCHINDUCTOR_DISABLE_MULTI_KERNEL_CACHE}"
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
    "train.num_workers=$(stage_num_workers "${stage}")"
    "train.pin_memory=$(bool_value "$(stage_pin_memory "${stage}")")"
    "train.sharded_eval_checkpoint=true"
    "train.compile=$(bool_value "$(stage_compile "${stage}")")"
    "train.OFFICIAL_EPOCH_LENGTH=$(stage_epoch_length "${stage}")"
    "optim.epochs=$(stage_epochs "${stage}")"
    "evaluation.eval_period_iterations=$(stage_eval_period "${stage}")"
    "checkpointing.period=${CHECKPOINT_PERIOD}"
    "checkpointing.max_to_keep=${CHECKPOINT_MAX_TO_KEEP}"
    "checkpointing.keep_every=${CHECKPOINT_KEEP_EVERY}"
    "student.arch=${STUDENT_ARCH}"
    "student.fp8_enabled=$(bool_value "$(stage_fp8 "${stage}")")"
    "student.fp8_filter=blocks"
  )
  local checkpointing_full
  checkpointing_full="$(stage_checkpointing_full "${stage}")"
  if [[ -n "${checkpointing_full}" ]]; then
    cmd+=("train.checkpointing_full=$(bool_value "${checkpointing_full}")")
  fi
  cmd+=("$@")

  echo "==> ${stage} | ${MODEL_NAME}"
  echo "    output_dir=${output_dir}"
  echo "    cuda_visible_devices=${CUDA_VISIBLE_DEVICES} | nproc_per_node=${nproc_per_node}"
  echo "    batch_per_gpu=$(stage_batch_per_gpu "${stage}") | workers=$(stage_num_workers "${stage}") | fp8=$(bool_value "$(stage_fp8 "${stage}")") | compile=$(bool_value "$(stage_compile "${stage}")") | pin_memory=$(bool_value "$(stage_pin_memory "${stage}")")"
  if [[ -n "${checkpointing_full}" ]]; then
    echo "    checkpointing_full=$(bool_value "${checkpointing_full}")"
  fi
  echo "    epochs=$(stage_epochs "${stage}") | epoch_length=$(stage_epoch_length "${stage}") | final_eval=training_$(stage_final_iteration "${stage}")"
  if [[ "${ENABLE_DISTRIBUTED_DIAGNOSTICS}" == "1" ]]; then
    echo "    distributed diagnostics: TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG_LEVEL}, NCCL_DEBUG=${NCCL_DEBUG_LEVEL}, NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS_LIST}"
    if [[ "${ENABLE_NCCL_DEBUG_FILE}" == "1" ]]; then
      echo "    nccl debug files: ${output_dir}/logs/nccl.%h.%p.log"
    fi
  fi
  run_or_print "${cmd[@]}"
}
