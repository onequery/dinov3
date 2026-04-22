#!/bin/bash

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DINO_ROOT="$(cd "${COMMON_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${DINO_ROOT}/.." && pwd)"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-dinov3_stack}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/home/gmail_asse/anaconda3/envs/${CONDA_ENV_NAME}}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PREFIX}/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
DATA_ROOT="${DATA_ROOT:-/NHNHOME/WORKSPACE/0526040018_A/heesu/cag_vision_fm/images}"
OUT_ROOT="${OUT_ROOT:-output/b200/1_pretrain}"
MODEL="${MODEL:-vit7b}"
MODELS="${MODELS:-vit7b}"
DRY_RUN="${DRY_RUN:-0}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${DINO_ROOT}/output/torchinductor_cache}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TORCHINDUCTOR_CACHE_DIR}/triton}"
TORCHINDUCTOR_FX_GRAPH_CACHE="${TORCHINDUCTOR_FX_GRAPH_CACHE:-0}"
TORCHINDUCTOR_AUTOGRAD_CACHE="${TORCHINDUCTOR_AUTOGRAD_CACHE:-0}"
TORCHINDUCTOR_DISABLE_MULTI_KERNEL_CACHE="${TORCHINDUCTOR_DISABLE_MULTI_KERNEL_CACHE:-1}"

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
      ;;
    *)
      die "MODEL must be vit7b for the B200 launcher, got: ${model}"
      ;;
  esac

  export MODEL_NAME
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

config_scalar() {
  local stage="$1"
  local key="$2"
  local config_file
  config_file="$(stage_config_file "${stage}")"
  [[ -x "${PYTHON_BIN}" ]] || die "PYTHON_BIN is not executable: ${PYTHON_BIN}"
  "${PYTHON_BIN}" -c '
import sys
from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
node = cfg
for part in sys.argv[2].split("."):
    node = node[part]
print(node)
' "${config_file}" "${key}"
}

stage_final_iteration() {
  local stage="$1"
  local epochs
  local epoch_length
  epochs="$(config_scalar "${stage}" "optim.epochs")"
  epoch_length="$(config_scalar "${stage}" "train.OFFICIAL_EPOCH_LENGTH")"
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

  local config_file
  config_file="$(stage_config_file "${stage}")"

  local -a cmd=(
    env
    "${env_args[@]}"
    "${launcher[@]}"
    --config-file "${config_file}"
    --output-dir "${output_dir}"
    "train.dataset_path=$(stage_dataset_path "${stage}")"
  )
  cmd+=("$@")

  local yaml_batch
  local yaml_workers
  local yaml_pin_memory
  local yaml_fp8
  local yaml_compile
  local yaml_checkpointing_full
  local yaml_epochs
  local yaml_epoch_length
  yaml_batch="$(config_scalar "${stage}" "train.batch_size_per_gpu")"
  yaml_workers="$(config_scalar "${stage}" "train.num_workers")"
  yaml_pin_memory="$(config_scalar "${stage}" "train.pin_memory")"
  yaml_fp8="$(config_scalar "${stage}" "student.fp8_enabled")"
  yaml_compile="$(config_scalar "${stage}" "train.compile")"
  yaml_checkpointing_full="$(config_scalar "${stage}" "train.checkpointing_full")"
  yaml_epochs="$(config_scalar "${stage}" "optim.epochs")"
  yaml_epoch_length="$(config_scalar "${stage}" "train.OFFICIAL_EPOCH_LENGTH")"

  echo "==> ${stage} | ${MODEL_NAME}"
  echo "    config_file=${config_file}"
  echo "    output_dir=${output_dir}"
  echo "    cuda_visible_devices=${CUDA_VISIBLE_DEVICES} | nproc_per_node=${nproc_per_node}"
  echo "    YAML recipe: batch_per_gpu=${yaml_batch} | workers=${yaml_workers} | fp8=${yaml_fp8} | compile=${yaml_compile} | pin_memory=${yaml_pin_memory} | checkpointing_full=${yaml_checkpointing_full}"
  echo "    epochs=${yaml_epochs} | epoch_length=${yaml_epoch_length} | final_eval=training_$(stage_final_iteration "${stage}")"
  if [[ "${ENABLE_DISTRIBUTED_DIAGNOSTICS}" == "1" ]]; then
    echo "    distributed diagnostics: TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG_LEVEL}, NCCL_DEBUG=${NCCL_DEBUG_LEVEL}, NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS_LIST}"
    if [[ "${ENABLE_NCCL_DEBUG_FILE}" == "1" ]]; then
      echo "    nccl debug files: ${output_dir}/logs/nccl.%h.%p.log"
    fi
  fi
  run_or_print "${cmd[@]}"
}
