#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
set -e
set -o pipefail

EVAL_DIR="output/train/2_imagenet1k/1_stage1_pretraining/eval"
OUT_ROOT="output/train/2_imagenet1k/1_stage1-2_k-NN"
CONFIG_FILE="output/train/2_imagenet1k/1_stage1_pretraining/config.yaml"
TRAIN_DATASET="ImageNet:split=TRAIN:root=/mnt/c/Users/heesu/workspace/dinov3_stack/input/imagenet-1k:extra=/mnt/c/Users/heesu/workspace/dinov3_stack/input/imagenet-1k"
VAL_DATASET="ImageNet:split=VAL:root=/mnt/c/Users/heesu/workspace/dinov3_stack/input/imagenet-1k:extra=/mnt/c/Users/heesu/workspace/dinov3_stack/input/imagenet-1k"
BATCH_SIZE=1024
NUM_WORKERS=4

for dir in $(ls -d "${EVAL_DIR}"/training_* 2>/dev/null | sort -V); do
  ckpt="${dir}/teacher_checkpoint.pth"
  [ -e "${ckpt}" ] || continue
  iter_dir="$(basename "${dir}")"
  iter="${iter_dir#training_}"
  out_dir="${OUT_ROOT}/${iter}"

  echo "Running k-NN eval for ${iter_dir}"
  PYTHONPATH=${PWD} python dinov3/eval/knn.py \
    model.config_file="${CONFIG_FILE}" \
    model.pretrained_weights="${ckpt}" \
    output_dir="${out_dir}" \
    train.dataset="${TRAIN_DATASET}" \
    eval.test_dataset="${VAL_DATASET}" \
    train.batch_size="${BATCH_SIZE}" \
    eval.batch_size="${BATCH_SIZE}" \
    train.num_workers="${NUM_WORKERS}" \
    eval.num_workers="${NUM_WORKERS}"
done
