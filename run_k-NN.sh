#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
set -e
set -o pipefail

TRAIN_DATASET="ImageNet:split=TRAIN:root=/mnt/c/Users/heesu/workspace/dinov3_stack/input/imagenet-1k:extra=/mnt/c/Users/heesu/workspace/dinov3_stack/input/imagenet-1k"
VAL_DATASET="ImageNet:split=VAL:root=/mnt/c/Users/heesu/workspace/dinov3_stack/input/imagenet-1k:extra=/mnt/c/Users/heesu/workspace/dinov3_stack/input/imagenet-1k"
BATCH_SIZE=1024
NUM_WORKERS=4

# ===========================
# 1.3
# Backbone: ViT-S/16
# Pretraining dataset: LVD-1689M
# Stage: stage 3. High resolution adpatation

# OUT_ROOT="output/1_pretrain/dinov3_vits16/1_lvd1689m/3_stage3_high_res_adapt/knn"

# PYTHONPATH=${PWD} python dinov3/eval/knn.py \
#   model.dino_hub=dinov3_vits16 \
#   output_dir="${OUT_ROOT}" \
#   train.dataset="${TRAIN_DATASET}" \
#   eval.test_dataset="${VAL_DATASET}" \
#   train.batch_size="${BATCH_SIZE}" \
#   eval.batch_size="${BATCH_SIZE}" \
#   train.num_workers="${NUM_WORKERS}" \
#   eval.num_workers="${NUM_WORKERS}"

# # -----------------------------

# 2.1
# Backbone: ViT-S/16
# Pretraining dataset: ImageNet-1k
# Stage: stage 1. Pretraining
EVAL_CKPT_PATH="dinov3/output/a6000/1_pretrain/dinov3_vits16/2_imagenet1k/1_stage1_pretrain/eval/training_124999/teacher_checkpoint.pth"
OUT_ROOT="output/a6000/1_pretrain/dinov3_vits16/2_imagenet1k/1_stage1_pretrain/knn"
CONFIG_FILE="output/a6000/1_pretrain/dinov3_vits16/2_imagenet1k/1_stage1_pretrain/config.yaml"

PYTHONPATH=${PWD} python dinov3/eval/knn.py \
  model.config_file="${CONFIG_FILE}" \
  model.pretrained_weights="${EVAL_CKPT_PATH}" \
  output_dir="${OUT_ROOT}" \
  train.dataset="${TRAIN_DATASET}" \
  eval.test_dataset="${VAL_DATASET}" \
  train.batch_size="${BATCH_SIZE}" \
  eval.batch_size="${BATCH_SIZE}" \
  train.num_workers="${NUM_WORKERS}" \
  eval.num_workers="${NUM_WORKERS}"

# -----------------------------

# # 2.2
# # Backbone: ViT-S/16
# # Pretraining dataset: ImageNet-1k
# # Stage: stage 2. Gram anchoring

# EVAL_CKPT_PATH="output/1_pretrain/dinov3_vits16/2_imagenet1k/2_stage2_gram_anchor/eval/training_49999/teacher_checkpoint.pth"
# OUT_ROOT="output/1_pretrain/dinov3_vits16/2_imagenet1k/2_stage2_gram_anchor/knn"
# CONFIG_FILE="output/1_pretrain/dinov3_vits16/2_imagenet1k/2_stage2_gram_anchor/config.yaml"

# PYTHONPATH=${PWD} python dinov3/eval/knn.py \
#   model.config_file="${CONFIG_FILE}" \
#   model.pretrained_weights="${EVAL_CKPT_PATH}" \
#   output_dir="${OUT_ROOT}" \
#   train.dataset="${TRAIN_DATASET}" \
#   eval.test_dataset="${VAL_DATASET}" \
#   train.batch_size="${BATCH_SIZE}" \
#   eval.batch_size="${BATCH_SIZE}" \
#   train.num_workers="${NUM_WORKERS}" \
#   eval.num_workers="${NUM_WORKERS}"

# # -----------------------------

# # 2.3
# # Backbone: ViT-S/16
# # Pretraining dataset: ImageNet-1k
# # Stage: stage 3. High resolution adaptation

# EVAL_CKPT_PATH="output/1_pretrain/dinov3_vits16/2_imagenet1k/3_stage3_high_res_adapt/eval/training_29999/teacher_checkpoint.pth"
# OUT_ROOT="output/1_pretrain/dinov3_vits16/2_imagenet1k/3_stage3_high_res_adapt/knn"
# CONFIG_FILE="output/1_pretrain/dinov3_vits16/2_imagenet1k/3_stage3_high_res_adapt/config.yaml"

# PYTHONPATH=${PWD} python dinov3/eval/knn.py \
#   model.config_file="${CONFIG_FILE}" \
#   model.pretrained_weights="${EVAL_CKPT_PATH}" \
#   output_dir="${OUT_ROOT}" \
#   train.dataset="${TRAIN_DATASET}" \
#   eval.test_dataset="${VAL_DATASET}" \
#   train.batch_size="${BATCH_SIZE}" \
#   eval.batch_size="${BATCH_SIZE}" \
#   train.num_workers="${NUM_WORKERS}" \
#   eval.num_workers="${NUM_WORKERS}"
