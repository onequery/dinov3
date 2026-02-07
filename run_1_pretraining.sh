#!/bin/bash

set -e
set -o pipefail

# =================================================
# MainGear settings (WSL, Single GPU)
# =================================================
# export CUDA_VISIBLE_DEVICES=0

# PYTHONPATH=${PWD} python dinov3/train/train.py \
#   --config-file dinov3/configs/train/dinov3_vits16_pretrain_rtx3080laptop.yaml \
#   --output-dir output/train/2_imagenet1k/1_stage1_pretraining \
#   train.dataset_path=ImageNet:split=TRAIN:root=/mnt/c/Users/heesu/workspace/dinov3_stack/input/imagenet-1k:extra=/mnt/c/Users/heesu/workspace/dinov3_stack/input/imagenet-1k

# =================================================
# A6000 settings (Single GPU)
# =================================================
# ------------------------------------
# 1. Pretraining - ImageNet-1k
# ------------------------------------
# export CUDA_VISIBLE_DEVICES=0

# PYTHONPATH=${PWD} python dinov3/train/train.py \
#   --config-file dinov3/configs/train/dinov3_vits16_pretrain_a6000.yaml \
#   --output-dir output/a6000/1_pretrain/dinov3_vits16/2_imagenet1k/1_stage1_pretrain \
#   train.dataset_path=ImageNet:split=TRAIN:root=/mnt/nas/external/public/raw/imagenet-1k:extra=/mnt/nas/external/public/raw/imagenet-1k

# ------------------------------------
# 2. Pretraining - CAG-Contrast-FM-3M
# ------------------------------------
export CUDA_VISIBLE_DEVICES=0

PYTHONPATH=${PWD} python dinov3/train/train.py \
  --config-file dinov3/configs/train/dinov3_vits16_pretrain_a6000.yaml \
  --output-dir output/a6000/1_pretrain/dinov3_vits16/3_cagcontfm3m/1_stage1_pretrain \
  train.dataset_path=CAGContrastFM3M:split=TRAIN:root=/mnt/nas/snubhcvc/project/cag_fm/pretrain/datasets/images

# =================================================
# A6000 settings (Multi-GPU)
# =================================================
# PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/train/train.py \
#   --nodes 1 \
#   --ngpus 1 \
#   --config-file dinov3/configs/train/dinov3_vits16_pretrain.yaml \
#   --output-dir output/train/1_stage1_exp_pretraining \
#   train.dataset_path=ImageNet:split=TRAIN:root=/mnt/nas/external/public/raw/imagenet-1k:extra=/mnt/nas/external/public/raw/imagenet-1k
