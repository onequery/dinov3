#!/bin/bash

# MainGear settings (WSL, Single GPU)
export CUDA_VISIBLE_DEVICES=0
set -e
set -o pipefail
PYTHONPATH=${PWD} python dinov3/train/train.py \
  --config-file dinov3/configs/train/dinov3_vits16_pretrain_rtx3080laptop.yaml \
  --output-dir output/train/1_stage1_exp_pretraining \
  train.dataset_path=ImageNet:split=TRAIN:root=/mnt/c/Users/heesu/workspace/dinov3_stack/input/imagenet-1k:extra=/mnt/c/Users/heesu/workspace/dinov3_stack/input/imagenet-1k

# A6000 settings (Test with Single GPU)
# export CUDA_VISIBLE_DEVICES=1
# set -e
# set -o pipefail
# PYTHONPATH=${PWD} python dinov3/train/train.py \
#   --config-file dinov3/configs/train/dinov3_vits16_pretrain.yaml \
#   --output-dir output/train/1_stage1_exp_pretraining \
#   train.dataset_path=ImageNet:split=TRAIN:root=/mnt/nas/external/public/raw/imagenet-1k:extra=/mnt/nas/external/public/raw/imagenet-1k

# # A6000 settings (Multi-GPU)
# PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/train/train.py \
#   --nodes 1 \
#   --ngpus 1 \
#   --config-file dinov3/configs/train/dinov3_vits16_pretrain.yaml \
#   --output-dir output/train/1_stage1_exp_pretraining \
#   train.dataset_path=ImageNet:split=TRAIN:root=/mnt/nas/external/public/raw/imagenet-1k:extra=/mnt/nas/external/public/raw/imagenet-1k