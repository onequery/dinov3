#!/bin/bash

# MainGear settings (WSL, Single GPU)
export CUDA_VISIBLE_DEVICES=0
set -e
set -o pipefail
PYTHONPATH=${PWD} python dinov3/train/train.py \
    --config-file dinov3/configs/train/dinov3_vits16_gram_anchor.yaml \
    --output-dir output/train/2_stage2_exp_gram_anchoring \
    train.dataset_path=ImageNet:split=TRAIN:root=/mnt/c/Users/heesu/workspace/dinov3_stack/input/imagenet-1k:extra=/mnt/c/Users/heesu/workspace/dinov3_stack/input/imagenet-1k \
    gram.ckpt=