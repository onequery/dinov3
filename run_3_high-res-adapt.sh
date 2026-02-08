#!/bin/bash
set -e
set -o pipefail

# =====================================================
# MainGear settings (WSL, Single GPU)
# =====================================================
# export CUDA_VISIBLE_DEVICES=0

# PYTHONPATH=${PWD} python dinov3/train/train.py \
#     --config-file dinov3/configs/train/dinov3_vits16_high_res_adapt_rtx3080laptop.yaml \
#     --output-dir output/train/2_imagenet1k/3_stage3_high_res_adapt \
#     train.dataset_path=ImageNet:split=TRAIN:root=/mnt/c/Users/heesu/workspace/dinov3_stack/input/imagenet-1k:extra=/mnt/c/Users/heesu/workspace/dinov3_stack/input/imagenet-1k \
#     gram.ckpt=output/train/2_imagenet1k/2_stage2_gram_anchor/eval/training_49999/teacher_checkpoint.pth \
#     student.resume_from_teacher_chkpt=output/train/2_imagenet1k/2_stage2_gram_anchor/eval/training_49999/teacher_checkpoint.pth

# =====================================================
# A6000 settings (Single GPU)
# =====================================================
# -------------------------------------------
# 1. High-Res Adaptation - ImageNet-1k
# -------------------------------------------
# export CUDA_VISIBLE_DEVICES=0

# PYTHONPATH=${PWD} python dinov3/train/train.py \
#     --config-file dinov3/configs/train/dinov3_vits16_high_res_adapt_a6000.yaml \
#     --output-dir output/a6000/1_pretrain/dinov3_vits16/2_imagenet1k/3_stage3_high_res_adapt \
#     train.dataset_path=ImageNet:split=TRAIN:root=/mnt/nas/external/public/raw/imagenet-1k:extra=/mnt/nas/external/public/raw/imagenet-1k \
#     gram.ckpt=output/a6000/1_pretrain/dinov3_vits16/2_imagenet1k/2_stage2_gram_anchor/eval/training_29999/teacher_checkpoint.pth \
#     student.resume_from_teacher_chkpt=output/a6000/1_pretrain/dinov3_vits16/2_imagenet1k/2_stage2_gram_anchor/eval/training_29999/teacher_checkpoint.pth

# --------------------------------------------
# 2. High-Res Adaptation - CAS-Contrast-FM-3M
# --------------------------------------------
export CUDA_VISIBLE_DEVICES=0

PYTHONPATH=${PWD} python dinov3/train/train.py \
    --config-file dinov3/configs/train/dinov3_vits16_high_res_adapt_a6000.yaml \
    --output-dir output/a6000/1_pretrain/dinov3_vits16/3_cagcontfm3m/3_stage3_high_res_adapt \
    train.dataset_path=CAGContrastFM3M:split=TRAIN:root=/mnt/nas/snubhcvc/project/cag_fm/pretrain/datasets/images \
    gram.ckpt=output/a6000/1_pretrain/dinov3_vits16/3_cagcontfm3m/2_stage2_gram_anchor/eval/training_29999/teacher_checkpoint.pth \
    student.resume_from_teacher_chkpt=output/a6000/1_pretrain/dinov3_vits16/3_cagcontfm3m/2_stage2_gram_anchor/eval/training_29999/teacher_checkpoint.pth