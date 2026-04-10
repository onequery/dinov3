#!/bin/bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# =====================================================
# A6000 settings (Single GPU)
# =====================================================
# --------------------------------------------
# 1. High-Res Adaptation + Content-State - CAS-Contrast-FM-3M
# --------------------------------------------
# export CUDA_VISIBLE_DEVICES=0
#
# PYTHONPATH=${PWD} python dinov3/train/train.py \
#     --config-file dinov3/configs/train/dinov3_vits16_high_res_adapt_content_state_a6000.yaml \
#     --output-dir output/a6000/1_pretrain/dinov3_vits16/3_cagcontfm3m/3_stage3_high_res_adapt_content_state \
#     train.dataset_path=CAGContrastFMContinuityV1:split=TRAIN:root=/mnt/nas/snubhcvc/project/cag_fm/pretrain/datasets/images \
#     gram.ckpt=output/a6000/1_pretrain/dinov3_vits16/3_cagcontfm3m/2_stage2_gram_anchor/eval/training_29999/teacher_checkpoint.pth \
#     student.resume_from_teacher_chkpt=output/a6000/1_pretrain/dinov3_vits16/3_cagcontfm3m/2_stage2_gram_anchor/eval/training_29999/teacher_checkpoint.pth

# =====================================================
# A6000 settings (Two GPU, Same Global Batch)
# =====================================================
# Preserve the global batch at 16 images/step:
#   - single GPU: batch_size_per_gpu = 16
#   - two GPU:    batch_size_per_gpu = 8
#
# Reduce dataloader workers for stability on this host. This is a
# systems/runtime knob rather than an experimental objective change, and can be
# overridden at launch time if needed.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-8}"

PYTHONPATH=${PWD} torchrun --standalone --nproc_per_node=2 -m dinov3.train.train \
    --config-file dinov3/configs/train/dinov3_vits16_high_res_adapt_content_state_a6000.yaml \
    --output-dir output/a6000/1_pretrain/dinov3_vits16/3_cagcontfm3m/3_stage3_high_res_adapt_content_state \
    train.dataset_path=CAGContrastFMContinuityV1:split=TRAIN:root=/mnt/nas/snubhcvc/project/cag_fm/pretrain/datasets/images \
    gram.ckpt=output/a6000/1_pretrain/dinov3_vits16/3_cagcontfm3m/2_stage2_gram_anchor/eval/training_29999/teacher_checkpoint.pth \
    student.resume_from_teacher_chkpt=output/a6000/1_pretrain/dinov3_vits16/3_cagcontfm3m/2_stage2_gram_anchor/eval/training_29999/teacher_checkpoint.pth \
    train.batch_size_per_gpu=8 \
    train.num_workers="${TRAIN_NUM_WORKERS}"

