#!/bin/bash

PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/train/train.py \
  --nodes 1 \
  --config-file dinov3/configs/train/dinov3_vits16_pretrain.yaml \
  --output-dir 1_stage1_exp_pretraining \
  train.dataset_path=imagenet:root=/mnt/c/Users/heesu/workspace/dinov3_stack/input/imagenet-1k