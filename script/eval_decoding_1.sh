#!/bin/bash

set -euo pipefail

usage() {
  echo "Usage: $0 <pretrained_model_name> <model_name> <cuda_id> [subject]"
  echo "  Example: $0 pretrain_base DELTA 0 unique_sent"
  exit 1
}

# 인자: 3개(필수) + 1개(선택)
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  usage
fi

model_name=$1
cuda=$2

if [ -n "${3:-}" ]; then
    subject=$3
else
    subject="unique_sent"
fi

python3 evaluation.py \
    --checkpoint_path save_model/checkpoints/DELTA/best/"$model_name".pt \
    --config_path save_model/config/"$model_name".json \
    --test_input word-level \
    -setting "$subject" \
    -cuda cuda:"$cuda"

