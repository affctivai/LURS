#!/bin/bash

set -euo pipefail

usage() {
  echo "Usage: $0 <model_name> <cuda_id> [subject]"
  echo "  Example: $0 LURS 0 unique_sent"
  exit 1
}

# 인자: 2개(필수) + 1개(선택)
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

python3 analysis_eeg_contribution.py \
    --checkpoint_path save_model/checkpoints/LURS/best/"$model_name".pt \
    --config_path save_model/config/"$model_name".json \
    -setting "$subject" \
    -cuda cuda:"$cuda"
