#!/bin/bash

set -euo pipefail

usage() {
  echo "Usage: $0 <pretrained_model_name> <model_name> <cuda_id> [subject]"
  echo "  Example: $0 pretrain DELTA 0 unique_sent"
  exit 1
}

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  usage
fi

pretrain_name=$1
model_name=$2
cuda=cuda:"$3"

# subject ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH']

if [ -n "${4:-}" ]; then
    subject=$4
else
    subject="ALL"
fi

batch=32
noise="linear"
time_step=1000
test_input="eeg"

if [ "$model_name" != "wo_pretrained" ]; then
    pretrained_module="con_${subject}_${noise}_${time_step}_${pretrain_name}-module"
    python3 pretrain.py --model_name "$pretrain_name" \
        --task_name task1_task2_taskNRv2 \
        --num_epoch 100 \
        -lr 0.00002 \
        -b $batch \
        --cuda "$cuda" \
        -subj "$subject" \
        -n "$noise" \
        -ts "$time_step" \
        -con \
        -s ./save_model/checkpoints/autoencoder \
        --test_input "$test_input"
else
    pretrained_module="Default_setting"
fi

python3 train.py --model_name "$model_name" \
    --num_epoch_step1 100 \
    --num_epoch_step2 50 \
    --config_path save_model/config/"$pretrained_module".json \
    -lr1 0.00002 \
    -lr2 0.00002 \
    -b $batch \
    --cuda "$cuda" \
    -s ./save_model/checkpoints/DELTA
