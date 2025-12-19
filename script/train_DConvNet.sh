#!/bin/bash

set -euo pipefail

usage() {
  echo "Usage: $0 <cuda_id>"
  echo "  Example: $0 0"
  exit 1
}

# 인자: 1개(필수)
if [ "$#" -lt 1 ]; then
  usage
fi

pretrain_name=DConv_pretrain
model_name=DConv
cuda=cuda:"$1"


# ['DELTA', 'wo_diffusion', 'DConv', 'EEGNet', 'wo_pretrained','main_diffusion', 'full_diffusion']
# [pretrain, DConv_pretrain, EEGNet_pretrain, wo_diffusion_pretrain]
# subject ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH']

if [ -n "${4:-}" ]; then
    subject=$4
else
    subject="unique_sent"
fi

batch=32
noise="linear"
time_step=1000

save_name="WcrosAttention_t-emb_DConvNet"


if [ "$model_name" != "w/o_pretrained" ]; then
    pretrained_module="$subject"_"$save_name"-module
    python3 pretrain.py --model_name "$pretrain_name" \
        --task_name task1_task2_taskNRv2 \
        --num_epoch 100 \
        -lr 0.00002 \
        -b $batch \
        --cuda "$cuda" \
        -setting "$subject" \
        -n "$noise" \
        -ts "$time_step" \
        --save_name "$save_name" \
        -s ./save_model/checkpoints/autoencoder
else
    pretrained_module="Default_setting"
fi

# 활성화 옵션 메모:
# -con \
# -geo\
# -kl \

python3 train.py --model_name "$model_name" \
    --task_name task1_task2_taskNRv2 \
    --num_epoch_step1 100 \
    --num_epoch_step2 50 \
    --config_path save_model/config/"$pretrained_module".json \
    -lr1 0.00002 \
    -lr2 0.00002 \
    -b $batch \
    --cuda "$cuda" \
    --save_name "$save_name" \
    -s ./save_model/checkpoints/DELTA    

# 활성화 옵션 메모:
# -con \
# -geo\
# -kl \


