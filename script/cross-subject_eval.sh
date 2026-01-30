#!/bin/bash

set -euo pipefail

usage() {
echo "Usage: $0 <pretrained_model_name> <model_name> <cuda_id>"
echo "  Example: $0 pretrain_base DELTA 0"
exit 1
}

# 인자: 3개(필수) 
if [ "$#" -lt 3 ]; then
usage
fi

pretrain_name=$1
model_name=$2
cuda=cuda:"$3"


# ['DELTA', 'wo_diffusion', 'DConv', 'EEGNet', 'wo_pretrained','main_diffusion', 'full_diffusion']
# [pretrain, DConv_pretrain, EEGNet_pretrain, wo_diffusion_pretrain]
# subject ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH']

batch=32
noise="sqrt"
time_step=1000
subject=cross
# subjects=( 'ZAB' 'ZDM' 'ZGW' 'ZJM' 'ZJN' 'ZJS' 'ZKB' 'ZKH' 'ZKW' 'ZMG' 'ZPH' )
# subjects=('YDR' 'YFR' 'YFS' 'YAC' 'YDG' 'YHS' 'YMD' 'YLS' 'YRH' 'YRK' 'YSD' 'YRP' 'YSL' 'YTL' 'YAG' 'YAK' 'YIS' 'YMS' )



python3 pretrain_cross.py --model_name "$pretrain_name" \
    --task_name taskNRv2 \
    --num_epoch 100 \
    -lr 0.00002 \
    -b $batch \
    --cuda "$cuda" \
    -setting "$subject" \
    -n "$noise" \
    -ts "$time_step" \
    -s ./save_model/checkpoints/autoencoder

# 활성화 옵션 메모:
# -con \
# -geo\
# -kl \