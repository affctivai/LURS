#!/bin/bash

set -euo pipefail

usage() {
echo "Usage: $0 <pretrained_model_name> <model_name> <cuda_id> [subject]"
echo "  Example: $0 pretrain_base DELTA 0 unique_sent"
exit 1
}

# 인자: 3개(필수) + 1개(선택)
if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
usage
fi

pretrain_name=$1
model_name=$2
cuda=cuda:"$3"


# ['DELTA', 'wo_diffusion', 'DConv', 'EEGNet', 'wo_pretrained','main_diffusion', 'full_diffusion']
# [pretrain, DConv_pretrain, EEGNet_pretrain, wo_diffusion_pretrain]
# subject ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH']
subject="unique_sent"
batch=32
noise="sqrt"
time_steps=( 1000 1500 500 )

# subjects=( 'ZAB' 'ZDM' 'ZGW' 'ZJM' 'ZJN' 'ZJS' 'ZKB' 'ZKH' 'ZKW' 'ZMG' 'ZPH' )
# subjects=('YDR' 'YFR' 'YFS' 'YAC' 'YDG' 'YHS' 'YMD' 'YLS' 'YRH' 'YRK' 'YSD' 'YRP' 'YSL' 'YTL' 'YAG' 'YAK' 'YIS' 'YMS' )
for time_step in "${time_steps[@]}"; do
    if [ "$model_name" != "w/o_pretrained" ]; then
        pretrained_module="$subject"_"$noise"_"$time_step"_"$pretrain_name"-module
        python3 pretrain.py --model_name "$pretrain_name" \
            --task_name task1_task2_taskNRv2 \
            --num_epoch 100 \
            -lr 0.00002 \
            -b $batch \
            --cuda "$cuda" \
            -setting "$subject" \
            -n "$noise" \
            -ts "$time_step" \
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
        -n "$noise" \
        -ts "$time_step" \
        -s ./save_model/checkpoints/DELTA    

    # 활성화 옵션 메모:
    # -con \
    # -geo\
    # -kl \
done
