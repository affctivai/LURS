#!/bin/bash

set -euo pipefail

usage() {
  echo "Usage: $0 <pretrained_model_name> <model_name> <cuda_id> [-s subject] [-p pretrain_save_name]"
  echo "  Example: $0 pretrain LURS 0 -s unique_sent -p my_pretrain_name"
  echo "  test_input=noise 베이스라인 실험용 스크립트"
  exit 1
}

# 인자: 3개(필수) + subject/pretrain_save_name는 순서 상관없는 옵션 플래그
if [ "$#" -lt 3 ]; then
  usage
fi

pretrain_name=$1
model_name=$2
cuda=cuda:"$3"
shift 3

# ['LURS', 'wo_diffusion', 'DConv', 'EEGNet', 'MLP', 'wo_pretrained','main_diffusion', 'full_diffusion']
# [pretrain, DConv_pretrain, EEGNet_pretrain, MLP_pretrain, wo_diffusion_pretrain]
# subject ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH']

subject="ALL"
pretrain_save_name=""

while getopts "s:p:" opt; do
  case "$opt" in
    s) subject=$OPTARG ;;
    p) pretrain_save_name=$OPTARG ;;
    *) usage ;;
  esac
done

batch=32
noise="linear"
time_step=1000
test_input="eeg"
save_name=$model_name

# pretrain 체크포인트/config 이름. -p 안 주면 기존 규칙대로 자동 생성
if [ -z "$pretrain_save_name" ]; then
    pretrain_save_name="${subject}_${noise}_${time_step}_${pretrain_name}-module"
fi

if [ "$model_name" != "wo_pretrained" ]; then
    pretrained_module=con_"$pretrain_save_name"
    save_name="$pretrained_module"_"$model_name"
    python3 pretrain.py --model_name "$pretrain_name" \
        --task_name task1_task2_taskNRv2 \
        --num_epoch 100 \
        -lr 0.00002 \
        -b $batch \
        --cuda "$cuda" \
        -ts "$time_step" \
        -n "$noise" \
        -s ./save_model/checkpoints/autoencoder \
        --test_input "$test_input" \
        -setting unique_sent \
        -con \
        -subj "$subject" \
        --save_name "$pretrain_save_name"
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
    -s ./save_model/checkpoints/LURS \
    --save_name "$save_name"
