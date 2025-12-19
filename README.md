# EEG-To-Text

## Overview
This project translates electroencephalography (EEG) signals into natural-language sentences.  
It combines the architecture of **NeuSpeech/EEG-To-Text** with the latent-diffusion framework from **CompVis/latent-diffusion**, adding several modifications for joint EEG–text representation learning. ([NeuSpeech/EEG-To-Text - GitHub](https://github.com/NeuSpeech/EEG-To-Text?utm_source=chatgpt.com), [High-Resolution Image Synthesis with Latent Diffusion Models](https://github.com/CompVis/latent-diffusion?utm_source=chatgpt.com))

---

## Prerequisites
| Package  | Tested Version |
|----------|----------------|
| Python   | 3.9.21 |
| PyTorch  | 1.9.0 |
| CUDA     | 12.4 |
| NumPy, SciPy, einops, transformers, transformers, tqdm, evaluate, sacrebleu, rouge_score

```
conda env create -f environment.yml
```
---

## Directory Layout
```
.
├── config/                 # JSON configs for training / evaluation
├── checkpoints/            # saved models (auto-generated)
├── data/                   # pickled datasets (auto-generated)
├── ldm/                    # latent-diffusion modules
├── scripts/
│   └── prepare_dataset.sh  # full preprocessing pipeline
├── *.py                    # training / evaluation entry points
└── README.md
```

---

## Dataset Preparation
 Run:
```bash
./scripts/prepare_dataset.sh
```
The script converts all recordings to pickled tensors and splits them into *train / dev / test*.

---

## Training

### 1. Easy start
```bash
./script/train_decoding.sh <pretrained_model_name> <model_name> <cuda_id> [subject]
```

pretrained_model_name must be included [pretrain, DConv_pretrain, EEGNet_pretrain, wo_diffusion_pretrain]

model_name must be included  ['DELTA', 'wo_diffusion', 'DConv', 'EEGNet', 'wo_pretrained','main_diffusion', 'full_diffusion']


subject is optional
subjects=( 'ZAB' 'ZDM' 'ZGW' 'ZJM' 'ZJN' 'ZJS' 'ZKB' 'ZKH' 'ZKW' 'ZMG' 'ZPH' )
subjects=('YDR' 'YFR' 'YFS' 'YAC' 'YDG' 'YHS' 'YMD' 'YLS' 'YRH' 'YRK' 'YSD' 'YRP' 'YSL' 'YTL' 'YAG' 'YAK' 'YIS' 'YMS' )


### 2. Autoencoder Pre-training
```bash
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
```
*Outputs*  
`checkpoints/autoencoder/best/pretrained_module_*.pt`

### 3. Two-Step Full Model Training
```bash
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
```

---

## Evaluation
```bash
python3 eval_decoding.py \
  --checkpoint_path checkpoints/best/task1_task2_taskNRv2_2steptraining_b32_100_50_2e-05_2e-05.pt \
  --config_path     config/decoding/task1_task2_taskNRv2_2steptraining_b32_100_50_2e-05_2e-05.json \
  --test_input EEG \
  --train_input EEG \
  --cuda cuda:0
```
