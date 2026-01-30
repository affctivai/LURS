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

### 1. Autoencoder Pre-training
```bash
python3 pretrain.py \
    --task_name task1_task2_taskNRv2 \
    --num_epoch 100 \
    --train_input EEG \
    -lr 0.00002 \
    -b 32 \
    --cuda cuda:0 \
    -s ./checkpoints/encoder
```
*Outputs*  
`checkpoints/autoencoder/best/pretrained_module_*.pt`

### 2. Two-Step Full Model Training
```bash
python3 train_diffusion.py --model_name BrainTranslator \
    --task_name task1_task2_taskNRv2 \
    --two_step \
    --num_epoch_step1 100 \
    --num_epoch_step2 50 \
    --train_input EEG \
    -lr1 0.00002 \
    -lr2 0.00002 \
    -b 32 \
    --cuda cuda:1 \
    -s ./checkpoints \
    --text_encoder _ \
    --eeg_encoder checkpoints/encoder/best/pretrained_module_task1_task2_taskNRv2_32_100_2e-05.pt \
    --save_name [savae name]
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
