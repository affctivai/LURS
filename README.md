# DELTA: Latent Diffusion for Open-Vocabulary EEG-to-Text Decoding

## Overview
This project translates electroencephalography (EEG) signals into natural-language sentences using a latent diffusion framework. It builds upon the architecture of NeuSpeech/EEG-To-Text and integrates core components of the CompVis/latent-diffusion model, while introducing several key innovations to enhance EEG–text representation learning. ([NeuSpeech/EEG-To-Text - GitHub](https://github.com/NeuSpeech/EEG-To-Text?utm_source=chatgpt.com), [High-Resolution Image Synthesis with Latent Diffusion Models](https://github.com/CompVis/latent-diffusion?utm_source=chatgpt.com))

The proposed model, DELTA (Diffusion-based EEG-to-Latent-Text Architecture), decouples EEG signal encoding from text generation by operating in a probabilistic latent space. It incorporates:

* A variational EEG encoder to model uncertainty and variability in brain signals.
* A denoising U-Net trained with latent-space diffusion, enabling semantically guided refinement of EEG representations.
* A pretrained BART decoder for natural and expressive language generation from refined latents.

Unlike previous deterministic models, DELTA supports open-vocabulary decoding and produces more diverse, coherent, and lexically rich outputs. It achieves state-of-the-art performance on the ZuCo dataset, particularly under non-teacher-forcing (Non-TF) conditions that better reflect real-world EEG-to-text applications.

---

## Prerequisites
| Package  | Tested Version |
|----------|----------------|
| Python   | 3.9.21 |
| PyTorch  | 1.9.0 |
| CUDA     | 12.4 |
| NumPy, SciPy, einops, transformers, transformers, tqdm, evaluate, sacrebleu, rouge_score

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

### 
```bash
conda env create -f delta_env.yaml
```

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
    --num_epoch 50 \
    --train_input EEG \
    -lr 0.00002 \
    -b 32 \
    --cuda cuda:0 \
    -s ./checkpoints/autoencoder
```
*Outputs*  
`checkpoints/autoencoder/best/pretrained_module_*.pt`

### 2. Two-Step Full Model Training
```bash
python3 train_diffusion.py --model_name BrainTranslator \
    --task_name task1_task2_taskNRv2 \
    --two_step \
    --pretrained \
    --not_load_step1_checkpoint \
    --num_epoch_step1 50 \
    --num_epoch_step2 20 \
    --train_input EEG \
    -lr1 0.00002 \
    -lr2 0.00002 \
    -b 32 \
    --cuda cuda:0 \
    -s ./checkpoints \
    --eeg_encoder checkpoints/encoder/best/pretrained_module_task1_task2_taskNRv2_32_100_2e-05.pt
```

---

## Evaluation
```bash
python3 eval_decoding.py \
    --checkpoint_path checkpoints/best/ldm_task1_task2_taskNRv2_2steptraining_b32_50_20_2e-05_2e-05.pt \
    --config_path config/decoding/ldm_task1_task2_taskNRv2_2steptraining_b32_50_20_2e-05_2e-05.json \
    --test_input EEG \
    --train_input EEG \
    -cuda cuda:0
```
