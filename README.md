# EEG-To-Text (LURS)

## Overview
This project decodes natural-language sentences from EEG signals recorded during reading (ZuCo dataset). It builds on **NeuSpeech/EEG-To-Text** and the latent-diffusion framework from **CompVis/latent-diffusion**, adding a stochastic autoencoder + diffusion denoising pipeline with an optional **LURS** latent-refinement step (single-step deterministic refinement of the sampled latent, see `3618_Latent_Level_Uncertainty_.pdf`).

Pipeline: `EEG -> stochastic encoder (pretrain) -> diffusion denoising -> (optional LURS refinement) -> fc1 -> BART decoder -> text`

---

## Prerequisites
| Package | Tested Version |
|---|---|
| Python | 3.9.21 |
| PyTorch | 1.9.0 |
| CUDA toolkit | 11.1 |
| transformers | 4.6.1 |

Other deps: numpy, scipy, h5py, tqdm, matplotlib, nltk, einops, omegaconf, evaluate, sacrebleu, rouge, jiwer, bert-score, tensorboard.

```bash
conda env create -f environment.yml
conda activate EEGToText
```

---

## Directory Layout
```
.
├── data/                        # ZuCo -> pickle preprocessing + PyTorch Datasets
│   ├── prepare_zuco_dataset/    # .mat -> .pickle conversion scripts
│   ├── data.py                  # EEG (band-power) dataset
│   └── data_for_rawEEG*.py       # raw-EEG dataset variants
├── model/
│   ├── pretrain_model.py        # stochastic autoencoder (+ EEGNet/DConv/MLP pretrain variants)
│   ├── decoding_model.py         # full model: pretrained encoder + diffusion + BART decoder
│   └── wrappers.py
├── module/
│   ├── ldm/                     # latent-diffusion modules (autoencoder, denoising U-Net, attention)
│   └── compare_model/           # baseline encoders: EEGNet, DeepConvNet, MLP
├── utils/
│   ├── config.py                 # argparse configs: pretrain_autoencoder / train_diffusion / eval_decoding
│   └── utils.py
├── pretrain.py                  # Stage 1: autoencoder pretraining
├── train.py                     # Stage 2: full model training (two-step)
├── evaluation.py                 # BLEU/ROUGE/BERTScore/... evaluation on a checkpoint
├── metrics.py                    # BERTScore, BLEURT, NLI entailment, entity-preservation metrics
├── analysis_eeg_contribution.py  # per-token delta = logP(y|EEG) - logP(y|NoEEG)
├── analysis_latent_refinement.py # LURS before/after refinement analysis (Sec. 3.3/4.2, Appendix B)
├── get_results.py                # parse result .txt logs into summary tables
├── script/                       # shell wrappers for the commands below
├── save_model/                   # checkpoints + configs (auto-generated, gitignored)
├── results/ , score_results/      # evaluation outputs
└── logs/                          # training/eval logs (gitignored)
```

---

## Dataset Preparation
Convert ZuCo `.mat` recordings to pickled tensors:
```bash
./script/prepare_dataset.sh EEG      # band-power features
./script/prepare_dataset.sh rawEEG   # raw EEG timeseries
```

---

## Training

Training is two-stage: (1) pretrain a stochastic EEG autoencoder, (2) train the full diffusion + BART decoding model on top of it. `script/train_decoding.sh` runs both stages back-to-back.

### 1. Autoencoder pretraining (standalone)
```bash
python3 pretrain.py \
    --model_name LURS \
    --task_name task1_task2_taskNRv2 \
    --num_epoch 100 \
    -lr 0.00002 \
    -b 32 \
    --cuda cuda:0 \
    -n linear -ts 1000 \
    -setting unique_sent \
    -subj ALL \
    -s ./save_model/checkpoints/autoencoder \
    --save_name my_pretrain_run
```
Key flags: `-con` (contrastive loss), `-kl` (KL loss), `-geo` (geometric loss), `-subj` (leave-one-subject-out; default `ALL`), `-n` (noise scheduler: linear/sqrt/cosine), `-ts` (diffusion time steps).
Outputs a checkpoint under `save_model/checkpoints/autoencoder/best/` and a matching config JSON under `save_model/config/`.

### 2. Full model training (two-step)
```bash
python3 train.py \
    --model_name LURS \
    --config_path save_model/config/<pretrain_save_name>.json \
    --num_epoch_step1 100 --num_epoch_step2 50 \
    -lr1 0.00002 -lr2 0.00002 \
    -b 32 \
    --cuda cuda:0 \
    -s ./save_model/checkpoints/LURS \
    --save_name my_run
```
`--model_name` selects the encoder/decoder variant: `LURS`, `wo_diffusion`, `DConv`, `EEGNet`, `MLP`, `wo_pretrained`, `main_diffusion`, `full_diffusion`.

### Convenience scripts (pretrain + train in one call)
```bash
./script/train_decoding.sh pretrain LURS 0 [-s subject] [-p pretrain_save_name]
./script/train_contrastive.sh pretrain LURS 0    # adds -con to pretrain
./script/train_kl.sh pretrain LURS 0             # adds -kl to pretrain
./script/train_EEGNet.sh 0                        # EEGNet baseline
./script/train_DConvNet.sh 0                      # DeepConvNet baseline
```
`-s` sets the subject split, `-p` overrides the auto-generated pretrain checkpoint name.

---

## Evaluation
```bash
python3 evaluation.py \
    --checkpoint_path save_model/checkpoints/LURS/best/<model_name>.pt \
    --config_path save_model/config/<model_name>.json \
    -setting unique_sent \
    -cuda cuda:0 \
    -b 32
```
Or:
```bash
./script/eval_decoding_1.sh <model_name> <cuda_id> [subject_setting]
./script/cross-subject_eval.sh <pretrained_model_name> <model_name> <cuda_id>
```
Reports BLEU-1..4, ROUGE, and (via `metrics.py`) BERTScore/BLEURT/NLI-entailment/entity-preservation. Results are written to `results/` and `score_results/`; `get_results.py` aggregates the `.txt` logs into summary tables.

---

## Analysis Scripts
Both take the same `--checkpoint_path`/`--config_path` args as `evaluation.py`:

```bash
./script/eeg_contribution_analysis.sh <model_name> <cuda_id> [subject_setting]
```
Per-token `delta_i = logP(y_i | y_<i, EEG) - logP(y_i | y_<i, NoEEG)`, isolating what the EEG input contributes to each predicted token (EEG replaced with gaussian noise for the counterfactual pass).

```bash
./script/latent_refinement_analysis.sh <model_name> <cuda_id> [subject_setting]
```
Before/after comparison of the LURS latent refinement step (Z0 vs Z*): memory-perturbation sensitivity, pairwise distance, and per-dimension variance among same-target-sentence latents — empirically checks the contraction property behind the refinement.
