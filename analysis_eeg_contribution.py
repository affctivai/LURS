"""Per-token EEG-conditioning contribution analysis.

For each token position i in the target sentence, computes:
    delta_i = logP(y_i | y_<i, EEG) - logP(y_i | y_<i, NoEEG)

Both terms are teacher-forced (same target_ids as labels, so y_<i is identical
in both passes) — the only thing that changes is the EEG signal fed into the
model. "EEG" is the real input_embeddings_batch; "NoEEG" replaces it with
random gaussian noise before model.layer.encode(), so the exact same encode ->
denoising -> fc1 -> BART pipeline runs on an uninformative EEG signal. delta_i
therefore isolates what the EEG input contributes to predicting each token,
holding everything else fixed.

Run with the same --checkpoint/--config args as evaluation.py.
"""
import os
import json

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration
from nltk.translate.bleu_score import corpus_bleu
from scipy.stats import pearsonr, spearmanr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.data import ZuCoNpyDataset
from model import *
from utils import *
from module.ldm.autoencoder import AutoencoderKL
from module.ldm.openaimodel import UNetModel
from module.compare_model.EEGNet import EEGNet
from module.compare_model.DeepConvNet import DeepConvNet
from module.compare_model.MLP import MLP


@torch.no_grad()
def compute_delta_logprob(model, dataloader, tokenizer, device):
    """Returns (records, all_deltas, all_meta):
      records    - one dict per test sentence: target string, per-token tokens/
                   delta, and sentence-level mean/sum delta.
      all_deltas - flat list of every valid token's delta_i (dataset-level stats).
      all_meta   - parallel list to all_deltas: {"target", "token", "position"},
                   used to trace top/bottom delta_i back to their source token.
    """
    model.eval()
    records = []
    all_deltas = []
    all_meta = []

    for batch in tqdm(dataloader):
        (input_embeddings, seq_len, input_masks, input_mask_invert,
         target_ids, target_mask, sentiment_label, sent_level_EEG) = batch

        input_embeddings_batch = input_embeddings.to(device).float()
        input_masks_batch = input_masks.to(device)
        target_ids_batch = target_ids.to(device)
        context = target_ids_batch.clone()

        labels = target_ids_batch.clone()
        labels[labels == tokenizer.pad_token_id] = -100

        # --- EEG-conditioned pass: logP(y_i | y_<i, EEG) ---
        out_eeg = model.evaluate(input_embeddings_batch, input_masks_batch, labels, context)
        logp_eeg = F.log_softmax(out_eeg.logits, dim=-1)

        # --- NoEEG pass: logP(y_i | y_<i, NoEEG), same pipeline with EEG replaced by random noise ---
        noise_eeg_batch = torch.randn_like(input_embeddings_batch)
        out_noeeg = model.evaluate(noise_eeg_batch, input_masks_batch, labels, context)
        logp_noeeg = F.log_softmax(out_noeeg.logits, dim=-1)

        valid_mask = labels != -100                       # (B, T) — pad positions excluded
        valid_mask &= labels != tokenizer.eos_token_id     # exclude </s>
        valid_mask[:, 0] = False                           # exclude leading <s> (bos)
        gather_idx = labels.clone()
        gather_idx[~valid_mask] = 0                        # dummy index, masked out below

        token_logp_eeg = logp_eeg.gather(-1, gather_idx.unsqueeze(-1)).squeeze(-1)      # (B, T)
        token_logp_noeeg = logp_noeeg.gather(-1, gather_idx.unsqueeze(-1)).squeeze(-1)   # (B, T)
        delta = token_logp_eeg - token_logp_noeeg                                        # (B, T)

        # Teacher-forced argmax prediction from the EEG-conditioned pass, same pipeline
        # as evaluation.py's "predicted string with tf" (logits_previous.argmax) — used
        # to pair each sentence with a sentence-level BLEU score for the Δ̄ vs BLEU correlation.
        pred_ids_batch = out_eeg.logits.argmax(dim=-1)  # (B, T)

        for b in range(delta.shape[0]):
            mask_b = valid_mask[b]
            delta_b = delta[b][mask_b].cpu().tolist()
            token_ids_b = labels[b][mask_b].cpu().tolist()
            tokens_b = tokenizer.convert_ids_to_tokens(token_ids_b)
            target_b = tokenizer.decode(target_ids_batch[b], skip_special_tokens=True)

            pred_ids_b = pred_ids_batch[b].cpu().tolist()
            truncated_pred_b = []
            for t in pred_ids_b:
                if t == tokenizer.eos_token_id:
                    break
                truncated_pred_b.append(t)
            pred_tokens_b = tokenizer.convert_ids_to_tokens(truncated_pred_b, skip_special_tokens=True)
            # same nltk corpus_bleu call evaluation.py uses, applied to one sentence at a time
            bleu_b = corpus_bleu([[tokens_b]], [pred_tokens_b], weights=(0.25, 0.25, 0.25, 0.25))

            records.append({
                "target": target_b,
                "tokens": tokens_b,
                "delta": delta_b,
                "mean_delta": sum(delta_b) / len(delta_b) if delta_b else 0.0,
                "sum_delta": sum(delta_b),
                "num_tokens": len(delta_b),
                "bleu": bleu_b,
            })
            all_deltas.extend(delta_b)
            all_meta.extend(
                {"target": target_b, "token": tok, "position": pos}
                for pos, tok in enumerate(tokens_b)
            )

    return records, all_deltas, all_meta


def top_bottom_tokens(all_deltas, all_meta, k=20):
    """Top-k tokens where EEG helps most (highest delta_i) and hurts most
    (lowest / most negative delta_i), traced back to their source token."""
    delta_tensor = torch.tensor(all_deltas)
    k = min(k, delta_tensor.numel())
    if k == 0:
        return {"top_eeg_helps": [], "top_eeg_hurts": []}

    top = torch.topk(delta_tensor, k)
    bottom = torch.topk(-delta_tensor, k)

    def _pack(indices, values):
        return [{**all_meta[idx], "delta": val} for idx, val in zip(indices.tolist(), values.tolist())]

    return {
        "top_eeg_helps": _pack(top.indices, top.values),
        "top_eeg_hurts": _pack(bottom.indices, -bottom.values),
    }


def summarize(records, all_deltas):
    micro_mean = sum(all_deltas) / len(all_deltas) if all_deltas else 0.0
    macro_mean = (sum(r["mean_delta"] for r in records) / len(records)) if records else 0.0
    positive_frac = (sum(1 for d in all_deltas if d > 0) / len(all_deltas)) if all_deltas else 0.0
    return {
        "num_sentences": len(records),
        "num_tokens": len(all_deltas),
        "micro_mean_delta": micro_mean,   # token-weighted average
        "macro_mean_delta": macro_mean,   # sentence-weighted average (equal weight per sentence)
        "positive_token_fraction": positive_frac,  # fraction of tokens where EEG helped (delta_i > 0)
    }


def correlate_delta_bleu(records):
    """Pearson/Spearman between each sentence's mean delta (Δ̄) and its sentence-level
    BLEU-4 (same nltk corpus_bleu used in evaluation.py, one sentence per call)."""
    means = [r["mean_delta"] for r in records]
    bleus = [r["bleu"] for r in records]
    pearson_r, pearson_p = pearsonr(means, bleus)
    spearman_r, spearman_p = spearmanr(means, bleus)
    return {
        "n": len(records),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
    }


def plot_histogram(all_deltas, output_path, bins=50):
    plt.figure(figsize=(8, 5))
    plt.hist(all_deltas, bins=bins, color='steelblue', edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', linewidth=1, label='delta_i = 0')
    plt.xlabel(r'$\delta_i = \log P(y_i \mid y_{<i}, EEG) - \log P(y_i \mid y_{<i}, NoEEG)$')
    plt.ylabel('token count')
    plt.title('Per-token EEG conditioning contribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


if __name__ == '__main__':
    batch_size = 32
    args = get_config('eval_decoding')

    with open(args['config_path'], 'r') as f:
        configs = json.load(f)
    training_config = configs['pretrained_args']
    module_state_dict_config = configs['module_state_dict']
    save_name = args['config_path'].split('/')[-1].split('.')[0] if args['save_name'] is None else args['save_name']

    is_con = training_config['is_con']
    is_geo = training_config['is_geo']
    is_kl = training_config['is_kl']

    eeg_type_choice = training_config['eeg_type']
    bands_choice = training_config['eeg_bands']
    subject_choice = training_config['subjects']
    test_input = training_config['test_input']
    task_name = training_config['task_name']
    model_name = configs['args']['model_name']

    if not os.path.exists('./results'):
        os.makedirs('./results')
    output_path = f'./results/eeg_contribution_{save_name}.json'

    set_seed(args['seed'])

    device = torch.device(args['cuda'] if torch.cuda.is_available() else "cpu")
    print(f'[INFO]using device {device}')

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    dataset_setting = args['dataset_setting']

    npy_root = "/mnt/Members/speech/jd_data/zuco_decoding_npy/"
    task_npy_dirs = []
    if 'task1' in task_name:
        task_npy_dirs.append(os.path.join(npy_root, "task1-SR"))
    if 'task2' in task_name:
        task_npy_dirs.append(os.path.join(npy_root, "task2-NR"))
    if 'task3' in task_name:
        task_npy_dirs.append(os.path.join(npy_root, "task3-TSR"))
    if 'taskNRv2' in task_name:
        task_npy_dirs.append(os.path.join(npy_root, "task2-NR-2.0"))

    test_set = ZuCoNpyDataset(task_npy_dirs, phase='test', tokenizer=tokenizer, subject=subject_choice,
                               eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting,
                               test_input=test_input)
    print('[INFO]test_set size: ', len(test_set))
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    pretrained_LM = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

    denoising_module_statte_dict = module_state_dict_config["denoising_module"]
    autoencoder_state_dict = module_state_dict_config["autoencoder"]

    if model_name == 'EEGNet':
        denoising_module = EEGNet(**denoising_module_statte_dict).to(device)
    elif model_name == 'DConv':
        denoising_module = DeepConvNet(**denoising_module_statte_dict).to(device)
    elif model_name == 'MLP':
        denoising_module = MLP(**denoising_module_statte_dict).to(device)
    else:
        denoising_module = UNetModel(**denoising_module_statte_dict).to(device)

    autoencoder = AutoencoderKL(**autoencoder_state_dict).to(device)

    is_diffusion = model_name in ('main_diffusion', 'full_diffusion')
    is_compare_model = model_name in ('EEGNet', 'DConv', 'MLP')

    pretrained_model_name = training_config['model_name']
    pretrain_opts = BuildOptions(model_name=pretrained_model_name, is_con=is_con, is_geo=is_geo, is_kl=is_kl,
                                 is_diffusion=is_diffusion, is_compare_model=is_compare_model,
                                 autoencoder=autoencoder, denoising_module=denoising_module,
                                 pretrained_LM=pretrained_LM, device=device)
    pretrained_module = make_model(pretrain_opts)

    opts = BuildOptions(model_name=model_name, is_con=is_con, is_geo=is_geo, is_kl=is_kl, device=device,
                         pretrained_module=pretrained_module, is_diffusion=is_diffusion,
                         is_compare_model=is_compare_model)
    model = make_model(opts)

    state_dict = torch.load(args['checkpoint_path'], map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    records, all_deltas, all_meta = compute_delta_logprob(model, test_dataloader, tokenizer, device)
    summary = summarize(records, all_deltas)
    extremes = top_bottom_tokens(all_deltas, all_meta, k=20)
    delta_bleu_corr = correlate_delta_bleu(records)

    print(json.dumps(summary, indent=2))
    print('[INFO]Δ̄ vs sentence BLEU-4 correlation:')
    print(json.dumps(delta_bleu_corr, indent=2))
    print('[INFO]top 20 tokens where EEG helps most:')
    for item in extremes["top_eeg_helps"]:
        print(f'  {item["delta"]:+.4f}  "{item["token"]}"  (pos {item["position"]}, target: {item["target"]})')
    print('[INFO]top 20 tokens where EEG hurts most:')
    for item in extremes["top_eeg_hurts"]:
        print(f'  {item["delta"]:+.4f}  "{item["token"]}"  (pos {item["position"]}, target: {item["target"]})')

    with open(output_path, 'w') as f:
        json.dump({"summary": summary, "extremes": extremes, "delta_bleu_correlation": delta_bleu_corr,
                    "records": records}, f, indent=2)
    print(f'[INFO]EEG contribution analysis saved in {output_path}')

    hist_path = f'./results/eeg_contribution_{save_name}_hist.png'
    plot_histogram(all_deltas, hist_path)
    print(f'[INFO]delta_i histogram saved in {hist_path}')
