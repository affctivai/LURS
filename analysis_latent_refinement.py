"""Before/after latent-refinement analysis (LURS, Sec. 3.3 / 4.2, Appendix B).

Computes, for the test set of a trained checkpoint:
  - Z0  : initial stochastic latent, Z0 ~ q_phi(Z | x)              (model.layer.encode)
  - Z*  : LURS single-step deterministic refinement of Z0            (Eq. 11-12)

Samples are grouped by target_ids first (i.e. same target sentence — typically
multiple subjects/trials reading the identical sentence), and every stat below is
computed *within* each group, then averaged across groups (equal weight per
sentence). Pooling all samples regardless of target sentence (the previous
approach) mixes "different content" variance with "same content, different
EEG/subject" variance and over-weights sentences with more recorded trials —
grouping isolates the latter, which is what the contraction property is actually
about.

Reports three before/after comparisons:
  1. Memory perturbation  : sensitivity of the latent to a perturbed input/resample,
                            i.e. ||Z(x) - Z(x')||_F, before vs after refinement.
                            Empirically checks the contraction property used in
                            Corollary B.3 (refinement should shrink this).
  2. Pairwise distance    : pairwise L2 distance among same-target-sentence latents,
                            before vs after refinement.
  3. Compare variance     : per-dimension and total variance among same-target-sentence
                            latents, before vs after refinement.

Groups with fewer than MIN_GROUP_SIZE samples (default 2) are skipped since
variance/pairwise-distance need at least 2 samples to be defined.

Run with the same --checkpoint/--config args as evaluation.py.
"""
import os
import json

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration

from data.data import ZuCoNpyDataset
from model import *
from utils import *
from module.ldm.autoencoder import AutoencoderKL
from module.ldm.openaimodel import UNetModel
from module.ldm.util import extract_into_tensor
from module.compare_model.EEGNet import EEGNet
from module.compare_model.DeepConvNet import DeepConvNet
from module.compare_model.MLP import MLP

MIN_GROUP_SIZE = 2  # groups smaller than this can't define variance/pairwise distance


def refine_latent(delta_model, z0):
    """LURS single-step deterministic refinement Z0 -> Z* (paper Eq. 11-12): fixed
    tau_ref=1, null condition C=0. Mirrors DELTA.generate()/evaluate()."""
    schedule = delta_model._schedule
    device = z0.device
    tau_ref = torch.ones((z0.shape[0],), device=device).long()
    sqrt_ac = extract_into_tensor(schedule['sqrt_alphas_cumprod'], tau_ref, z0.shape)
    sqrt_1mac = extract_into_tensor(schedule['sqrt_one_minus_alphas_cumprod'], tau_ref, z0.shape)
    z_tilde = sqrt_ac * z0
    # predicted_noise = delta_model.layer.denoising(z_tilde, tau_ref, None)
    null_condition = torch.zeros((z0.shape[0], z0.shape[1], 512), device=device, dtype=z_tilde.dtype)
    predicted_noise = delta_model.layer.denoising(z_tilde, tau_ref, null_condition)   # ε̂_ref = ε_θ(Z̃, τref, 0)
    z_star = (z_tilde - sqrt_1mac * predicted_noise) / sqrt_ac

    return z_star


@torch.no_grad()
def collect_latents(model, dataloader, device, max_batches=None):
    """For each test sample, gather:
      - Z0, Z*                      : latent before/after refinement
      - ||Z0 - Z0'||_F, ||Z*-Z*'||_F: perturbation sensitivity before/after refinement,
                                      where the perturbed pair is a fresh posterior draw
                                      (Z0' ~ q_phi(Z|x), same x).
      - target_key                  : hashable target_ids sequence, used to group samples
                                      that share the same target sentence.
    """
    z0_all, zstar_all = [], []
    delta0_all, deltastar_all = [], []
    target_keys = []

    for i, batch in tqdm(enumerate(dataloader)):
        if max_batches is not None and i >= max_batches:
            break
        input_embeddings, target_ids_batch = batch[0], batch[4]
        x = input_embeddings.to(device).float()

        z0, _ = model.layer.encode(x)
        zstar = refine_latent(model, z0)
        # perturbation = fresh posterior resample (captures q_phi(Z|x) stochasticity)
        # --- posterior 재샘플링: 같은 x, encode를 다시 호출해 다른 Z0 샘플 ---
        z0_pert, _ = model.layer.encode(x)          # 같은 입력, posterior.sample()이 새 ε 사용
        zstar_pert = refine_latent(model, z0_pert)

        z0_all.append(z0.reshape(z0.shape[0], -1).cpu())
        zstar_all.append(zstar.reshape(zstar.shape[0], -1).cpu())

        delta0_all.append((z0 - z0_pert).reshape(z0.shape[0], -1).norm(dim=1).cpu())
        deltastar_all.append((zstar - zstar_pert).reshape(zstar.shape[0], -1).norm(dim=1).cpu())

        target_keys.extend(tuple(t.tolist()) for t in target_ids_batch)

    return (
        torch.cat(z0_all, dim=0),
        torch.cat(zstar_all, dim=0),
        torch.cat(delta0_all, dim=0),
        torch.cat(deltastar_all, dim=0),
        target_keys,
    )


def group_by_target(target_keys):
    """Map each distinct target_ids sequence (= target sentence) to the sample
    indices that share it (e.g. different subjects/trials reading that sentence)."""
    groups = {}
    for idx, key in enumerate(target_keys):
        groups.setdefault(key, []).append(idx)
    return groups


def _grouped(stat_fn, latents, groups, min_group_size=2):
    """Apply `stat_fn` within each same-target group, then average the resulting
    stats across groups (equal weight per sentence, regardless of group size)."""
    per_group = [stat_fn(latents[idx]) for idx in groups.values() if len(idx) >= min_group_size]
    if not per_group:
        return None
    averaged = {key: sum(g[key] for g in per_group) / len(per_group) for key in per_group[0]}
    averaged["num_groups"] = len(per_group)
    return averaged


def pairwise_distance_stats(latents, max_samples=500):
    """Mean/std/min/max of pairwise Euclidean distances across (a subsample of) the batch."""
    if latents.shape[0] > max_samples:
        idx = torch.randperm(latents.shape[0])[:max_samples]
        latents = latents[idx]
    dists = torch.cdist(latents, latents, p=2)
    n = dists.shape[0]
    off_diag = dists[~torch.eye(n, dtype=torch.bool)]
    return {
        "mean": off_diag.mean().item(),
        "std": off_diag.std().item(),
        "min": off_diag.min().item(),
        "max": off_diag.max().item(),
    }


def variance_stats(latents):
    """Per-dimension variance (averaged, summed) and variance of the per-sample norm."""
    per_dim_var = latents.var(dim=0, unbiased=True)
    return {
        "mean_per_dim_variance": per_dim_var.mean().item(),
        "total_variance": per_dim_var.sum().item(),
        "norm_std": latents.norm(dim=1).std().item(),
    }


def perturbation_stats(deltas):
    """Summary of per-sample ||Z(x) - Z(perturbed x)||_F sensitivity."""
    return {
        "mean": deltas.mean().item(),
        "std": deltas.std().item(),
        "max": deltas.max().item(),
    }


if __name__ == '__main__':
    batch_size = 8
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
    output_path = f'./results/latent_refinement_{save_name}.json'

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

    z0, zstar, delta0, deltastar, target_keys = collect_latents(model, test_dataloader, device)
    groups = group_by_target(target_keys)
    group_sizes = [len(idx) for idx in groups.values()]
    usable_groups = sum(1 for n in group_sizes if n >= MIN_GROUP_SIZE)
    print(f'[INFO]{len(groups)} distinct target sentences, '
          f'{usable_groups} with >= {MIN_GROUP_SIZE} samples (usable for grouped stats)')

    results = {
        "group_info": {
            "num_distinct_targets": len(groups),
            "num_usable_groups": usable_groups,
            "min_group_size": MIN_GROUP_SIZE,
        },
        "pairwise_distance": {
            "before_refinement": _grouped(pairwise_distance_stats, z0, groups, MIN_GROUP_SIZE),
            "after_refinement": _grouped(pairwise_distance_stats, zstar, groups, MIN_GROUP_SIZE),
        },
        "variance": {
            "before_refinement": _grouped(variance_stats, z0, groups, MIN_GROUP_SIZE),
            "after_refinement": _grouped(variance_stats, zstar, groups, MIN_GROUP_SIZE),
        },
        "memory_perturbation": {
            "before_refinement": _grouped(perturbation_stats, delta0, groups, MIN_GROUP_SIZE),
            "after_refinement": _grouped(perturbation_stats, deltastar, groups, MIN_GROUP_SIZE),
        },
    }
    if results["memory_perturbation"]["before_refinement"] and results["memory_perturbation"]["before_refinement"]["mean"]:
        results["memory_perturbation"]["contraction_ratio"] = (
            results["memory_perturbation"]["after_refinement"]["mean"]
            / results["memory_perturbation"]["before_refinement"]["mean"]
        )

    print(json.dumps(results, indent=2))
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'[INFO]latent refinement analysis saved in {output_path}')
