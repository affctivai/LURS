from ldm.openaimodel import UNetModel
import torch
from functools import partial
import numpy as np
import torch.nn as nn
from ldm.util import nonlinearity, exists, default, q_sample, extract_into_tensor
from ldm.autoencoder import AutoencoderKL

def register_schedule(given_betas=None, beta_schedule="linear", timesteps=1000,
                        linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3, device='cuda'):
    if exists(given_betas):
        betas = given_betas
    else:
        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                    cosine_s=cosine_s)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

    timesteps, = betas.shape
    num_timesteps = int(timesteps)
    linear_start = linear_start
    linear_end = linear_end
    assert alphas_cumprod.shape[0] == num_timesteps, 'alphas have to be defined for each timestep'

    to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

    # betas =  to_torch(betas)
    # alphas_cumprod = to_torch(alphas_cumprod)
    # alphas_cumprod_prev = to_torch(alphas_cumprod_prev)
    v_posterior=0
    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = (1 - v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                1. - alphas_cumprod) + v_posterior * betas
    # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

    parameterization = 'eps'
    if parameterization == "eps":
        lvlb_weights = betas ** 2 / (
                    2 * posterior_variance * alphas * (1 - alphas_cumprod))
    elif parameterization == "x0":
        lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
    else:
        raise NotImplementedError("mu not supported")
    # TODO how to choose this term
    lvlb_weights[0] = lvlb_weights[1]
    lvlb_weights = to_torch(lvlb_weights)
    assert not torch.isnan(lvlb_weights).all()

    return {
        'betas': to_torch(betas),
        'alphas_cumprod': to_torch(alphas_cumprod),
        'alphas_cumprod_prev': to_torch(alphas_cumprod_prev),
        'sqrt_alphas_cumprod': to_torch(np.sqrt(alphas_cumprod)),
        'sqrt_one_minus_alphas_cumprod': to_torch(np.sqrt(1. - alphas_cumprod)),
        'log_one_minus_alphas_cumprod': to_torch(np.log(1. - alphas_cumprod)),
        'sqrt_recip_alphas_cumprod': to_torch(np.sqrt(1. / alphas_cumprod)),
        'sqrt_recipm1_alphas_cumprod': to_torch(np.sqrt(1. / alphas_cumprod - 1)),
        'posterior_variance': to_torch(posterior_variance),
        'posterior_log_variance_clipped': to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        'posterior_mean_coef1': to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)),
        'posterior_mean_coef2': to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)),
        'lvlb_weights': lvlb_weights
    }
def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()



if __name__ == '__main__':
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = UNetModel(image_size=850,dims = 1, in_channels=56, out_channels=56, model_channels=320, attention_resolutions=[4, 2, 1], num_res_blocks=2, channel_mult=[1, 2, 4, 4], num_heads=8, use_spatial_transformer=True, transformer_depth=1, context_dim=1280, use_checkpoint=True, legacy=False)
    unet = unet.to(device)
    
    # 예시 입력 데이터
    batch_size = 8
    channels = 4
    context_dim = 1280
    context_dim = 560
    seq_len = 56
    text = torch.rand(batch_size, seq_len, context_dim, device=device)
    timesteps = torch.randint(0, 1000, (batch_size,), device=device).long()

    autoencoder = AutoencoderKL(
        embed_dim=56,
        monitor="val/rec_loss",
        ddconfig={
            "double_z": True,
            "z_channels": 58,
            "resolution": 256,
            "in_channels": 56,
            "out_ch": 56,
            "ch": 128,
            "ch_mult": [1, 2, 4, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0
        },
        lossconfig={
            "target": "torch.nn.Identity"
        }
    )
    autoencoder = autoencoder.to(device)

    posterior = autoencoder.encode(text)
    z = posterior.sample()

    noise = None
    noise = default(noise, lambda: torch.randn_like(z)).to(device)

    schedule = register_schedule(device = device)
    x_noisy = q_sample(z, timesteps, schedule['sqrt_alphas_cumprod'], schedule['sqrt_one_minus_alphas_cumprod'], noise=noise)
    print('text: ', z.shape)
    print('x_noisy: ', x_noisy.shape)

    context = torch.rand(batch_size, seq_len, context_dim)  # (16, 1280)
    context = context.to(device)
    # 예시 타임스텝
    
    x_noisy = x_noisy.to(device)
    # forward 함수 호출
    predicted_noise = unet(x_noisy, timesteps, context)
    print('predicted_noise :', predicted_noise.shape)  # 결과 확인
    # x0_est = (x_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
    
        

    sqrt_one_minus_alphas_cumprod = extract_into_tensor(schedule['sqrt_one_minus_alphas_cumprod'], timesteps, predicted_noise.shape)    
    sqrt_alphas_cumprod = extract_into_tensor(schedule['sqrt_alphas_cumprod'], timesteps, predicted_noise.shape)

    x_recon = (x_noisy - (sqrt_one_minus_alphas_cumprod * predicted_noise)) / sqrt_alphas_cumprod
    # x_recon = (x_noisy - schedule['sqrt_one_minus_alphas_cumprod'] * predicted_noise) / schedule['sqrt_alphas_cumprod']

    output = autoencoder.decode(x_recon)
    print('output: ', output.shape)  # 결과 확인
