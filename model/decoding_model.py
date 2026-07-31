import torch.nn as nn
import torch.utils.data
import torch

# from module.ldm.diffusion_module import register_schedule
# from module.ldm.util import q_sample, extract_into_tensor
from model.util import *
from .wrappers import Wrapper_DELTA, Wrapper_pretrain
from module.ldm.diffusion_module import register_schedule
from module.ldm.util import q_sample, extract_into_tensor, noise_like, default
import math
from tqdm import tqdm

__all__ = ["DELTA"]

class DELTA(Wrapper_DELTA):
    def __init__(self, pretrained: Wrapper_pretrain, is_diffusion: bool, is_compare_model: bool,
                 noise_scheduler:str,
                 time_step:int, device: torch.device):
        super(DELTA, self).__init__()
        self.layer=pretrained

        # self.is_diffusion = is_diffusion[0]
        # self.is_DConvNet = is_diffusion[1]
        self.is_diffusion = is_diffusion
        self.is_compare_model = is_compare_model

        # if self.is_diffusion:
        self._schedule = register_schedule(beta_schedule = noise_scheduler, device = device)
        self.time_step = time_step
            # self.fc2 = nn.Linear(840, 512)
            # self.fc3 = nn.Linear(512, 105)

        self.fc1 = nn.Linear(512, 1024)
        # self.dropout = nn.Dropout(drop_out)


        # if self.is_DConvNet:
        #     self.fc3 = nn.Linear(1024, 105)
        # else:
        #     self.fc3 = nn.Linear(512, 105)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self._schedule['sqrt_recip_alphas_cumprod'], t, x_t.shape) * x_t -
                extract_into_tensor(self._schedule['sqrt_recipm1_alphas_cumprod'], t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self._schedule['posterior_mean_coef1'], t, x_t.shape) * x_start +
                extract_into_tensor(self._schedule['posterior_mean_coef2'], t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self._schedule['posterior_variance'], t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self._schedule['posterior_log_variance_clipped'], t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, condition=None, clip_denoised=False):
        model_out = self.layer.denoising(x, t, condition)
        x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    # @torch.no_grad()
    # def p_mean_variance(self, x, t, clip_denoised: bool):
    #     model_out = self.model(x, t)
    #     if self.parameterization == "eps":
    #         x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
    #     elif self.parameterization == "x0":
    #         x_recon = model_out
    #     if clip_denoised:
    #         x_recon.clamp_(-1., 1.)
    #
    #     model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
    #     return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, condition=None, clip_denoised=False, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, condition=condition, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # @torch.no_grad()
    # def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
    #     b, *_, device = *x.shape, x.device
    #     model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
    #     noise = noise_like(x.shape, device, repeat_noise)
    #     # no noise when t == 0
    #     nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
    #     return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, condition=None, return_intermediates=False):
        device = self._schedule['betas'].device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.time_step)), desc='Sampling t', total=self.time_step):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), condition=condition)
            intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    # @torch.no_grad()
    # def p_sample_loop(self, shape, return_intermediates=False):
    #     device = self.betas.device
    #     b = shape[0]
    #     img = torch.randn(shape, device=device)
    #     intermediates = [img]
    #     for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
    #         img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
    #                             clip_denoised=self.clip_denoised)
    #         if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
    #             intermediates.append(img)
    #     if return_intermediates:
    #         return img, intermediates
    #     return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)


    @torch.no_grad()
    def encode_and_sample(self, input_embeddings_batch):
        """encode() + reverse-diffusion p_sample_loop() shared by evaluate()/generate() so a
        caller can run it once and pass embedding_out to both instead of sampling twice."""
        z, _ = self.layer.encode(input_embeddings_batch)
        latent_vector = self.p_sample_loop(z.shape) # (B, 56, 512)
        embedding_out = self.fc1(latent_vector) # (B, 56, 1024)
        return embedding_out

    @torch.no_grad()
    def generate(
            self,
            input_embeddings_batch, input_masks_batch,dummy_decoder_input_ids, context,
            embedding_out=None,
            **kwargs,
    ):
        if embedding_out is None:
            embedding_out = self.encode_and_sample(input_embeddings_batch)

        output=self.layer.pretrained_LM.generate(
            inputs_embeds = embedding_out,
            attention_mask = input_masks_batch[:,:embedding_out.shape[1]],
            decoder_input_ids = dummy_decoder_input_ids,  # Exclude the last token for decoder input
            **kwargs,
            )


        return output

    def evaluate(self, input_embeddings_batch, input_masks_batch, target_ids_batch_converted, context, embedding_out=None):
        if embedding_out is None:
            embedding_out = self.encode_and_sample(input_embeddings_batch)

        out = self.layer.pretrained_LM(inputs_embeds = embedding_out, attention_mask = input_masks_batch,
                                        return_dict = True, labels = target_ids_batch_converted)# (B, 56, vocab_size)

        return out
    
    def forward(self, input_embeddings_batch, input_masks_batch, target_ids_batch_converted, context, stage):
        '''
        input_embeddings: EEG [32, 56, 840]
        input_masks_batch: [32, 56]
        input_masks_invert: [32, 56]
        target_ids_batch_converted: Target Text [32, 56]
        context: Target Text [32, 56]
        stage: Training stage
        '''


        z, _ = self.layer.encode(input_embeddings_batch)

        
        if self.is_compare_model:
            # compare_model baselines (EEGNet/DConv/MLP) run as plain feature extractors, no diffusion schedule
            latent_vector = self.layer.denoising(z) # (B, 56, 512)
            
        elif self.is_diffusion:
            condition = self.layer.text_embedding(context)
            device = input_embeddings_batch.device
            # --- diffusion process ---
            noise = torch.randn_like(z, device=device)
            timesteps = torch.randint(0, self.time_step, (z.shape[0],), device=device).long() # (B,)
            x_noisy = q_sample(z, timesteps, self._schedule['sqrt_alphas_cumprod'], self._schedule['sqrt_one_minus_alphas_cumprod'], noise=noise) # (B, 1024, 8)
            predicted_noise = self.layer.denoising(x_noisy, timesteps, condition)
            # --- recon latent vector z ---
            sqrt_one_minus_alphas_cumprod = extract_into_tensor(self._schedule['sqrt_one_minus_alphas_cumprod'], timesteps, predicted_noise.shape)    
            sqrt_alphas_cumprod = extract_into_tensor(self._schedule['sqrt_alphas_cumprod'], timesteps, predicted_noise.shape)
            z_recon = (x_noisy - (sqrt_one_minus_alphas_cumprod * predicted_noise)) / sqrt_alphas_cumprod # (B, 56, 512)
            
            latent_vector = z_recon



        else:
            # condition = self.layer.text_embedding(context)
            latent_vector = self.layer.denoising(z) # (B, 56, 512)

            # device = input_embeddings_batch.device

            # # --- 식 11: deterministic forward-mean, τ_ref = 1, 노이즈 샘플링 없음 ---
            # tau_ref = torch.ones((z.shape[0],), device=device).long()  # τ_ref = 1
            # sqrt_ac  = extract_into_tensor(self._schedule['sqrt_alphas_cumprod'],
            #                             tau_ref, z.shape)            # √ᾱ_τref
            # sqrt_1mac = extract_into_tensor(self._schedule['sqrt_one_minus_alphas_cumprod'],
            #                                 tau_ref, z.shape)           # √(1-ᾱ_τref)
            # z_tilde = sqrt_ac * z                                       # Z̃_τref = √ᾱ_τref · Z_0

            # predicted_noise = self.layer.denoising(z_tilde, tau_ref, None) 
            # # --- 식 12: null condition(C=0)으로 노이즈 예측 후 x0-estimate ---
            # # null_condition = torch.zeros((z.shape[0], context.shape[1], 512), device=device, dtype=z_tilde.dtype)
            # # predicted_noise = self.layer.denoising(z_tilde, tau_ref, null_condition)   # ε̂_ref = ε_θ(Z̃, τref, 0)
            # z_star = (z_tilde - sqrt_1mac * predicted_noise) / sqrt_ac        # Z* = x0-estimate

            # latent_vector = z_star

        # embedding_out = self.dropout(self.fc1(latent_vector)) # (B, 56, 1024)
        embedding_out = self.fc1(latent_vector) # (B, 56, 1024)

        out = self.layer.pretrained_LM(inputs_embeds = embedding_out, attention_mask = input_masks_batch,
                                        return_dict = True, labels = target_ids_batch_converted)# (B, 56, vocab_size)
        
        # if stage == 'first_stage':
        #     return out, F.mse_loss(noise, predicted_noise, reduction='mean')
        # else:
        #     return out, out.loss  
        return out, out.loss      