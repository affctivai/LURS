import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from transformers import PreTrainedModel
from abc import ABC, abstractmethod
from module.ldm.diffusion_module import register_schedule
from module.ldm.util import q_sample, extract_into_tensor
from .util import *
from .wrappers import Wrapper_pretrain
import sys


__all__ = ["pretrainedmodule"]

class pretrainedmodule(Wrapper_pretrain):
    def __init__(self, encoder:nn.Module,
                 denoising_module:nn.Module,
                 pretrained_LM:PreTrainedModel,
                 is_con: bool,
                 is_geo: bool,
                 is_kl: bool,
                 is_diffusion: bool,
                 device: torch.device,
                 noise_scheduler:str,
                 time_step:int
                 ):
    
        super(pretrainedmodule, self).__init__()
        
        self.encoder = encoder

        self.denoising_module = denoising_module
        if is_diffusion:
            self._schedule = register_schedule(beta_schedule = noise_scheduler,timesteps=time_step, device = device)
            # self._schedule = register_schedule(beta_schedule = noise_scheduler, device = device)
            self.time_step = time_step
        self.pretrained_LM = pretrained_LM

        self.is_con = is_con
        self.is_geo = is_geo
        self.is_kl = is_kl
        self.is_diffusion = is_diffusion


        self.alpha = 0.0
        self.beta = 0.0
        self.theta = 0.0

        if self.is_kl:
            self.alpha = 0.001
        if self.is_con:
            self.beta = 0.001
            self.contrastive_learning = ContrastiveLatentLoss(gamma=0.075)
        if self.is_geo:
            self.theta = 0.001
            self.geometry_loss = GeometryPreservingLoss()

        self.fc1 = nn.Linear(105, 512)
        self.fc2 = nn.Linear(1024, 512)
        # if self.is_DConvNet:
        #     self.fc3 = nn.Linear(1024, 105)
        # else:
        #     self.fc3 = nn.Linear(512, 105)
        self.fc3 = nn.Linear(512, 105)




    def encode(self, x):
        posterior = self.encoder.encode(x)
        encoded_x = posterior.sample() # (B, 56, 105)
        z = self.fc1(encoded_x) # (B, 56, 512)
        # if self.is_con:
        #     z = F.layer_norm(z, z.shape[-1:])
        kl_loss = posterior.kl()
        return z, kl_loss
    
    def denoising(self, x, emb = None, condition = None):
        return self.denoising_module(x, emb, condition)
    
    def _feature_extraction_forward(self, z, x, condition):
        timesteps = None
        # --- feature extraction ---
        predicted_noise = self.denoising(z, timesteps, condition)
        z_recon = self.fc3(predicted_noise)
        # --- decode ---
        x_recon = self.encoder.decode(z_recon)
        loss = F.mse_loss(x_recon, x)
        return x_recon, loss

    def text_embedding(self, x):
        embeded_context = self.pretrained_LM.model.shared(x)
        condition = self.fc2(embeded_context)

        return condition
    
    def forward(self, x, context, target_ids_batch, eeg_mask=None):

        z, kl_loss = self.encode(x)


        condition = self.text_embedding(context)
        
        if self.is_diffusion:
            device = x.device
            # --- diffusion process ---
            noise = torch.randn_like(z, device=device)
            timesteps = torch.randint(0, self.time_step, (z.shape[0],), device=device).long() # (B,)
            x_noisy = q_sample(z, timesteps, self._schedule['sqrt_alphas_cumprod'], self._schedule['sqrt_one_minus_alphas_cumprod'], noise=noise) # (B, 1024, 8)
            predicted_noise = self.denoising(x_noisy, timesteps, condition)

            # predicted_noise = self.denoising(x_noisy, timesteps)
            # --- recon latent vector z ---
            sqrt_one_minus_alphas_cumprod = extract_into_tensor(self._schedule['sqrt_one_minus_alphas_cumprod'], timesteps, predicted_noise.shape)    
            sqrt_alphas_cumprod = extract_into_tensor(self._schedule['sqrt_alphas_cumprod'], timesteps, predicted_noise.shape)
            z_recon = (x_noisy - (sqrt_one_minus_alphas_cumprod * predicted_noise)) / sqrt_alphas_cumprod # (B, 56, 106)
            z_recon = self.fc3(z_recon)
            # --- decode ---
            x_recon = self.encoder.decode(z_recon)
            # --- loss design ---    
            loss = F.mse_loss(predicted_noise, noise)
        

        else:
            x_recon, loss = self._feature_extraction_forward(z, x, condition)


        if self.is_kl:
            loss = loss + self.alpha * kl_loss
        if self.is_con or self.is_geo:
            valid_mask = (target_ids_batch != -100)
            valid_mask_float = valid_mask.float()
            if self.is_con:
                loss = loss + self.beta * self.contrastive_learning(condition, z, valid_mask_float, eeg_mask)
                # loss = loss + self.beta * self.contrastive_learning(condition, z, None, None)
            if self.is_geo:
                loss = loss + self.theta * self.geometry_loss(z, x, valid_mask_float)

        return x_recon, loss
    