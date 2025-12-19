import torch.nn as nn
import torch.utils.data
import torch

# from module.ldm.diffusion_module import register_schedule
# from module.ldm.util import q_sample, extract_into_tensor
from model.util import *
from .wrappers import Wrapper_DELTA, Wrapper_pretrain
from module.ldm.diffusion_module import register_schedule
from module.ldm.util import q_sample, extract_into_tensor

__all__ = ["DELTA"]
    
class DELTA(Wrapper_DELTA):
    def __init__(self, pretrained: Wrapper_pretrain, is_diffusion: bool,
                 noise_scheduler:str,
                 time_step:int, device: torch.device):
        super(DELTA, self).__init__()
        self.layer=pretrained

        # self.is_diffusion = is_diffusion[0]
        # self.is_DConvNet = is_diffusion[1]
        self.is_diffusion = is_diffusion
        
        if self.is_diffusion:
            self._schedule = register_schedule(beta_schedule = noise_scheduler, device = device)
            self.time_step = time_step

        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(840, 512)
        # if self.is_DConvNet:
        #     self.fc3 = nn.Linear(1024, 105)
        # else:
        #     self.fc3 = nn.Linear(512, 105)
        self.fc3 = nn.Linear(512, 105)
    @torch.no_grad()
    def generate(
            self,
            input_embeddings_batch, input_masks_batch,dummy_decoder_input_ids, context,
            **kwargs,
    ):
        

        # z, _ = self.layer.encode(input_embeddings_batch)
        
        # latent_vector = self.layer.denoising(z) # (B, 56, 512)

        # if not self.is_DConvNet:
        #     embedding_out = self.fc1(latent_vector) # (B, 56, 1024)
        # else:
        #     embedding_out = latent_vector


        z, _ = self.layer.encode(input_embeddings_batch)

        condition = self.layer.text_embedding(context)

        if self.is_diffusion:
            device = input_embeddings_batch.device
            # --- diffusion process ---
            noise = torch.randn_like(z, device=device)
            timesteps = torch.randint(0, self.time_step, (z.shape[0],), device=device).long() # (B,)
            x_noisy = q_sample(z, timesteps, self._schedule['sqrt_alphas_cumprod'], self._schedule['sqrt_one_minus_alphas_cumprod'], noise=noise) # (B, 1024, 8)
            predicted_noise = self.layer.denoising(x_noisy, timesteps, condition)
            # --- recon latent vector z ---
            sqrt_one_minus_alphas_cumprod = extract_into_tensor(self._schedule['sqrt_one_minus_alphas_cumprod'], timesteps, predicted_noise.shape)    
            sqrt_alphas_cumprod = extract_into_tensor(self._schedule['sqrt_alphas_cumprod'], timesteps, predicted_noise.shape)
            z_recon = (x_noisy - (sqrt_one_minus_alphas_cumprod * predicted_noise)) / sqrt_alphas_cumprod # (B, 56, 106)
            z_recon = self.fc3(z_recon)
            z_recon = self.layer.encoder.decode(z_recon)
            latent_vector = self.fc2(z_recon)
        else:
            z, _ = self.layer.encode(input_embeddings_batch)
            
            # condition = self.layer.text_embedding(context)
            latent_vector = self.layer.denoising(z) # (B, 56, 512)

        embedding_out = self.fc1(latent_vector) # (B, 56, 1024)

        
        output=self.layer.pretrained_LM.generate(
            inputs_embeds = embedding_out,
            attention_mask = input_masks_batch[:,:embedding_out.shape[1]],
            decoder_input_ids = dummy_decoder_input_ids,  # Exclude the last token for decoder input
            **kwargs,
            )
        

        return output


    def forward(self, input_embeddings_batch, input_masks_batch, target_ids_batch_converted, context):
        '''
        input_embeddings: EEG [32, 56, 840]
        input_masks_batch: [32, 56]
        input_masks_invert: [32, 56]
        target_ids_batch_converted: Target Text [32, 56]
        context: Target Text [32, 56]
        '''

        # z, _ = self.layer.encode(input_embeddings_batch)
        
        # # condition = self.layer.text_embedding(context)
        # latent_vector = self.layer.denoising(z) # (B, 56, 512)
        
        # if not self.is_DConvNet:
        #     embedding_out = self.fc1(latent_vector) # (B, 56, 1024)
        # else:
        #     embedding_out = latent_vector
        
        # out = self.layer.pretrained_LM(inputs_embeds = embedding_out, attention_mask = input_masks_batch,
        #                                 return_dict = True, labels = target_ids_batch_converted)# (B, 56, vocab_size)
        # return out, out.loss

        z, _ = self.layer.encode(input_embeddings_batch)

        condition = self.layer.text_embedding(context)

        if self.is_diffusion:
            device = input_embeddings_batch.device
            # --- diffusion process ---
            noise = torch.randn_like(z, device=device)
            timesteps = torch.randint(0, self.time_step, (z.shape[0],), device=device).long() # (B,)
            x_noisy = q_sample(z, timesteps, self._schedule['sqrt_alphas_cumprod'], self._schedule['sqrt_one_minus_alphas_cumprod'], noise=noise) # (B, 1024, 8)
            predicted_noise = self.layer.denoising(x_noisy, timesteps, condition)
            # --- recon latent vector z ---
            sqrt_one_minus_alphas_cumprod = extract_into_tensor(self._schedule['sqrt_one_minus_alphas_cumprod'], timesteps, predicted_noise.shape)    
            sqrt_alphas_cumprod = extract_into_tensor(self._schedule['sqrt_alphas_cumprod'], timesteps, predicted_noise.shape)
            z_recon = (x_noisy - (sqrt_one_minus_alphas_cumprod * predicted_noise)) / sqrt_alphas_cumprod # (B, 56, 106)
            z_recon = self.fc3(z_recon)
            z_recon = self.layer.encoder.decode(z_recon)
            latent_vector = self.fc2(z_recon)
        else:
            z, _ = self.layer.encode(input_embeddings_batch)
            
            # condition = self.layer.text_embedding(context)
            latent_vector = self.layer.denoising(z) # (B, 56, 512)

        embedding_out = self.fc1(latent_vector) # (B, 56, 1024)
        
        out = self.layer.pretrained_LM(inputs_embeds = embedding_out, attention_mask = input_masks_batch,
                                        return_dict = True, labels = target_ids_batch_converted)# (B, 56, vocab_size)
        return out, out.loss        