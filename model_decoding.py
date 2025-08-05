import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from ldm.diffusion_module import register_schedule
from ldm.util import q_sample, extract_into_tensor, checkpoint

import torch, torch.nn.functional as F
import torch.nn as nn
import torch
from ldm.attention import SpatialTransformer
from ldm.openaimodel import UNetModel


def cos_sim(A, B, dim= -1, eps= 1e-4):

    # ℓ2-norm
    norm_A = F.normalize(A, p=2, dim=dim, eps=eps)
    norm_B = F.normalize(B, p=2, dim=dim, eps=eps)

    # sim = torch.matmul(norm_A, norm_B.transpose(-1, -2))
    return norm_A @ norm_B.transpose(-1, -2)

class ContrastiveLatentLoss(nn.Module):
    def __init__(self, gamma=0.07):
        """
        Contrastive loss for aligning EEG Condition with Text Latent in Latent Diffusion Model (LDM).
        """
        super(ContrastiveLatentLoss, self).__init__()
        self.gamma = gamma
        
        hidden_dim = 512 
        projection_dim = 256

        self.text_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        self.eeg_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, text_latent, eeg_condition, mask=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (text_latent, eeg_condition), self.parameters(), True
        )
    
    def _forward(self, text_latent, eeg_condition, mask=None):
        """
        Args:
            text_latent (torch.Tensor): Latent representation of text (batch_size, latent_dim)
            eeg_condition (torch.Tensor): EEG-based Condition Vector (batch_size, latent_dim)
        
        Returns:
            torch.Tensor: Contrastive loss value
        """
        
        batch_size, _, _ = text_latent.shape
        text_sent_emb = text_latent.mean(dim=1)  # (batch_size, hidden_dim)
        eeg_sent_emb = eeg_condition.mean(dim=1) # (batch_size, hidden_dim)

        text_embedding = self.text_projection(text_sent_emb)
        eeg_embedding = self.eeg_projection(eeg_sent_emb)


        # (2) 각 문장 내 단어 간 similarity 계산
        similarity = cos_sim(text_embedding, eeg_embedding) / self.gamma

        # (3) 정답 labels 생성 (각 단어가 동일 위치 단어와 매칭되도록)
        labels = torch.arange(batch_size).to(eeg_sent_emb.device)
        # labels shape: [batch_size, seq_len] (정답은 항상 [0,1,...,seq_len-1])

        # (5) Contrastive loss 계산 (단어 수준 Contrastive loss)
        loss = F.cross_entropy(similarity, labels)

        return loss

# warm-up 안 넣은 버전
class pretrainedmodule(nn.Module):
    def __init__(self, encoder, unet, pretrained, device):
        super(pretrainedmodule, self).__init__()
        self.alpha = 0.01
        self.beta = 1e-9
        # self.contrastive_learning = ContrastiveLatentLoss(gamma=0.075)

        self.encoder = encoder
        self.unet = unet
        self.schedule = register_schedule(device = device)
        self.pretrained = pretrained

        self.fc1 = nn.Linear(105, 512)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 105)

    def encode(self, x):
        posterior = self.encoder.encode(x)
        # encoded_x = posterior
        encoded_x = posterior.sample() # (B, 56, 105)
        z = self.fc1(encoded_x) # (B, 56, 512)
        kl_loss = posterior.kl()
        return z, kl_loss
        # return z
    
    def denoising(self, x, emb = None, condition = None):
        return self.unet(x, emb, condition)
    
    def text_embedding(self, x):
        embeded_context = self.pretrained.model.shared(x)
        condition = self.fc2(embeded_context)

        return condition
    
    def forward(self, x, context, target_ids_batch):
        device = x.device
        z, kl_loss = self.encode(x)
        # z = self.encode(x)

        noise = torch.randn_like(z, device=device)
        timesteps = torch.randint(0, 1000, (z.shape[0],), device=device).long() # (B,)
        x_noisy = q_sample(z, timesteps, self.schedule['sqrt_alphas_cumprod'], self.schedule['sqrt_one_minus_alphas_cumprod'], noise=noise) # (B, 1024, 8)

        condition = self.text_embedding(context)


        predicted_noise = self.denoising(x_noisy, timesteps, condition)

        # --- recon latent vector z ---
        sqrt_one_minus_alphas_cumprod = extract_into_tensor(self.schedule['sqrt_one_minus_alphas_cumprod'], timesteps, predicted_noise.shape)    
        sqrt_alphas_cumprod = extract_into_tensor(self.schedule['sqrt_alphas_cumprod'], timesteps, predicted_noise.shape)
        z_recon = (x_noisy - (sqrt_one_minus_alphas_cumprod * predicted_noise)) / sqrt_alphas_cumprod # (B, 56, 106)
        z_recon = self.fc3(z_recon)
        
        x_recon = self.encoder.decode(z_recon)
        # --- loss design ---    
        # loss = F.mse_loss(x_recon, x) + self.beta * kl_loss
        loss = F.mse_loss(x_recon, x)
        
        # loss = loss + self.alpha * self.contrastive_learning(z, condition)
        

        return x_recon, loss
    
    
class LDMTranslator(nn.Module):
    def __init__(self, pretrained):
        super(LDMTranslator, self).__init__()
        self.layer=pretrained
        self.fc1 = nn.Linear(512, 1024)
    
    @torch.no_grad()
    def generate(
            self,
            input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted,dummy_decoder_input_ids,
            generation_config = None,
            logits_processor = None,
            stopping_criteria = None,
            prefix_allowed_tokens_fn= None,
            synced_gpus= None,
            assistant_model = None,
            streamer= None,
            negative_prompt_ids= None,
            negative_prompt_attention_mask = None,
            **kwargs,
    ):
        z = self.layer.encode(input_embeddings_batch)

        latent_vector = self.layer.denoising(z) # (B, 56, 512)

        embedding_out = self.fc1(latent_vector) # (B, 56, 1024)

        
        output=self.layer.pretrained.generate(
            inputs_embeds = embedding_out,
            attention_mask = input_masks_batch[:,:embedding_out.shape[1]],
            decoder_input_ids = dummy_decoder_input_ids,  # Exclude the last token for decoder input
            **kwargs,
            )
        

        return output


    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted, context, stage = 'second_stage'):
        '''
        input_embeddings: EEG [32, 56, 840]
        input_masks_batch: [32, 56]
        input_masks_invert: [32, 56]
        target_ids_batch_converted: Target Text [32, 56]
        context: Target Text [32, 56]
        '''

        z = self.layer.encode(input_embeddings_batch)

 
        latent_vector = self.layer.denoising(z) # (B, 56, 512)


        embedding_out = self.fc1(latent_vector) # (B, 56, 1024)


        out = self.layer.pretrained(inputs_embeds = embedding_out, attention_mask = input_masks_batch,
                                        return_dict = True, labels = target_ids_batch_converted)# (B, 56, vocab_size)
        return out, out.loss
        
        

    def evaluation(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted, context):
        '''
        input_embeddings: EEG [32, 56, 840]
        input_masks_batch: [32, 56]
        input_masks_invert: [32, 56]
        target_ids_batch_converted: Target Text [32, 56]
        context: Target Text [32, 56]
        '''

        z = self.layer.encode(input_embeddings_batch)

        latent_vector = self.layer.denoising(z) # (B, 56, 512)

        embedding_out = self.fc1(latent_vector) # (B, 56, 1024)


        out = self.layer.pretrained(inputs_embeds = embedding_out, attention_mask = input_masks_batch,
                                        return_dict = True, labels = target_ids_batch_converted)
        return out