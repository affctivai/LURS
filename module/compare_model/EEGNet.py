import sys
import torch.nn as nn
import torch as th
from .util import *
import math
import numpy as np
from inspect import isfunction
from einops import rearrange, repeat    
from torch import einsum
import torch.nn.functional as F
from module.ldm.util import checkpoint
from module.ldm.attention import SpatialTransformer
from module.ldm.attention import CrossAttention

from einops import repeat
import math
class EEGNet(nn.Module):
    def __init__(
            self,
            n_classes,
            input_shape, # 1, 1, 512, 56
            F1=None, # 16
            D=None, # 2
            F2=None, # auto
            T1=None, # 32
            T2=None, # 16
            P1=None, # 4
            P2=None, # 8
            drop_out=None, # 0.5
            pool_mode=None, # mean
            weight_init_method=None,
            sequence =56,
            output_features=512
    ):
        super().__init__()
        _, c, s, t = input_shape
        pooling_layer = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]

        if F2 == 'auto':
            F2 = F1 * D # 16 * 2 = 32

        # Spectral
        self.spectral = nn.Sequential(
            nn.Conv2d(1, F1, (1, T1), bias=False, padding='same'), # (Batch, 16, D, S)
            nn.BatchNorm2d(F1))

        # Spatial
        self.spatial = nn.Sequential(
            Conv2dWithConstraint(F1, F1 * D, (s, 1), bias=False, groups=F1), # (Batch, 32, 1, S)
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            pooling_layer((1, P1)), # (Batch, 32, 1, S/4)
            nn.Dropout(drop_out)
        )

        # Temporal
        self.temporal = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, T2), bias=False, padding='same', groups=F1 * D), # (Batch, 32, 1, S/4)
            nn.Conv2d(F2, F2, 1, bias=False), # (Batch, 32, 1, S/4)
            nn.BatchNorm2d(F2),
            nn.ELU(),
            pooling_layer((1, P2)), # (Batch, 32, 1, S/32)
            nn.Dropout(drop_out)
        )


        ### projection layers
        self.conv1x1 = nn.Conv1d(1, sequence, 1, bias=False)
        self.projection_head = nn.Linear(F2, output_features)


        initialize_weight(self, weight_init_method)

        self.crossattention = SpatialTransformer(in_channels=512, n_heads = 8, d_head= 64, context_dim=512, depth = 2)

        ### time embedding
        self.embed_dim = F1 // 4
        time_ebed = F1 // 2
        
        
        self.time_embedding = nn.Sequential(
            nn.Linear(self.embed_dim, time_ebed),
            nn.SiLU(),
            nn.Linear(time_ebed, time_ebed),
        )

        self.emb_layers = nn.ModuleList()

        block_dim = [time_ebed, F1, F1 * D, F2]
        # block_dim = [8, 16, 32, 32]
        for i in range(1,len(block_dim)):
            # time embedding
            emb_layer = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    block_dim[i-1],
                    block_dim[i],
                ),
            )
            self.emb_layers.append(emb_layer)

    def timestep_embedding(self, timesteps, dim, max_period=10000, repeat_only=False):
        """
        Create sinusoidal timestep embeddings.
        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        if not repeat_only:
            half = dim // 2 # 95
            freqs = torch.exp(
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
            ).to(device=timesteps.device)
            args = timesteps[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        else:
            embedding = repeat(timesteps, 'b -> b d', d=dim)
        return embedding

    def forward(self, x, timesteps = None, condition = None):

        x = x.permute(0, 2, 1).unsqueeze(1) # B, 1, D, S
        if timesteps is not None:
            t_emb = self.timestep_embedding(timesteps, self.embed_dim, repeat_only=False)
            emb = self.time_embedding(t_emb) # B, F2

        out = self.spectral(x)
        if emb is not None and self.emb_layers is not None:
            emb = self.emb_layers[0](emb).type(out.dtype)
            while len(emb.shape) < len(out.shape):
                emb = emb[..., None]
            out = out + emb
            while emb.dim() > 2:
                emb = emb.squeeze(-1)

        out = self.spatial(out)
        if emb is not None and self.emb_layers is not None:
            emb = self.emb_layers[1](emb).type(out.dtype)
            while len(emb.shape) < len(out.shape):
                emb = emb[..., None]
            out = out + emb
            while emb.dim() > 2:
                emb = emb.squeeze(-1)                
        
        
        out = self.temporal(out)
        if emb is not None and self.emb_layers is not None:
            emb = self.emb_layers[2](emb).type(out.dtype)
            while len(emb.shape) < len(out.shape):
                emb = emb[..., None]
            out = out + emb
            while emb.dim() > 2:
                emb = emb.squeeze(-1)       

        out = out.squeeze(2)  # 높이(H)가 1인 차원 제거 -> (Batch, F2, 1)
        out = out.permute(0, 2, 1)  # 채널과 시간 차원 교환 -> (Batch,1, F2)
        out = self.conv1x1(out)  # -> (Batch, 56, F2)
        out = self.projection_head(out) # -> (Batch, 56, 512)
        out = self.crossattention(out.permute(0,2,1), condition)
        out = out.permute(0,2,1)

        return out