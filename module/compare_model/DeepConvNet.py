from typing import List

import torch
import torch.nn as nn
from .util import *

from einops import rearrange, repeat    
from torch import einsum
from inspect import isfunction
from module.ldm.attention import SpatialTransformer
from module.ldm.attention import CrossAttention
from einops import repeat
import math
class DeepConvNet(nn.Module):
    def __init__(
            self,
            input_shape: List[int], # 32, 1, 512, 56
            first_conv_length: int, # 5
            block_out_channels: List[int], # 32, 64, 128, 256, 512
            pool_size: int, # 3
            weight_init_method=None,
    ) -> None:
        super(DeepConvNet, self).__init__()

        self.embedding_dim = block_out_channels[-1]
        b, c, s, t = input_shape
        # temporal_padding = (0, (first_conv_length - 1) // 2)
        temporal_padding = 3

        self.first_conv_block = nn.Sequential(
            Conv2dWithConstraint(1, block_out_channels[0], kernel_size=(1, first_conv_length), max_norm=2, padding=temporal_padding),
            Conv2dWithConstraint(block_out_channels[0], block_out_channels[1], kernel_size=(s, 1), bias=False,
                                 max_norm=2),
            nn.BatchNorm2d(block_out_channels[1]),
            nn.ELU(),
            nn.MaxPool2d((1, pool_size))
        )

        self.deep_block = nn.ModuleList(
            [self.default_block(block_out_channels[i - 1], block_out_channels[i], first_conv_length, pool_size) for i in
             range(2, 5)]
        )

        self.projection = nn.Linear(25, 56, bias=False)

        initialize_weight(self, weight_init_method)
        # self.crossattention= CrossAttention(query_dim=512, context_dim=512)
        self.crossattention = SpatialTransformer(in_channels=512, n_heads = 8, d_head= 64, context_dim=512, depth = 2)
        ### time embedding
        self.embed_dim = block_out_channels[0]
        
        
        self.time_embed_dim = block_out_channels[1]
        self.time_embedding = nn.Sequential(
            nn.Linear(block_out_channels[0], self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        self.emb_layers = nn.ModuleList()

        # block_out_channels = [32, 64, 128, 256, 512]
        for i in range(2, 5):
            # time embedding
            emb_layer = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    block_out_channels[i - 1],
                    block_out_channels[i],
                ),
            )
            self.emb_layers.append(emb_layer)



    def default_block(self, in_channels, out_channels, T, P):
        # temporal_padding = (0, (T - 1) // 2)
        temporal_padding = 3
        default_block = nn.Sequential(
            nn.Dropout(0.5),
            Conv2dWithConstraint(in_channels, out_channels, (1, T), bias=False, max_norm=2, padding=temporal_padding),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.MaxPool2d((1, P))
        )
        return default_block

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
        # timestep: (B,)
        batch_size = x.shape[0]
        x = x.permute(0, 2, 1).unsqueeze(1) # [32, 1, 512, 56]
        out = self.first_conv_block(x) # B 64 7 19

        if timesteps is not None:
            t_emb = self.timestep_embedding(timesteps, self.embed_dim, repeat_only=False)
            emb = self.time_embedding(t_emb)
            # B, 64

        for i, block in enumerate(self.deep_block):
            out = block(out)
            
            if emb is not None and self.emb_layers is not None:
                emb = self.emb_layers[i](emb).type(out.dtype)
                while len(emb.shape) < len(out.shape):
                    emb = emb[..., None]
                out = out + emb
                while emb.dim() > 2:
                    emb = emb.squeeze(-1)
            # after deep block 0: torch.Size([32, 128, 13, 7])
            # after deep block 1: torch.Size([32, 256, 19, 3])
            # after deep block 2: torch.Size([32, 512, 25, 1])

        out = out.view(batch_size, self.embedding_dim, -1) # -> [32, 512, 25]
        out = self.projection(out)
        out = out.permute(0, 2, 1)
        # out = self.crossattention(out, condition)
        out = self.crossattention(out.permute(0,2,1), condition)
        out = out.permute(0,2,1)
        return out
