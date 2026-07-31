import math
import torch
import torch.nn as nn
from .util import *
from einops import repeat
from module.ldm.attention import SpatialTransformer


class MLP(nn.Module):
    def __init__(
            self,
            input_shape,       # (1, 1, 512, 56)
            hidden_dim=6144,   # width of each MLP block
            num_layers=10,     # number of stacked MLP blocks
            drop_out=0.5,
            weight_init_method=None,
            output_features=512,
    ):
        super().__init__()
        _, c, s, t = input_shape  # s: input feature dim, t: sequence length
        self.sequence = t

        self.input_proj = nn.Sequential(
            nn.Linear(s, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_out),
        )

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(drop_out),
            )
            for _ in range(num_layers)
        ])

        self.projection_head = nn.Linear(hidden_dim, output_features)

        initialize_weight(self, weight_init_method)

        self.crossattention = SpatialTransformer(in_channels=512, n_heads=8, d_head=64, context_dim=512, depth=2)

        ### time embedding, same convention as EEGNet/DeepConvNet
        self.embed_dim = hidden_dim // 8
        self.time_embed_dim = hidden_dim // 4
        self.time_embedding = nn.Sequential(
            nn.Linear(self.embed_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        self.emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.time_embed_dim, hidden_dim),
            )
            for _ in range(num_layers)
        ])

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
            half = dim // 2
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

    def forward(self, x, timesteps=None, condition=None):
        # x: (B, S=56, D=512)
        out = self.input_proj(x)  # (B, 56, hidden_dim)

        emb = None
        if timesteps is not None:
            t_emb = self.timestep_embedding(timesteps, self.embed_dim, repeat_only=False)
            emb = self.time_embedding(t_emb)  # (B, time_embed_dim)

        for i, block in enumerate(self.blocks):
            out = block(out)
            if emb is not None:
                cur_emb = self.emb_layers[i](emb).type(out.dtype)  # (B, hidden_dim)
                out = out + cur_emb.unsqueeze(1)

        out = self.projection_head(out)  # (B, 56, 512)
        out = self.crossattention(out.permute(0, 2, 1), condition)
        out = out.permute(0, 2, 1)  # (B, 56, 512)

        return out
