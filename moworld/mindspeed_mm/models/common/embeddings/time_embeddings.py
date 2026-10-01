import math

import torch
import torch.nn as nn
from einops import rearrange, repeat


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False, dtype=torch.float32):
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    return embedding.to(dtype)


class TimeStepEmbedding(nn.Module):
    def __init__(self, hidden_size, time_embed_dim=None, max_period=10000, repeat_only=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.time_embed_dim = time_embed_dim if time_embed_dim is not None else hidden_size
        self.max_period = max_period
        self.repeat_only = repeat_only
        self.time_embed = nn.Sequential(
            nn.Linear(self.hidden_size, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the module (assuming that all the module parameters have the same dtype)."""
        params = tuple(self.parameters())
        if len(params) > 0:
            return params[0].dtype
        else:
            buffers = tuple(self.buffers())
            return buffers[0].dtype

    def forward(self, timesteps):
        emb = timestep_embedding(timesteps, self.hidden_size, self.max_period, self.repeat_only, dtype=self.dtype)
        return self.time_embed(emb)
