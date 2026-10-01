import torch
from torch import nn
from diffusers.models.embeddings import PixArtAlphaTextProjection, Timesteps, TimestepEmbedding


class CombinedTimestepTextProjEmbeddings(nn.Module):
    def __init__(self, timestep_embed_dim, embedding_dim, pooled_projection_dim):
        super().__init__()

        self.time_proj = Timesteps(num_channels=timestep_embed_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=timestep_embed_dim, time_embed_dim=embedding_dim)
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")

    def forward(self, timestep, pooled_projection):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))  # (N, D)

        pooled_projections = self.text_embedder(pooled_projection)
        timesteps_emb = timesteps_emb.float()
        pooled_projections = pooled_projections.float()
        conditioning = timesteps_emb + pooled_projections
        if conditioning.dtype != torch.float32:
            raise ValueError("Conditioning embeddings must be float32.")

        return conditioning
