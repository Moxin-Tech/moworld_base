import math
from typing import Optional
from einops import rearrange

import torch
from torch import nn
import torch_npu

from mindspeed_mm.models.common.blocks import MLP
from mindspeed_mm.models.common.activations import get_activation_layer
from mindspeed_mm.models.common.normalize import normalize as get_norm_layer
from mindspeed_mm.models.common.embeddings.time_embeddings import TimeStepEmbedding


class TextProjection(nn.Module):
    """
    Projects text embeddings. Also handles dropout for classifier-free guidance.
    """

    def __init__(self, in_channels, hidden_size, act_layer):
        super().__init__()
        self.linear_1 = nn.Linear(
            in_features=in_channels,
            out_features=hidden_size,
            bias=True,
        )
        self.act_1 = act_layer()
        self.linear_2 = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=True,
        )

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class IndividualTokenRefinerBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        heads_num,
        mlp_width_ratio: str = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layernorm",
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        self.head_dim = head_dim
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        act_layer = get_activation_layer(act_type)
        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True,),
        )
        # Zero-initialize the modulation
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

        self.norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=True, eps=1e-6,
        )

        self.self_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=qkv_bias,
        )

        self.self_attn_q_norm = (
            get_norm_layer(head_dim, affine=True, eps=1e-6, norm_type=qk_norm_type)
            if qk_norm
            else nn.Identity()
        )

        self.self_attn_k_norm = (
            get_norm_layer(head_dim, affine=True, eps=1e-6, norm_type=qk_norm_type)
            if qk_norm
            else nn.Identity()
        )

        self.self_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias,
        )

        self.norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=True, eps=1e-6,
        )

        self.mlp = MLP(
            in_channels=hidden_size,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=mlp_drop_rate
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,  # timestep_aware_representations + context_aware_representations
        attn_mask: torch.Tensor = None,
    ):
        gate_msa, gate_mlp = self.adaLN_modulation(c).unsqueeze(1).chunk(2, dim=-1)

        norm_x = self.norm1(x)
        qkv = self.self_attn_qkv(norm_x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
        # Apply QK-Norm if needed
        q = self.self_attn_q_norm(q).to(v)
        k = self.self_attn_k_norm(k).to(v)

        # Self-Attention
        attn = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            head_num=self.heads_num,
            atten_mask=~attn_mask,
            input_layout="BSND",
            scale=1 / math.sqrt(self.head_dim)
        )[0]

        attn = attn.view(attn.shape[0], attn.shape[1], -1)  # bsnd -> bsh
        x = x + self.self_attn_proj(attn) * gate_msa

        # FFN Layer
        x = x + self.mlp(self.norm2(x)) * gate_mlp

        return x


class IndividualTokenRefiner(nn.Module):
    def __init__(
        self,
        hidden_size,
        heads_num,
        depth,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                IndividualTokenRefinerBlock(
                    hidden_size=hidden_size,
                    heads_num=heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    act_type=act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.LongTensor,
        mask: Optional[torch.Tensor] = None,
    ):
        self_attn_mask = None
        if mask is not None:
            batch_size = mask.shape[0]
            seq_len = mask.shape[-1]
            mask = mask.to(x.device)
            # batch_size x 1 x seq_len x seq_len
            self_attn_mask_1 = mask.view(batch_size, 1, 1, seq_len).repeat(
                1, 1, seq_len, 1
            )
            # batch_size x 1 x seq_len x seq_len
            self_attn_mask_2 = self_attn_mask_1.transpose(2, 3)
            # batch_size x 1 x seq_len x seq_len, 1 for broadcasting of heads_num
            self_attn_mask = (self_attn_mask_1 & self_attn_mask_2).bool()
            # avoids self-attention weight being NaN for padding tokens
            self_attn_mask[:, :, :, 0] = True

        for block in self.blocks:
            x = block(x, c, self_attn_mask)
        return x


class SingleTokenRefiner(nn.Module):
    """
    A single token refiner block for llm text embedding refine.
    """
    def __init__(
        self,
        in_channels,
        hidden_size,
        time_embed_dim,
        heads_num,
        depth,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.input_embedder = nn.Linear(
            in_channels, hidden_size, bias=True,
        )

        act_layer = get_activation_layer(act_type)
        # Build timestep embedding layer
        self.t_embedder = TimeStepEmbedding(time_embed_dim, time_embed_dim=hidden_size)
        # Build context embedding layer
        self.c_embedder = TextProjection(
            in_channels, hidden_size, act_layer
        )

        self.individual_token_refiner = IndividualTokenRefiner(
            hidden_size=hidden_size,
            heads_num=heads_num,
            depth=depth,
            mlp_width_ratio=mlp_width_ratio,
            mlp_drop_rate=mlp_drop_rate,
            act_type=act_type,
            qk_norm=qk_norm,
            qk_norm_type=qk_norm_type,
            qkv_bias=qkv_bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.LongTensor,
        mask: Optional[torch.LongTensor] = None,
    ):
        timestep_aware_representations = self.t_embedder(t)

        if mask is None:
            context_aware_representations = x.mean(dim=1)
        else:
            mask_float = mask.float().unsqueeze(-1)  # [b, s1, 1]
            context_aware_representations = (x * mask_float).sum(
                dim=1
            ) / mask_float.sum(dim=1)
        context_aware_representations = self.c_embedder(context_aware_representations)
        c = timestep_aware_representations + context_aware_representations

        x = self.input_embedder(x)

        x = self.individual_token_refiner(x, c, mask)

        return x
