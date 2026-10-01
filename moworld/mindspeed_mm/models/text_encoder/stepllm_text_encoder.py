# Copyright 2025 StepFun Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# ==============================================================================
import math
from typing import Optional
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
import numpy as np

from transformers.modeling_utils import PreTrainedModel

from mindspeed_mm.models.common.normalize import normalize

try:
    from transformers.modeling_utils import PretrainedConfig
except ImportError:
    # support newer transformers for qwen3omni
    from transformers.modeling_utils import PreTrainedConfig as PretrainedConfig

DTYPE_FP16_MIN = float(np.finfo(np.float16).min)


def _get_alibi_slopes(n_heads):
    n = 2 ** math.floor(math.log2(n_heads))
    m0 = torch.tensor(2.0 ** (-8.0 / n), dtype=torch.float32).to("cpu")
    slopes = torch.pow(m0, torch.arange(1, n + 1, dtype=torch.float32).to("cpu"))
    if n < n_heads:
        m1 = torch.tensor(2.0**(-4.0 / n), dtype=torch.float32).to("cpu")
        mm = torch.pow(m1, torch.arange(1, 1 + 2 * (n_heads - n), 2, dtype=torch.float32).to("cpu"))
        slopes = torch.cat([slopes, mm])
    return slopes


def _get_mask(seq_len, b, n):
    slopes = _get_alibi_slopes(n)
    tril = torch.tril(torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool)).to(torch.int32)
    bias_row = torch.arange(seq_len).view(1, -1)
    bias_cols = torch.arange(seq_len).view(-1, 1)
    bias = -torch.sqrt(bias_cols - bias_row)
    bias = bias.view(1, seq_len, seq_len) * slopes.view(-1, 1, 1)
    bias = bias.masked_fill(tril == 0, DTYPE_FP16_MIN)
    return bias


class LLaMaEmbedding(nn.Module):
    """Language model embeddings."""

    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        padded_vocab_size = config.padded_vocab_size
        hidden_dropout = config.hidden_dropout
        self.word_embeddings = nn.Embedding(
            padded_vocab_size, hidden_size,
        )
        self.embedding_dropout = nn.Dropout(hidden_dropout)

    def forward(self, input_ids):
        # Embeddings.
        embeddings = self.word_embeddings(input_ids)
        # Data format change to avoid explicit transposes : [b s h] --> [s b h].
        embeddings = embeddings.transpose(0, 1).contiguous()
        # Dropout.
        embeddings = self.embedding_dropout(embeddings)
        return embeddings


class FlashSelfAttention(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super().__init__()
        self.dropout_p = attention_dropout

    def forward(self, q, k, v, cu_seqlens=None):
        if cu_seqlens is None:
            alibi_mask = _get_mask(q.size(1), q.size(0), q.size(2))
            alibi_mask = alibi_mask[:, :q.size(2), :, :].to(q.dtype).to(q.device)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=alibi_mask)
            output = output.transpose(1, 2)
        else:
            raise ValueError('cu_seqlens is not supported!')

        return output


class MultiQueryAttention(nn.Module):
    def __init__(self, cfg, layer_id=None):
        super().__init__()

        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.max_seq_len = cfg.seq_length
        self.use_flash_attention = cfg.use_flash_attn

        self.n_groups = cfg.num_attention_groups
        self.n_local_heads = cfg.num_attention_heads
        self.n_local_groups = self.n_groups

        self.wqkv = nn.Linear(cfg.hidden_size, cfg.hidden_size + self.head_dim * 2 * self.n_groups, bias=False)
        self.wo = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.core_attention = FlashSelfAttention(attention_dropout=cfg.attention_dropout)
        self.layer_id = layer_id

    def __call__(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        cu_seqlens: Optional[torch.Tensor],
    ):
        seqlen, bsz, dim = x.shape
        xqkv = self.wqkv(x)

        xq, xkv = torch.split(xqkv, (dim, self.head_dim * 2 * self.n_groups), dim=-1)

        # gather on 1st dimension
        xq = xq.view(seqlen, bsz, self.n_local_heads, self.head_dim)
        xkv = xkv.view(seqlen, bsz, self.n_local_groups, 2 * self.head_dim)
        xk, xv = xkv.chunk(2, -1)

        # rotary embedding + flash attn
        xq = rearrange(xq, "s b h d -> b s h d")
        xk = rearrange(xk, "s b h d -> b s h d")
        xv = rearrange(xv, "s b h d -> b s h d")

        q_per_kv = self.n_local_heads // self.n_local_groups
        if q_per_kv > 1:
            b, s, h, d = xk.size()
            if h == 1:
                xk = xk.expand(b, s, q_per_kv, d)
                xv = xv.expand(b, s, q_per_kv, d)
            else:
                ''' To cover the cases where h > 1, we have
                    the following implementation, which is equivalent to:
                        xk = xk.repeat_interleave(q_per_kv, dim=-2)
                        xv = xv.repeat_interleave(q_per_kv, dim=-2)
                    but can avoid calling aten::item() that involves cpu.
                '''
                idx = torch.arange(q_per_kv * h, device=xk.device).reshape(q_per_kv, -1).permute(1, 0).flatten()
                xk = torch.index_select(xk.repeat(1, 1, q_per_kv, 1), 2, idx).contiguous()
                xv = torch.index_select(xv.repeat(1, 1, q_per_kv, 1), 2, idx).contiguous()

        if self.use_flash_attention:
            output = self.core_attention(xq, xk, xv, cu_seqlens=cu_seqlens)
            # reduce-scatter only support first dimension now
            output = rearrange(output, "b s h d -> s b (h d)").contiguous()
        else:
            xq, xk, xv = [
                rearrange(x, "b s ... -> s b ...").contiguous()
                for x in (xq, xk, xv)
            ]
            output = self.core_attention(xq, xk, xv, mask)
        output = self.wo(output)
        return output


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
    ):
        super().__init__()

        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def swiglu(self, x):
        x = torch.chunk(x, 2, dim=-1)
        return F.silu(x[0]) * x[1]

    def __call__(self, x):
        x = self.swiglu(self.w1(x))
        output = self.w2(x)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, cfg, layer_id: int):
        super().__init__()

        self.n_heads = cfg.num_attention_heads
        self.dim = cfg.hidden_size
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.attention = MultiQueryAttention(cfg, layer_id=layer_id)

        self.feed_forward = FeedForward(dim=cfg.hidden_size, hidden_dim=cfg.ffn_hidden_size)
        self.layer_id = layer_id
        self.attention_norm = normalize(cfg.hidden_size, eps=cfg.layernorm_epsilon, norm_type="rmsnorm")
        self.ffn_norm = normalize(cfg.hidden_size, eps=cfg.layernorm_epsilon, norm_type="rmsnorm")

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        cu_seqlens: Optional[torch.Tensor],
    ):
        residual = self.attention(self.attention_norm(x), mask, cu_seqlens)
        h = x + residual
        ffn_res = self.feed_forward(self.ffn_norm(h))
        out = h + ffn_res
        return out


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_layers = config.num_layers
        self.layers = self._build_layers(config)

    def _build_layers(self, config):
        layers = nn.ModuleList()
        for layer_id in range(self.num_layers):
            layers.append(
                TransformerBlock(
                    config,
                    layer_id=layer_id + 1,
                )
            )
        return layers

    def forward(
        self,
        hidden_states,
        attention_mask,
        cu_seqlens=None,
    ):
        for _, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask,
                cu_seqlens,
            )
        return hidden_states


class StepLLmModel(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config):
        super().__init__(config)
        self.tok_embeddings = LLaMaEmbedding(config)
        self.transformer = Transformer(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        **kwargs
    ):
        hidden_states = self.tok_embeddings(input_ids)

        hidden_states = self.transformer(
            hidden_states,
            attention_mask,
        )
        return {"last_hidden_state": hidden_states.transpose(0, 1)}
