import torch
from torch import nn
from torch.nn import LayerNorm
from transformers.activations import ACT2FN

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec

from mindspeed_mm.models.common.module import MultiModalModule


class GlmMLP(MultiModalModule):
    def __init__(
        self,
        config: TransformerConfig,
        submodules: ModuleSpec,
    ):
        super().__init__(config=config)

        dim = config.out_hidden_size
        context_dim = config.intermediate_size
        hidden_act = config.hidden_act
        bias = False

        self.proj = nn.Linear(dim, dim, bias=bias)
        self.post_projection_norm = LayerNorm(dim)
        self.gate_proj = nn.Linear(dim, context_dim, bias=bias)
        self.up_proj = nn.Linear(dim, context_dim, bias=bias)
        self.down_proj = nn.Linear(context_dim, dim, bias=bias)
        self.act1 = nn.GELU()
        self.act_fn = ACT2FN[hidden_act]

    def forward(
        self,
        hidden_state,
    ):
        hidden_state = self.proj(hidden_state)
        hidden_state = self.act1(self.post_projection_norm(hidden_state))
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))
