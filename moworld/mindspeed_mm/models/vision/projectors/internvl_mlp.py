import torch
from torch import nn

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec

from mindspeed_mm.models.common.module import MultiModalModule


class InternVLMLP(MultiModalModule):
    def __init__(
        self,
        config: TransformerConfig,
        submodules: ModuleSpec,
    ):
        super().__init__(config=config)

        downsample_ratio = config.downsample_ratio
        vit_hidden_size = config.vit_hidden_size
        llm_hidden_size = config.llm_hidden_size

        use_norm = getattr(config, "norm", True)
        self.norm = nn.LayerNorm(vit_hidden_size * int(1 / downsample_ratio) ** 2) if use_norm else None
        self.linear_fc1 = nn.Linear(vit_hidden_size * int(1 / downsample_ratio) ** 2, llm_hidden_size)
        self.activation_func = nn.GELU()
        self.linear_fc2 = nn.Linear(llm_hidden_size, llm_hidden_size)

    def forward(
        self,
        hidden_state,
    ):
        if self.norm is not None:
            hidden_state = self.norm(hidden_state)
        hidden_state = self.linear_fc1(hidden_state)
        hidden_state = self.activation_func(hidden_state)
        hidden_state = self.linear_fc2(hidden_state)

        return hidden_state
