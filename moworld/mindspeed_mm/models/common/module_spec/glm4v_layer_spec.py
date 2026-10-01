# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass, field
from typing import Dict, Union, Optional, Any

from torch import Tensor

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import make_viewless_tensor, deprecate_inference_params
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.extensions.transformer_engine import (
    TENorm,
)
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)
from mindspeed.core.megatron_basic.megatron_basic import PTNorm

from mindspeed_mm.models.vision.vision_encoders.glm4v_vl_vit_model import Glm4vSelfAttention, Glm4vVisionAttention
from mindspeed_mm.models.common.module_spec.llava_layer_spec import get_mlp_module_spec


@dataclass
class Glm4vTransformerLayerSubmodules:
    input_layernorm: Union[ModuleSpec, type] = IdentityOp
    self_attention: Union[ModuleSpec, type] = IdentityOp
    self_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # New for GLM4V
    post_self_attn_layernorm: Union[ModuleSpec, type] = IdentityOp

    pre_cross_attn_layernorm: Union[ModuleSpec, type] = IdentityOp
    cross_attention: Union[ModuleSpec, type] = IdentityOp
    cross_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    pre_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp
    mlp: Union[ModuleSpec, type] = IdentityOp
    mlp_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # New for GLM4V
    post_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp

    # Mapping for sharded tensor keys to be applied in `sharded_state_dict` method
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


class Glm4vTransformerLayer(TransformerLayer):
    def __init__(
        self,
        config: TransformerConfig,
        submodules: Glm4vTransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
    ):
        config.layernorm_epsilon = config.rms_norm_eps
        super().__init__(config=config, submodules=submodules, layer_number=layer_number)

        self.post_self_attn_layernorm = build_module(
            submodules.post_self_attn_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        self.post_mlp_layernorm = build_module(
            submodules.post_mlp_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[Any] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[Any] = None,
    ):
        inference_context = deprecate_inference_params(inference_context, inference_params)

        # Residual connection.
        residual = hidden_states

        # Optional Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
        )
        attention_output_with_bias = (self.post_self_attn_layernorm(attention_output_with_bias[0]), attention_output_with_bias[1])

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

        # Cross attention.
        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        # MLP.
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)
        mlp_output_with_bias = (self.post_mlp_layernorm(mlp_output_with_bias[0]), mlp_output_with_bias[1])

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.

        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        return output, context


def get_glm4v_layer_spec(config=None, *args, **kwargs) -> ModuleSpec:
    mlp = get_mlp_module_spec(use_te=False)
    return ModuleSpec(
        module=Glm4vTransformerLayer,
        submodules=Glm4vTransformerLayerSubmodules(
            input_layernorm=PTNorm,
            self_attention=ModuleSpec(
                module=Glm4vSelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ),
            ),
            post_self_attn_layernorm=PTNorm,
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=PTNorm,
            mlp=mlp,
            post_mlp_layernorm=PTNorm,
            mlp_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map={
                'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
            },
        ),
    )


def get_glm4v_vit_layer_spec(config=None, is_vit=True, *args, **kwargs) -> ModuleSpec:
    mlp = get_mlp_module_spec(use_te=False)
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=PTNorm,
            self_attention=ModuleSpec(
               module=Glm4vVisionAttention,
               params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    core_attention=DotProductAttention,
                    linear_qkv=IdentityOp,
                    linear_proj=IdentityOp,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=PTNorm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )
