# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from typing import Optional

from torch import nn

from megatron.legacy.model.rms_norm import RMSNorm

from megatron.core.transformer.enums import AttnMaskType, LayerType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
    get_transformer_layer_offset,
)

from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.extensions.transformer_engine import TENorm, TERowParallelLinear
from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec
from megatron.core.transformer.transformer_block import TENorm
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TEDotProductAttention
    )
except ImportError:
    pass

from mindspeed_mm.mcore.models.qwen3_5.modules import Qwen3_5SelfAttention, PatchMergerSubmodules, Qwen3_5SelfAttentionSubmodules
from mindspeed_mm.mcore.models.qwen3_5.gated_delta_net import GatedDeltaNet, GatedDeltaNetSubmodules, Qwen3_5GatedRMSNorm


class PTNorm:
    """
    A conditional wrapper to initialize an instance of Transformer-Engine's
    `LayerNorm` or `RMSNorm` based on input
    """

    def __new__(
            cls, config: TransformerConfig, hidden_size: int, eps: float = 1e-5,
    ):
        if config.normalization == "LayerNorm":
            instance = nn.LayerNorm(
                normalized_shape=hidden_size,
                eps=eps,
            )
        elif config.normalization == "RMSNorm":
            instance = RMSNorm(
                dim=hidden_size,
                eps=eps,
                sequence_parallel=config.sequence_parallel,
            )
        else:
            raise Exception('Only LayerNorm and RMSNorm are curently supported')

        return instance


def get_gated_delta_net_module_spec(config: TransformerConfig) -> ModuleSpec:
    """Build module spec for GatedDeltaNet attention."""
    spec = ModuleSpec(
        module=GatedDeltaNet,
        submodules=GatedDeltaNetSubmodules(
            in_proj=TEColumnParallelLinear if config.use_te else ColumnParallelLinear,
            out_norm=Qwen3_5GatedRMSNorm,
            out_proj=TERowParallelLinear if config.use_te else TERowParallelLinear,
        )
    )
    return spec


def get_self_attention_module_spec(config: TransformerConfig) -> ModuleSpec:
    """Get self-attention module spec."""
    return ModuleSpec(
        module=Qwen3_5SelfAttention,
        params={"attn_mask_type": AttnMaskType.causal, "attention_type": "self"},
        submodules=Qwen3_5SelfAttentionSubmodules(
            linear_qkv=TEColumnParallelLinear if config.use_te else ColumnParallelLinear,
            core_attention=DotProductAttention,
            linear_proj=RowParallelLinear,
            q_layernorm=PTNorm if config.qk_layernorm else IdentityOp,
            k_layernorm=PTNorm if config.qk_layernorm else IdentityOp,
        )
    )


def get_qwen3_5_text_block_spec(
    config: TransformerConfig, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
) -> TransformerBlockSubmodules:
    """Build transformer block spec with experimental attention variants (e.g., linear attention).

    This function constructs a heterogeneous transformer block that supports mixing different
    attention mechanisms (experimental vs standard) and MLP types (MoE vs dense) across layers.
    **Note that, this API is a experimental API in the short term, and might be deprecated in the
    future. In the long run, we will move to a new design that better support hybrid models.**

    Key Design:
        1. Attention and MLP patterns: The attention pattern and MLP pattern are orthogonal
           and determined independently. This allows flexible combinations (e.g., linear attention
           with MoE, or standard attention with dense MLP).
           - Attention pattern: derived from `config.linear_attention_freq` or
             `config.experimental_attention_variant`.
           - MLP pattern: derived from `config.moe_layer_freq`.

        2. Per-Layer Spec Construction: Iterates through layers, constructing transformer
           layer specs based on attention and MLP patterns.

        3. Pipeline Slicing: Extracts layer specs for the current pipeline stage.

    Args:
        config: Transformer configuration containing model hyperparameters and feature flags.
        vp_stage: Virtual pipeline stage index for interleaved pipeline parallelism.
        pp_rank: Pipeline model parallel rank.

    Returns:
        TransformerBlockSubmodules containing per-layer specs and final layer norm.

    Note:
        Currently only supports transformer_engine backend. Kitchen backend can be used as a
        wrapper with TE fallback for unsupported operations.
    """

    layer_specs = []
    for layer_number in range(config.num_layers):
        attention = (
            get_self_attention_module_spec(config)
            if (layer_number + 1) % config.linear_attention_freq == 0
            else get_gated_delta_net_module_spec(config)
        )
        mlp = get_mlp_module_spec(
            use_te=config.use_te,
            num_experts=config.num_moe_experts,
            moe_grouped_gemm=config.moe_grouped_gemm
        )

        layer_specs.append(
            ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=PTNorm,
                    self_attention=attention,
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=PTNorm,
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                ),
            )
        )

    # Slice the layer specs to only include the layers that are built in this pipeline stage.
    if config.pipeline_model_parallel_layout is not None:
        local_layer_ids = config.pipeline_model_parallel_layout.get_layer_id_list(
            layer_type=LayerType.decoder, vp_stage=vp_stage, pp_rank=pp_rank
        )
    else:
        # offset = get_transformer_layer_offset(config, vp_stage=vp_stage, pp_rank=pp_rank)
        # num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage, pp_rank=pp_rank)
        offset = get_transformer_layer_offset(config)
        num_layers_to_build = get_num_layers_to_build(config)
        local_layer_ids = range(offset, offset + num_layers_to_build)

    layer_specs = [layer_specs[layer_id] for layer_id in local_layer_ids]

    return TransformerBlockSubmodules(layer_specs=layer_specs, layer_norm=PTNorm)


def get_qwe3_5_vit_layer_local_spec(config=None) -> ModuleSpec:
    '''
    Returns ViT layer spec
    '''
    mlp = get_mlp_module_spec(use_te=config.use_te)
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=Qwen3_5SelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask, "attention_type": "self"},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear if not config.use_te else TEColumnParallelLinear,
                    core_attention=DotProductAttention if not config.use_te else TEDotProductAttention,
                    linear_proj=RowParallelLinear if not config.use_te else TERowParallelLinear,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )


def get_vision_patch_merger_spec(config=None) -> ModuleSpec:
    return PatchMergerSubmodules(
        patch_norm=TENorm,
        linear_fc1=ColumnParallelLinear if not config.use_te else TEColumnParallelLinear,
        linear_fc2=RowParallelLinear if not config.use_te else TERowParallelLinear,
    )
