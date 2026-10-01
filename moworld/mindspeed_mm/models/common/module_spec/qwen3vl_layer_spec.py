# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from typing import Union
from dataclasses import dataclass
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.extensions.transformer_engine import (
    TENorm,
    TEColumnParallelGroupedLinear,
    TELayerNormColumnParallelLinear,
    TERowParallelGroupedLinear,
    TERowParallelLinear
)
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.experts import SequentialMLP
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.transformer_block import TENorm
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)

from mindspeed_mm.models.vision.vision_encoders.qwen2vl_vit_model import Qwen2vlSelfAttention, Qwen2vlVitSelfAttention
from mindspeed_mm.models.vision.vision_encoders.vision_transformer_block import Qwen3VLVisionTransformerLayer
try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TEDotProductAttention
    )
except ImportError:
    pass


def get_mlp_module_spec(
    use_te=True, num_experts=None, moe_grouped_gemm=False, use_shared_experts=None
) -> ModuleSpec:
    if num_experts is None:
        # Dense MLP w/ or w/o TE modules.
        return ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=TELayerNormColumnParallelLinear if use_te else ColumnParallelLinear,
                linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
            ),
        )
    else:
        # Mixture of experts with modules in megatron core.
        if use_te and moe_grouped_gemm:
            linear_fc1 = TEColumnParallelGroupedLinear
            linear_fc2 = TERowParallelGroupedLinear
        else:
            linear_fc1 = ColumnParallelLinear
            linear_fc2 = RowParallelLinear

        use_te_grouped_gemm = use_te and TEColumnParallelGroupedLinear is not None

        if use_shared_experts is not None:
            shared_experts = ModuleSpec(module=SharedExpertMLP,
                                        params={"gate": False},
                                        submodules=MLPSubmodules(
                                            linear_fc1=linear_fc1,
                                            linear_fc2=linear_fc2,)
                                        )
        else:
            shared_experts = None

        return ModuleSpec(
            module=MoELayer,
            submodules=(
                MoESubmodules(
                    experts=ModuleSpec(
                        module=SequentialMLP,
                        submodules=MLPSubmodules(
                            linear_fc1=linear_fc1,
                            linear_fc2=linear_fc2,
                        )
                    ),
                    shared_experts=shared_experts
                )
                if not moe_grouped_gemm or use_te_grouped_gemm
                else None
            ),
        )


def get_qwen3vl_llm_layer_local_spec(config=None, *args, **kwargs) -> ModuleSpec:
    if config.num_moe_experts:
        mlp = get_mlp_module_spec(use_te=config.use_te, num_experts=config.num_moe_experts, moe_grouped_gemm=config.moe_grouped_gemm)
    else:
        mlp = get_mlp_module_spec(use_te=config.use_te)
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=TENorm,
            self_attention=ModuleSpec(
                module=Qwen2vlSelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear if not config.use_te else TEColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear if not config.use_te else TERowParallelLinear,
                    q_layernorm=TENorm if config.qk_layernorm else IdentityOp,
                    k_layernorm=TENorm if config.qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=TENorm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map={
                'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
            },
        ),
    )


def get_qwen3vl_layer_spec(config=None, is_vit=True, *args, **kwargs) -> ModuleSpec:
    attn_mask_type = AttnMaskType.no_mask if is_vit else AttnMaskType.causal
    mlp = get_mlp_module_spec(use_te=False)
    return ModuleSpec(
        module=Qwen3VLVisionTransformerLayer,
        submodules=Qwen3VLVisonTransformerLayerSubmodules(
            input_layernorm=TENorm,
            self_attention=ModuleSpec(
                module=Qwen2vlVitSelfAttention,
                params={
                    "attn_mask_type": attn_mask_type
                },
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=TENorm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
            deepstack_layer=mlp
        ),
    )


@dataclass
class Qwen3VLVisonTransformerLayerSubmodules(TransformerLayerSubmodules):
    deepstack_layer: Union[ModuleSpec, type] = IdentityOp
