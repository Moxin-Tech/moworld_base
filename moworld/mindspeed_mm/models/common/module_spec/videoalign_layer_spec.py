# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025 Alibaba PAI and Nvidia Megatron-LM Team.

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
from megatron.core.models.gpt.gpt_layer_specs import _get_mlp_module_spec

from mindspeed_mm.models.common.module_spec.llava_layer_spec import get_mlp_module_spec
from mindspeed_mm.models.vision.vision_encoders.qwen2vl_vit_model import Qwen2vlVitSelfAttention, Qwen2vlSelfAttention

from mindspeed_mm.patchs.canonical_layer_patch import (
    PatchSplitQKVSelfAttention,
    _patch_get_mlp_module_spec,
    PatchViTSelfAttention,
    SplitQKVSelfAttentionSubmodules
    )


def get_videoalign_llm_layer_spec(config=None, *args, **kwargs) -> ModuleSpec:
    qk_layernorm = False

    mlp = _patch_get_mlp_module_spec(use_te=False)
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=TENorm,
            self_attention=ModuleSpec(
                module=PatchSplitQKVSelfAttention,
                params={"attn_mask_type": AttnMaskType.padding_causal},
                submodules=SplitQKVSelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    q_proj=ColumnParallelLinear,
                    k_proj=ColumnParallelLinear,
                    v_proj=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=TENorm if qk_layernorm else IdentityOp,
                    k_layernorm=TENorm if qk_layernorm else IdentityOp,
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


def get_videoalign_layer_spec(config=None, is_vit=True, *args, **kwargs) -> ModuleSpec:
    attn_mask_type = AttnMaskType.no_mask if is_vit else AttnMaskType.causal

    mlp = get_mlp_module_spec(use_te=False)
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=TENorm,
            self_attention=ModuleSpec(
                module=PatchViTSelfAttention,
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
        ),
    )
