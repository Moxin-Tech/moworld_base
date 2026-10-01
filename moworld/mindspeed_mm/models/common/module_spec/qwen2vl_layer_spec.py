# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2023 Alibaba PAI and Nvidia Megatron-LM Team.

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
from megatron.training import get_args
from mindspeed_mm.models.common.module_spec.llava_layer_spec import get_mlp_module_spec
from mindspeed_mm.models.vision.vision_encoders.qwen2vl_vit_model import Qwen2vlVitSelfAttention, Qwen2vlSelfAttention, Qwen2_5VitDotProductAttention
from mindspeed_mm.patchs.canonical_layer_patch import (
    PatchSplitQKVSelfAttention,
    _patch_get_mlp_module_spec,
    PatchViTSelfAttention,
    SplitQKVSelfAttentionSubmodules
    )


def get_qwen2vl_llm_layer_spec(config=None, *args, **kwargs) -> ModuleSpec:

    if get_args().hetero_parallel:
        from mindspeed_mm.utils.hetero_utils.hetero_CP_utils import get_hetero_dotproductattention
        DOTPRODUCTATTENTION = get_hetero_dotproductattention(config)
    else:
        DOTPRODUCTATTENTION = DotProductAttention

    qk_layernorm = False
    canonical_model = getattr(config, 'canonical_model', False)
    if canonical_model:
        mlp = _patch_get_mlp_module_spec(use_te=False)
    else:
        mlp = _get_mlp_module_spec(use_te=False)
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=TENorm,
            self_attention=ModuleSpec(
                module=Qwen2vlSelfAttention if not canonical_model else PatchSplitQKVSelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DOTPRODUCTATTENTION,
                    linear_proj=RowParallelLinear,
                    q_layernorm=TENorm if qk_layernorm else IdentityOp,
                    k_layernorm=TENorm if qk_layernorm else IdentityOp,
                ) if not canonical_model else
                SplitQKVSelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    q_proj=ColumnParallelLinear,
                    k_proj=ColumnParallelLinear,
                    v_proj=ColumnParallelLinear,
                    core_attention=DOTPRODUCTATTENTION,
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


def get_qwen2vl_layer_spec(config=None, is_vit=True, *args, **kwargs) -> ModuleSpec:
    attn_mask_type = AttnMaskType.no_mask if is_vit else AttnMaskType.causal

    if get_args().hetero_parallel:
        from mindspeed_mm.utils.hetero_utils.hetero_CP_utils import get_hetero_dotproductattention
        DOTPRODUCTATTENTION = get_hetero_dotproductattention(config)
    else:
        DOTPRODUCTATTENTION = DotProductAttention

    canonical_model = getattr(config, 'canonical_model', False)
    if canonical_model:
        mlp = _patch_get_mlp_module_spec(use_te=False)
    else:
        mlp = _get_mlp_module_spec(use_te=False)
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=TENorm,
            self_attention=ModuleSpec(
                module=Qwen2vlVitSelfAttention if not canonical_model else PatchViTSelfAttention,
                params={
                    "attn_mask_type": attn_mask_type
                },
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DOTPRODUCTATTENTION,
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


def get_qwen2_5_vit_layer_spec(config=None, is_vit=True, *args, **kwargs) -> ModuleSpec:
    if hasattr(config, "use_vit_dp") and config.use_vit_dp:
        core_attention = Qwen2_5VitDotProductAttention
    else:
        core_attention = DotProductAttention

    attn_mask_type = AttnMaskType.no_mask if is_vit else AttnMaskType.causal

    mlp = get_mlp_module_spec(use_te=False)
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=TENorm,
            self_attention=ModuleSpec(
                module=Qwen2vlVitSelfAttention,
                params={
                    "attn_mask_type": attn_mask_type
                },
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=core_attention,
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
