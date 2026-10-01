# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

"""
MultiHeadLatent Layer Specification, which is mainly for DeepseekVL.
"""

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer import ModuleSpec, TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.extensions.transformer_engine import TENorm
from mindspeed_mm.models.common.transformer.multi_head_latent_attention import (
    MLASelfAttentionSubmodules,
    MLASelfAttentionWithMMSplitSubmodules,
    MultiHeadLatentAttention,
    LinearNoTP,
)
from mindspeed_mm.models.common.transformer.mla_dot_product_attention import MlaDotProductAttention


def get_deepseekvl_model_spec(config, **kwargs):
    qk_layernorm = config.qk_layernorm
    mla_mm_split = config.mla_mm_split
    layer_spec = ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=TENorm,
            self_attention=ModuleSpec(
                module=MultiHeadLatentAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=MLASelfAttentionSubmodules(
                    linear_qkv=LinearNoTP,
                    core_attention=MlaDotProductAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=TENorm if qk_layernorm else IdentityOp,
                    k_layernorm=TENorm if qk_layernorm else IdentityOp,
                    linear_qb=ColumnParallelLinear,
                    linear_kvb=ColumnParallelLinear,
                )
                if not mla_mm_split
                else MLASelfAttentionWithMMSplitSubmodules(
                    linear_qkv=LinearNoTP,
                    core_attention=MlaDotProductAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=TENorm if qk_layernorm else IdentityOp,
                    k_layernorm=TENorm if qk_layernorm else IdentityOp,
                    linear_qk_nope=ColumnParallelLinear,
                    linear_qk_rope=ColumnParallelLinear,
                    linear_kv_nope=ColumnParallelLinear,
                    linear_v=ColumnParallelLinear,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=TENorm,
            # different mlp spec varied from different layers.
            # So the real deepseek_mlp_spec would be defined in build_layer of Transformer Block
            mlp=ModuleSpec,
            mlp_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map={
                "input_layernorm.": "self_attention.linear_qkv.layer_norm_",
                "pre_mlp_layernorm.": "mlp.linear_fc1.layer_norm_",
            },
        ),
    )

    return layer_spec
