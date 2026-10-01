# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2023 Alibaba PAI and Nvidia Megatron-LM Team.

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer import ModuleSpec, TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.training import get_args
from mindspeed_mm.models.audio.omni_audio_encoder import QwenOmniAudioSelfAttention
from mindspeed_mm.models.common.module_spec.llava_layer_spec import get_mlp_module_spec


def get_qwen_omni_audio_layer_spec(config=None, is_vit=True, *args, **kwargs) -> ModuleSpec:
    attn_mask_type = AttnMaskType.no_mask if is_vit else AttnMaskType.causal

    if get_args().hetero_parallel:
        from mindspeed_mm.utils.hetero_utils.hetero_CP_utils import get_hetero_dotproductattention
        DOTPRODUCTATTENTION = get_hetero_dotproductattention(config)
    else:
        DOTPRODUCTATTENTION = DotProductAttention

    mlp = get_mlp_module_spec(use_te=False)
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=TENorm,
            self_attention=ModuleSpec(
                module=QwenOmniAudioSelfAttention,
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
