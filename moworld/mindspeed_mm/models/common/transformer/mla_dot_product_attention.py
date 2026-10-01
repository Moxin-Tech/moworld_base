# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import math
from megatron.training import get_args
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core import parallel_state
from megatron.core.utils import divide


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class MlaDotProductAttention(DotProductAttention):
    """
    A special type of Dot Product Attention based on DotProductAttention.
    """

    def __init__(
            self,
            config: TransformerConfig,
            layer_number: int,
            attn_mask_type: AttnMaskType,
            attention_type: str,
            attention_dropout: float = None,
            cp_comm_type=None,
            softmax_scale=None
    ):
        super().__init__(
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            attention_dropout=attention_dropout,
            cp_comm_type=cp_comm_type,
            softmax_scale=softmax_scale
        )
        args = get_args()

        self.scale_mask_softmax.scale = True
        projection_size = self.config.kv_channels * self.config.num_attention_heads

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = divide(projection_size, world_size)
        self.q_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.softmax_scale = self.q_head_dim ** (-0.5)

        if args.rope_scaling_type is not None:
            mscale_all_dim = args.rope_scaling_mscale_all_dim if args.rope_scaling_mscale_all_dim else 0
            scaling_factor = args.rope_scaling_factor

            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

        self.norm_factor = 1.0 / self.softmax_scale
