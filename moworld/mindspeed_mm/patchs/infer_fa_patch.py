import math
from functools import wraps
from typing import Optional

import torch
from torch import Tensor
import torch_npu

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.training import get_args

try:
    from einops import rearrange
except ImportError:
    rearrange = None


def dot_product_attention_forward_infer_wrapper(fn):
    @wraps(fn)
    def wrapper(self, query, key, value, attention_mask, **kwargs):
        if not hasattr(get_args().mm.model, "generation_config"):
            raise AssertionError("This infer fa patch is only available for inference.")
        if not getattr(get_args().mm.model.generation_config, "kv_cache", False):
            raise AssertionError("Inference fa is only available when kv_cache is True.")
        if get_args().use_flash_attn and getattr(self.config, "use_infer_fa", False):
            return dot_product_attention_forward_infer(self, query, key, value, attention_mask, **kwargs)
        return fn(self, query, key, value, attention_mask, **kwargs)

    return wrapper


def dot_product_attention_forward_infer(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType = None,
        attention_bias: Tensor = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
):
    bsz = query.shape[1]
    query = query.transpose(0, 1).contiguous()  # [b s h d]
    key = key.transpose(0, 1).contiguous()
    value = value.transpose(0, 1).contiguous()
    if query.shape[1] == 1:
        attention_mask_npu = None
    else:
        attention_mask_npu = torch.triu(
            torch.ones([query.shape[1], key.shape[1]], dtype=torch.bool, device=query.device), diagonal=1)

    attn_output = torch_npu.npu_fused_infer_attention_score(
            query, key, value,
            pse_shift=None,
            atten_mask=attention_mask_npu,
            actual_seq_lengths=[query.shape[1]],
            actual_seq_lengths_kv=[key.shape[1]],
            num_heads=query.shape[2],
            num_key_value_heads=key.shape[2],
            scale=1.0 / math.sqrt(query.shape[-1]),
            input_layout="BSND",
    )[0]
    attn_output = rearrange(attn_output, 'b s h d -> s b (h d)', s=query.shape[1], b=bsz)
    return attn_output
