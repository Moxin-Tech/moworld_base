# coding=utf-8
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Tuple, List

import torch
from torch import Tensor
from torch.nn import Module
import torch.distributed as dist
from megatron.core import mpu
from megatron.training import get_args
from mindspeed.megatron_adaptor import get_mindspeed_args
from mindspeed.patch_utils import MindSpeedPatchesManager as pm

try:
    from mindspeed.core.context_parallel.ulysses_context_parallel.unaligned_cp.mapping import all_to_all
    native_all_to_all = False
except ImportError:
    from mindspeed_mm.models.common.communications import all_to_all
    native_all_to_all = True

from mindspeed_mm.models.common.communications import cal_split_sizes


class UlyssesContextAttention(torch.nn.Module):
    """Initialization.
    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """
    def __init__(
            self,
            local_attention: Module,
            sequence_process_group: torch.distributed.ProcessGroup,
            scatter_idx: int = 2,
            gather_idx: int = 0,
    ) -> None:
        super(UlyssesContextAttention, self).__init__()
        self.local_attn = local_attention
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx

    def forward(self, query: Tensor, key: Tensor, value: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """ forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """

        attention_mask = args[0]
        packed_seq_params = kwargs['packed_seq_params']
        if attention_mask is None:
            act_seq_len = packed_seq_params.cu_seqlens_q[-1]
        else:
            act_seq_len = attention_mask.shape[-1]

        if getattr(self.local_attn.config, "use_remove_padding", False):
            from mindspeed.utils import get_actual_seq_len
            act_seq_len = get_actual_seq_len()[0]
            attention_mask = torch.triu(
                torch.ones([2048, 2048], dtype=torch.bool, device=query.device), diagonal=1
            )
            args_list = list(args)
            args_list[0] = attention_mask
            args = tuple(args_list)

        if packed_seq_params is not None:
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
        scatter_sizes_query = cal_split_sizes(query.shape[self.scatter_idx], dist.get_world_size(self.spg))
        scatter_sizes_key = cal_split_sizes(key.shape[self.scatter_idx], dist.get_world_size(self.spg))
        scatter_sizes_value = cal_split_sizes(value.shape[self.scatter_idx], dist.get_world_size(self.spg))

        gather_sizes = cal_split_sizes(act_seq_len, dist.get_world_size(self.spg))
        if not native_all_to_all:
            query_layer = all_to_all(query, self.spg, self.scatter_idx, self.gather_idx, act_seq_len)
            key_layer = all_to_all(key, self.spg, self.scatter_idx, self.gather_idx, act_seq_len)
            value_layer = all_to_all(value, self.spg, self.scatter_idx, self.gather_idx, act_seq_len)
        else:
            query_layer = all_to_all(query, self.spg, self.scatter_idx, self.gather_idx, scatter_sizes_query, gather_sizes)
            key_layer = all_to_all(key, self.spg, self.scatter_idx, self.gather_idx, scatter_sizes_key, gather_sizes)
            value_layer = all_to_all(value, self.spg, self.scatter_idx, self.gather_idx, scatter_sizes_value, gather_sizes)

        context_layer = self.local_attn(query_layer, key_layer, value_layer, *args, **kwargs)
        if get_mindspeed_args().context_parallel_algo == "hybrid_cp_algo" and context_layer.dim() == 3:
            context_layer = context_layer.unsqueeze(1)
        else:
            context_shape = context_layer.shape
            context_layer = context_layer.reshape(context_shape[0], context_shape[1],
                                                  scatter_sizes_query[dist.get_rank(self.spg)], -1)

        if not native_all_to_all:
            output = all_to_all(context_layer, self.spg, self.gather_idx, self.scatter_idx, query.shape[self.scatter_idx])
        else:
            output = all_to_all(context_layer, self.spg, self.gather_idx, self.scatter_idx, gather_sizes, scatter_sizes_query)
        output = output.reshape(output.shape[0], output.shape[1], -1)
        if packed_seq_params is not None:
            output = output.squeeze(1)

        # out e.g., [s/p::h]
        return output


def get_rotary_seq_len(
    self,
    inference_params,
    transformer,
    transformer_input,
    transformer_config
) -> float:
    if inference_params is not None:
        rotary_seq_len = inference_params.max_sequence_length
    else:
        if transformer.input_tensor is not None:
            rotary_seq_len = transformer.input_tensor.size(0)
        else:
            rotary_seq_len = transformer_input.size(0)

        if hasattr(get_args(), "context_parallel_size"):
            if int(get_args().context_parallel_size) > 1:
                rotary_seq_len = torch.Tensor([rotary_seq_len]).cuda().int()
                torch.distributed.all_reduce(rotary_seq_len, op=torch.distributed.ReduceOp.SUM, group=mpu.get_context_parallel_group())
                rotary_seq_len = rotary_seq_len.item()

    return rotary_seq_len


mindspeed_args = get_mindspeed_args()
if hasattr(mindspeed_args, 'context_parallel_algo') and hasattr(mindspeed_args, 'context_parallel_size'):
    if mindspeed_args.context_parallel_algo in ["ulysses_cp_algo", "hybrid_cp_algo"] and int(mindspeed_args.context_parallel_size) > 1:
        pm.register_patch('mindspeed.core.context_parallel.ulysses_context_parallel.ulysses_context_parallel.UlyssesContextAttention',
                        UlyssesContextAttention, force_patch=True)
        pm.register_patch('megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.get_rotary_seq_len',
                        get_rotary_seq_len, force_patch=True)
        pm.apply_patches()
