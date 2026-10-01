from typing import Any
import torch
from torch import Tensor
import torch.distributed as dist
import megatron.core.parallel_state as mpu
from megatron.core.transformer.dot_product_attention import DotProductAttention as MegatronDotProductAttention
from mindspeed.megatron_adaptor import get_mindspeed_args
from mindspeed.core.context_parallel.model_parallel_utils import get_context_parallel_group_for_hybrid_ulysses
from mindspeed.core.context_parallel.dot_product_attention import CPDotProductAttentionImpl
try:
    from mindspeed.core.context_parallel.ulysses_context_parallel.unaligned_cp.mapping import all_to_all
    native_all_to_all = False
except ImportError:
    from mindspeed_mm.models.common.communications import all_to_all
    native_all_to_all = True
from mindspeed_mm.models.common.communications import cal_split_sizes



def get_hetero_dotproductattention(config):
    if config.context_parallel_size > 1:
        return HeteroCPDotProductAttention
    else:
        return MegatronDotProductAttention


class HeteroCPDotProductAttention(CPDotProductAttentionImpl, MegatronDotProductAttention):

    def __init__(self, *args, scatter_idx=2, gather_idx=0, **kwargs):
        CPDotProductAttentionImpl.__init__(self, *args, **kwargs)
        config = self.config
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx

        if config.context_parallel_algo in ['hybrid_cp_algo', 'hybrid_adaptive_cp_algo', 'ulysses_cp_algo']:
            self.spg = mpu.get_context_parallel_group()
            if config.context_parallel_algo in ['hybrid_cp_algo', 'hybrid_adaptive_cp_algo']:
                self.spg = get_context_parallel_group_for_hybrid_ulysses()
        else:
            raise NotImplementedError(f'algorithm {config.context_parallel_algo} not implemented yet')
        self.DPA_forward = super().forward
        self.spg_world_size = dist.get_world_size(self.spg)


    def forward(self, query: Tensor, key: Tensor, value: Tensor, *args: Any, **kwargs: Any):
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

        if getattr(self.config, "use_remove_padding", False):
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
        scatter_sizes_query = cal_split_sizes(query.shape[self.scatter_idx], self.spg_world_size)
        scatter_sizes_key = cal_split_sizes(key.shape[self.scatter_idx], self.spg_world_size)
        scatter_sizes_value = cal_split_sizes(value.shape[self.scatter_idx], self.spg_world_size)

        gather_sizes = cal_split_sizes(act_seq_len, self.spg_world_size)
        if not native_all_to_all:
            query_layer = all_to_all(query, self.spg, self.scatter_idx, self.gather_idx, act_seq_len)
            key_layer = all_to_all(key, self.spg, self.scatter_idx, self.gather_idx, act_seq_len)
            value_layer = all_to_all(value, self.spg, self.scatter_idx, self.gather_idx, act_seq_len)
        else:
            query_layer = all_to_all(query, self.spg, self.scatter_idx, self.gather_idx, scatter_sizes_query, gather_sizes)
            key_layer = all_to_all(key, self.spg, self.scatter_idx, self.gather_idx, scatter_sizes_key, gather_sizes)
            value_layer = all_to_all(value, self.spg, self.scatter_idx, self.gather_idx, scatter_sizes_value, gather_sizes)

        context_layer = self.DPA_forward(query_layer, key_layer, value_layer, *args, **kwargs)
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
