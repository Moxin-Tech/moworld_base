from typing import Tuple, Optional, Union, List

import torch
from torch import Tensor
from torch.distributed import ProcessGroup

from megatron.core import mpu
from megatron.training import get_args
from megatron.core.packed_seq_params import PackedSeqParams
from mindspeed.core.context_parallel.ulysses_context_parallel.unaligned_cp.mapping import all_to_all
from mindspeed.core.context_parallel.model_parallel_utils import (
    get_context_parallel_group_for_hybrid_ulysses,
    get_context_parallel_group_for_hybrid_ring,
    get_context_parallel_for_hybrid_ring_world_size,
    get_context_parallel_for_hybrid_ulysses_world_size,
    get_context_parallel_for_hybrid_ring_global_ranks,
    get_context_parallel_for_hybrid_ring_rank
)

from mindspeed_mm.models.common.communications import gather_sequence_chunks_to_packed_tensor, split_each_sequence_in_packed_tensor
from mindspeed_mm.models.common.communications import cal_split_sizes, cal_split_sizes_multi, gather_forward_split_backward, split_forward_gather_backward

_TOTAL_SEQ_LEN = None
_VISUAL_SEQ_LEN = None
_VISUAL_PER_SEQ_LEN = None
_AUDIO_SEQ_LEN = None


def get_seq_len(des: str = None) -> Optional[Union[int, List[int]]]:
    des_to_var = {
        "total": _TOTAL_SEQ_LEN,
        "visual": _VISUAL_SEQ_LEN,
        "per_visual": _VISUAL_PER_SEQ_LEN,
        "audio": _AUDIO_SEQ_LEN
    }
    return des_to_var[des]


def set_seq_len(des: str = None, seq_len: Optional[Union[int, List[int]]] = None) -> None:
    des_to_var_name = {
        "total": "_TOTAL_SEQ_LEN",
        "visual": "_VISUAL_SEQ_LEN",
        "per_visual": "_VISUAL_PER_SEQ_LEN",
        "audio": "_AUDIO_SEQ_LEN"
    }
    global _TOTAL_SEQ_LEN, _VISUAL_SEQ_LEN, _VISUAL_PER_SEQ_LEN, _AUDIO_SEQ_LEN
    var_name = des_to_var_name[des]
    globals()[var_name] = seq_len


def gather_seq_scatter_heads(
    input_tensor: Tensor,
    seq_dim: int,
    head_dim: int,
    gather_size: int,
    group: ProcessGroup = None
) -> Tensor:
    group = mpu.get_context_parallel_group() if group is None else group
    if not group:
        return input_tensor

    return all_to_all(input_tensor, group, scatter_dim=head_dim, gather_dim=seq_dim, gather_size=gather_size)


def gather_heads_scatter_seq(
    input_tensor: Tensor,
    head_dim: int,
    seq_dim: int,
    gather_size: int,
    group: ProcessGroup = None
) -> Tensor:
    group = mpu.get_context_parallel_group() if group is None else group
    if not group:
        return input_tensor

    return all_to_all(input_tensor, group, scatter_dim=seq_dim, gather_dim=head_dim, gather_size=gather_size)


def gather_seq_scatter_heads_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_dim: int, head_dim: int, gather_size: int, group: ProcessGroup = None):
    q = gather_seq_scatter_heads(q, seq_dim, head_dim, gather_size, group)
    k = gather_seq_scatter_heads(k, seq_dim, head_dim, gather_size, group)
    v = gather_seq_scatter_heads(v, seq_dim, head_dim, gather_size, group)
    return q, k, v


def gather_visual_seqs_with_cp(
    x: torch.Tensor,  # Concatenated sequences: s1+s2+s3+...
    dim: int = 0
):
    """
    Gather visual sequences across context parallel (CP) ranks during the forward pass,
    and split gradients back during the backward pass.

    This function supports multiple CP strategies:
      - **Ulysses CP**: All-gather full sequence using precomputed per-rank sequence lengths.
      - **Megatron CP**: Reconstruct packed tensor from sequence chunks distributed across CP ranks.
      - **Hybrid CP**: First gather within Ulysses subgroups, then across ring-based CP groups.
    """
    megatron_args = get_args()
    if megatron_args.context_parallel_algo == "ulysses_cp_algo":
        gather_sizes = cal_split_sizes(get_seq_len("visual"), mpu.get_context_parallel_world_size())
        x = gather_forward_split_backward(
            x,
            mpu.get_context_parallel_group(),
            dim=dim,
            grad_scale="up",
            gather_sizes=gather_sizes
        )

    elif megatron_args.context_parallel_algo == "megatron_cp_algo":
        all_split_sizes_tensor = cal_split_sizes_multi(get_seq_len("per_visual"), mpu.get_context_parallel_world_size())
        x = gather_sequence_chunks_to_packed_tensor(
            x,
            all_split_sizes_tensor,
            mpu.get_context_parallel_group(),
            dim=dim,
        )
    elif megatron_args.context_parallel_algo == "hybrid_cp_algo":
        # Step 1: Gather within Ulysses subgroups (inner CP group)
        # First, compute how visual tokens are distributed across ring CP ranks
        all_split_sizes_tensor = cal_split_sizes_multi(get_seq_len("per_visual"), get_context_parallel_for_hybrid_ring_world_size())
        gather_sizes = cal_split_sizes(all_split_sizes_tensor[get_context_parallel_for_hybrid_ring_rank()].sum(), get_context_parallel_for_hybrid_ulysses_world_size())
        x = gather_forward_split_backward(
            x,
            get_context_parallel_group_for_hybrid_ulysses(),
            dim=dim,
            grad_scale="up",
            gather_sizes=gather_sizes
        )
        # Step 2: Gather across ring CP ranks
        x = gather_sequence_chunks_to_packed_tensor(
            x,
            all_split_sizes_tensor,
            get_context_parallel_group_for_hybrid_ring(),
            dim=dim,
        )
    else:
        raise NotImplementedError(f"Only support `ulysses_cp_algo`,`megatron_cp_algo`,`hybrid_cp_algo`, but got {megatron_args.context_parallel_algo}")

    return x


def split_visual_seqs_with_cp(
    x: torch.Tensor,
    dim: int = 0
):
    """
    Split visual sequences across context parallel (CP) ranks during the forward pass,
    and gather full gradients during the backward pass.

    This function supports three CP strategies:
      - **Ulysses CP**: Splits the entire packed sequence uniformly (or near-uniformly) across all CP ranks.
      - **Megatron CP (Ring-style)**: Splits *each individual sample sequence* (e.g., image tokens) across CP ranks,
        then concatenates the resulting shards to form a new packed tensor.
      - **Hybrid CP**: First applies ring-based splitting per sample (Megatron-style), then further splits the result
        using Ulysses within each ring subgroup.
    Args:
        x: Concatenated sequences: s1+s2+s3+...
    """
    args = get_args()
    if args.context_parallel_algo == "ulysses_cp_algo":
        seq_len = get_seq_len("visual")
        split_gather_sizes = cal_split_sizes(seq_len, mpu.get_context_parallel_world_size())
        x = split_forward_gather_backward(
            x,
            mpu.get_context_parallel_group(),
            dim=dim,
            split_sizes=split_gather_sizes
        )# [s1+s2+s3+..., h]
    elif args.context_parallel_algo == "megatron_cp_algo":
        sequence_lengths = get_seq_len("per_visual")
        x = split_each_sequence_in_packed_tensor(
            x,
            mpu.get_context_parallel_group(),
            sequence_lengths,
            dim=dim
        )
    elif args.context_parallel_algo == "hybrid_cp_algo":
        sequence_lengths = get_seq_len("per_visual")
        # Step 1: Apply ring-based (Megatron-style) splitting per sample
        x = split_each_sequence_in_packed_tensor(
            x,
            get_context_parallel_group_for_hybrid_ring(),
            sequence_lengths,
            dim=dim
        )
        # Step 2: Further split the resulting packed shard using Ulysses within the ring subgroup
        split_gather_sizes = cal_split_sizes(x.shape[dim], get_context_parallel_for_hybrid_ulysses_world_size())
        x = split_forward_gather_backward(
            x,
            get_context_parallel_group_for_hybrid_ulysses(),
            dim=dim,
            split_sizes=split_gather_sizes
        )  # [s1+s2+s3+..., h]
    else:
        raise NotImplementedError(f"Only support `ulysses_cp_algo`,`megatron_cp_algo`,`hybrid_cp_algo`, but got {args.context_parallel_algo}")

    return x


def split_audio_seqs_with_cp(
    x: torch.Tensor,
    dim: int = 0
):
    """
    Split audio sequences across context parallel (CP) ranks during the forward pass,
    and gather full gradients during the backward pass.

    This function only supports three CP strategies now:
      - **Ulysses CP**: Splits the entire packed sequence uniformly (or near-uniformly) across all CP ranks.
    Args:
        x: Concatenated sequences: s1+s2+s3+...
    """
    args = get_args()
    if args.context_parallel_algo == "ulysses_cp_algo":
        seq_len = get_seq_len("audio")
        split_gather_sizes = cal_split_sizes(seq_len, mpu.get_context_parallel_world_size())
        x = split_forward_gather_backward(
            x,
            mpu.get_context_parallel_group(),
            dim=dim,
            split_sizes=split_gather_sizes
        )# [s1+s2+s3+..., h]
    else:
        raise NotImplementedError(f"Only support `ulysses_cp_algo`, but got {args.context_parallel_algo}")

    return x
