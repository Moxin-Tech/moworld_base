# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import os
from typing import List, Optional, Union
from functools import wraps

import torch
from torch import distributed as dist
from torch import inf
try:
    from transformer_engine.pytorch.optimizers import (
        multi_tensor_applier,
        multi_tensor_l2norm,
        multi_tensor_scale,
    )

    l2_norm_impl = multi_tensor_l2norm
    multi_tensor_scale_impl = multi_tensor_scale
except ImportError:
    try:
        import amp_C
        from apex.multi_tensor_apply import multi_tensor_applier

        l2_norm_impl = amp_C.multi_tensor_l2norm
        multi_tensor_scale_impl = amp_C.multi_tensor_scale
    except ImportError:
        import warnings

        warnings.warn(
            f'Transformer Engine and Apex are not installed. '
            'Falling back to local implementations of multi_tensor_applier, '
            'multi_tensor_l2norm, and multi_tensor_scale'
        )

        from megatron.core.utils import (
            local_multi_tensor_applier,
            local_multi_tensor_l2_norm,
            local_multi_tensor_scale,
        )

        multi_tensor_applier = local_multi_tensor_applier
        l2_norm_impl = local_multi_tensor_l2_norm
        multi_tensor_scale_impl = local_multi_tensor_scale
from megatron.core import mpu
from megatron.core.utils import get_data_parallel_group_if_dtensor, to_local_if_dtensor

from mindspeed_mm.patchs.layerwise_disaggregated_training.parallel_state_patch import get_layerwise_disaggregated_training,\
    get_pipeline_model_parallel_group_for_vdp_cross_cloud_tp, get_pipeline_model_parallel_group_for_vdp_cross_edge_cloud, \
    get_vdp_size, is_vtp_enabled, vtp_hierarchical_barrier, vtp_allreduce, is_vdp_enabled


def _ldt_allreduce_model_parallel(tensor, op, group=None):
    """Allreduce on model_parallel_group, VTP-aware.

    When VTP is active, replaces flat cross-network allreduce with
    hierarchical allreduce (TP → PP → broadcast).
    """
    if is_vtp_enabled() and not is_vdp_enabled():
        vtp_allreduce(tensor, op=op)
    elif is_vdp_enabled():
        if op == torch.distributed.ReduceOp.SUM or group is not None:
            torch.distributed.all_reduce(tensor, op=op, group=group)
        else:
            # AR in TP group
            tp_group = mpu.get_tensor_model_parallel_group()
            torch.distributed.all_reduce(tensor, op=op, group=tp_group)

            # AllReduce on the group composed of all first TP rank
            vdp_cross_cloud_tp_group = get_pipeline_model_parallel_group_for_vdp_cross_cloud_tp()
            if vdp_cross_cloud_tp_group is not None:
                torch.distributed.all_reduce(tensor, op=op, group=vdp_cross_cloud_tp_group)

            vdp_cross_edge_cloud_group = get_pipeline_model_parallel_group_for_vdp_cross_edge_cloud()
            if vdp_cross_edge_cloud_group is not None:
                torch.distributed.all_reduce(tensor, op=op, group=vdp_cross_edge_cloud_group)

            vdp_cross_cloud_tp_group = get_pipeline_model_parallel_group_for_vdp_cross_cloud_tp()
            if vdp_cross_cloud_tp_group is not None:
                torch.distributed.all_reduce(tensor, op=op, group=vdp_cross_cloud_tp_group)

            # Broadcast to all ranks in TP group
            tp_ranks = torch.distributed.get_process_group_ranks(tp_group)
            torch.distributed.broadcast(tensor, src=tp_ranks[0], group=tp_group)
    else:
        torch.distributed.all_reduce(tensor, op=op, group=group)


def vtp_all_gather_into_tensor_wrapper(original_all_gather):
    """VTP-aware all_gather wrapper for timer statistics collection."""
    def wrapper(output_tensor, input_tensor, group=None, async_op=False):
        try:
            if is_vtp_enabled() and group is None:
                # Skip global all_gather in VTP mode - only rank0 needs timer stats
                return original_all_gather(output_tensor, input_tensor, group=group, async_op=async_op)
        except ImportError:
            pass
        return original_all_gather(output_tensor, input_tensor, group=group, async_op=async_op)
    return wrapper


def ldt_vdp_barrier_wrapper(fn):
    """
    This function wraps a function to add VDP barrier and VTP-aware hierarchical barrier.
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):

        if is_vdp_enabled():
            # VDP barriers
            if mpu.get_tensor_model_parallel_world_size() > 1: # 直接去掉，跑vtp有问题。
                fn(group=mpu.get_tensor_model_parallel_group())

            cross_cloud_tp_group = get_pipeline_model_parallel_group_for_vdp_cross_cloud_tp()
            if cross_cloud_tp_group is not None:
                fn(group=cross_cloud_tp_group)

            cross_edge_cloud_group = get_pipeline_model_parallel_group_for_vdp_cross_edge_cloud()
            if cross_edge_cloud_group is not None:
                fn(group=cross_edge_cloud_group)

            return None
        elif is_vtp_enabled() and kwargs.get('group') is None:
            vtp_hierarchical_barrier()
            return None
        else:
            return fn(*args, **kwargs)

    return wrapper


def ldt_reduce_max_stat_across_model_parallel_group(stat: float) -> float:
    """
    Ranks without an optimizer will have no grad_norm or num_zeros_in_grad stats.
    We need to ensure the logging and writer rank has those values.
    This function reduces a stat tensor across the model parallel group with VTP and VDP support.

    We use an all_reduce max since the values have already been summed across optimizer ranks where possible
    """
    if stat is None:
        stat = -1.0
    stat = torch.tensor([stat], dtype=torch.float32, device=torch.cuda.current_device())

    _ldt_allreduce_model_parallel(stat, op=torch.distributed.ReduceOp.MAX)

    if stat.item() == -1.0:
        return None
    else:
        return stat.item()


def ldt_logical_and_across_model_parallel_group(value: bool) -> bool:
    """
    This function gathers a bool value across the model parallel group with VTP and VDP support
    """
    if value is True:
        value = 1
    else:
        value = 0
    value = torch.tensor([value], dtype=torch.int, device=torch.cuda.current_device())

    _ldt_allreduce_model_parallel(value, torch.distributed.ReduceOp.MIN)

    return bool(value.item())


def ldt_get_grad_norm_fp32(
    grads_for_norm: Union[List[torch.Tensor], torch.Tensor],
    norm_type: Union[int, float] = 2,
    grad_stats_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
) -> float:
    """Calculate the norm of gradients in fp32 with VTP and VDP support.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters for VDP scenario
    and VTP-aware hierarchical allreduce.

    Arguments:
        grads_for_norm (Iterable[Tensor] or Tensor): an iterable of Tensors or a single
            Tensor that will be used for calculating the grad norm.
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        grad_stats_parallel_group (group): Process group for reducing the grad norms. This is
            generally the model-parallel group for non-distributed optimizers, and the entire
            world for the distributed optimizer.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """

    if isinstance(grads_for_norm, torch.Tensor):
        grads_for_norm = [grads_for_norm]

    data_parallel_group = None
    for grad in grads_for_norm:
        data_parallel_group = get_data_parallel_group_if_dtensor(grad, data_parallel_group)

    grads_for_norm = [to_local_if_dtensor(grad) for grad in grads_for_norm]

    # Norm parameters.
    norm_type = float(norm_type)
    total_norm = 0.0

    # Calculate norm.
    if norm_type == inf:
        total_norm = max(grad.abs().max() for grad in grads_for_norm)
        total_norm_cuda = torch.tensor([float(total_norm)], dtype=torch.float, device='cuda')
        # Take max across all data-parallel GPUs if using FSDP and then all model-parallel GPUs.
        if data_parallel_group:
            torch.distributed.all_reduce(
                total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=data_parallel_group
            )

        if not is_vdp_enabled() and is_vtp_enabled():
            # VTP-aware allreduce
            _ldt_allreduce_model_parallel(
                total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=grad_stats_parallel_group
            )
        else:
            torch.distributed.all_reduce(
                total_norm, op=torch.distributed.ReduceOp.SUM, group=grad_stats_parallel_group
            )

        total_norm = total_norm_cuda[0].item()

    else:
        if norm_type == 2.0:
            dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
            # Use apex's multi-tensor applier for efficiency reasons.
            # Multi-tensor applier takes a function and a list of list
            # and performs the operation on that list all in one kernel.
            if grads_for_norm:
                grad_norm, _ = multi_tensor_applier(
                    l2_norm_impl,
                    dummy_overflow_buf,
                    [grads_for_norm],
                    False,  # no per-parameter norm
                )
            else:
                grad_norm = torch.tensor([0], dtype=torch.float, device='cuda')
            # Since we will be summing across data parallel groups,
            # we need the pow(norm-type).
            # Virtual DP scenario, edge side average grad_norm across DP domains.
            if mpu.is_pipeline_first_stage(ignore_virtual=True):
                grad_norm = grad_norm / get_vdp_size()

            total_norm = grad_norm**norm_type

        else:
            for grad in grads_for_norm:
                grad_norm = torch.norm(grad, norm_type)

                # Virtual DP scenario, edge side average grad_norm across DP domains.
                if mpu.is_pipeline_first_stage(ignore_virtual=True):
                    grad_norm = grad_norm / get_vdp_size()

                total_norm += grad_norm**norm_type

        # Sum across all data-parallel GPUs if using FSDP and then all model-parallel GPUs.
        if data_parallel_group:
            torch.distributed.all_reduce(
                total_norm, op=torch.distributed.ReduceOp.SUM, group=data_parallel_group
            )

        if not is_vdp_enabled() and is_vtp_enabled():
            # VTP-aware allreduce
            _ldt_allreduce_model_parallel(
                total_norm, op=torch.distributed.ReduceOp.SUM, group=grad_stats_parallel_group
            )
        else:
            torch.distributed.all_reduce(
                total_norm, op=torch.distributed.ReduceOp.SUM, group=grad_stats_parallel_group
            )
        # VTP-aware allreduce
        total_norm = total_norm.item() ** (1.0 / norm_type)

    return total_norm


class VDPAllReduceManager:
    """
    Manager class for virtual DP all-reduce operations.
    """
    def __init__(self, enable_vdp: bool = False, vdp_role: str = "cloud"):
        self.enable_vdp = enable_vdp
        self.vdp_role = vdp_role

    def safe_all_reduce(self, tensor: torch.Tensor, group: Union[dist.ProcessGroup, List[dist.ProcessGroup]], op: dist.ReduceOp = dist.ReduceOp.SUM):
        """
        Perform an all-reduce operation on the tensor if virtual DP is enabled.
        support edge and cloud side.
        """
        if not self.enable_vdp:
            dist.all_reduce(tensor, op=op, group=group)
            return

        if self.vdp_role == "cloud":
            self._cloud_allreduce(tensor, op=op, group=group)
        else:
            if isinstance(group, list):
                self._edge_allreduce(tensor, group, op)
            else:
                dist.all_reduce(tensor, op=op, group=group)

    def _edge_allreduce(self, tensor: torch.Tensor, groups: List[dist.ProcessGroup], op: dist.ReduceOp):
        """
        edge side perform an all-reduce operation
        """
        for group in groups:
            dist.all_reduce(tensor, op=op, group=group)

    def _cloud_allreduce(self, tensor: torch.Tensor, group: dist.ProcessGroup, op: dist.ReduceOp):
        """
        cloud side perform an all-reduce operation
        """
        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)

        if world_size == 1:
            dist.all_reduce(tensor, op=op, group=group)
            return

        all_ranks = list(range(world_size))
        all_ranks.sort()
        for curr_rank in all_ranks:
            if curr_rank == rank:
                dist.all_reduce(tensor, op=op, group=group)

            dist.barrier(group=group)

    def safe_multi_allreduce(self, tensor: torch.Tensor, groups: Union[dist.ProcessGroup, List[dist.ProcessGroup]], ops: Optional[List[dist.ReduceOp]] = None):
        """
        Perform multi all-reduce operations
        """
        if isinstance(groups, list):
            groups_list = groups
        else:
            groups_list = [groups]

        if ops is None:
            ops = [dist.ReduceOp.SUM] * len(groups_list)

        if len(groups_list) != len(ops):
            raise ValueError("The length of groups_list and ops must be the same.")

        for group, op in zip(groups_list, ops):
            self.safe_all_reduce(tensor, group, op)


def get_vdp_manager():
    """
    Get the VDPAllReduceManager instance.
    """
    if int(os.environ.get('GROUP_RANK')) == 0 or int(os.environ.get('RANK')) == 0:
        vdp_role = 'edge'
    else:
        vdp_role = 'cloud'

    if get_layerwise_disaggregated_training():
        return VDPAllReduceManager(enable_vdp=True, vdp_role=vdp_role)
    else:
        return None
