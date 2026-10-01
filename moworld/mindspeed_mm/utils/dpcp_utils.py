
import os
import importlib
import copy
from einops import rearrange
import torch
import torch_npu
import torch.distributed
import numpy as np
import megatron.core.parallel_state as global_mpu
from megatron.core.num_microbatches_calculator import reconfigure_num_microbatches_calculator
from megatron.core import mpu
from megatron.training import get_args, print_rank_0
from mindspeed_mm.data.data_utils.constants import (
    VIDEO,
    PROMPT_IDS,
    PROMPT_MASK,
    VIDEO_MASK
)

_PARALLEL_STRATEGY_LIST = []
_PARALLEL_STRATEGY_GROUP = {}
CACHED_BATCH = []


def deep_copy_batch(batch_data):
    """
    Recursively deep copy the batch data.
    Handles various data types including torch.Tensor, dict, list, tuple, and basic types.
    """
    if isinstance(batch_data, torch.Tensor):
        return batch_data.clone().detach()
    elif isinstance(batch_data, dict):
        return {key: deep_copy_batch(value) for key, value in batch_data.items()}
    elif isinstance(batch_data, list):
        return [deep_copy_batch(item) for item in batch_data]
    elif isinstance(batch_data, tuple):
        return tuple(deep_copy_batch(item) for item in batch_data)
    elif isinstance(batch_data, (str, int, float, bool, type(None))):
        return batch_data
    else:
        return copy.deepcopy(batch_data)


def move_tensors_to_device(obj, target_device):
    """
    Move all tensors in the object to the specified device.
    Handles various data types including torch.Tensor, dict, list, tuple, and set.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(target_device)
    elif isinstance(obj, dict):
        return {key: move_tensors_to_device(value, target_device) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        processed_items = [move_tensors_to_device(item, target_device) for item in obj]
        return type(obj)(processed_items)
    elif isinstance(obj, set):
        return {move_tensors_to_device(item, target_device) for item in obj}
    else:
        return obj


def generate_parallel_strategy_options(max_cp_size):
    """
    Generate possible parallel strategies based on the maximum context parallel size.
    """
    ori_dp_size = global_mpu.get_data_parallel_world_size()
    ori_cp_size = global_mpu.get_context_parallel_world_size()
    ori_dpcp_world_size = ori_dp_size * ori_cp_size

    max_cp_size = min(ori_dpcp_world_size, max_cp_size)
    max_cp_power = int(max_cp_size).bit_length() - 1

    global _PARALLEL_STRATEGY_LIST
    for cp_power in range(max_cp_power + 1):
        cp_size = 2**cp_power
        dp_size = int(ori_dpcp_world_size // cp_size)
        _PARALLEL_STRATEGY_LIST.append([dp_size, cp_size])


def initialize_parall_switch_list(timeout):
    """
    Initialize the parallel strategy groups with the given timeout.
    """
    args = get_args()
    generate_parallel_strategy_options(args.mm.model.max_cp_size)
    for index, option in enumerate(_PARALLEL_STRATEGY_LIST):
        rank_generator = global_mpu.RankGenerator(
            tp=global_mpu.get_tensor_model_parallel_world_size(),
            ep=global_mpu.get_expert_model_parallel_world_size(),
            dp=option[0],
            pp=global_mpu.get_pipeline_model_parallel_world_size(),
            cp=option[1],
            order="tp-cp-ep-dp-pp",
        )
        _PARALLEL_STRATEGY_GROUP[index] = {}
        _PARALLEL_STRATEGY_GROUP[index]['dp_group'] = []
        _PARALLEL_STRATEGY_GROUP[index]['dp_group_gloo'] = []
        _PARALLEL_STRATEGY_GROUP[index]['dp'] = rank_generator.get_ranks('dp')

        for ranks in _PARALLEL_STRATEGY_GROUP[index]['dp']:
            _PARALLEL_STRATEGY_GROUP[index]['dp_group'].append(torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=global_mpu.get_nccl_options('dp', {})
            ))
            _PARALLEL_STRATEGY_GROUP[index]['dp_group_gloo'].append(torch.distributed.new_group(ranks, timeout=timeout, backend="gloo"))

        _PARALLEL_STRATEGY_GROUP[index]['cp_group'] = []
        _PARALLEL_STRATEGY_GROUP[index]['cp'] = rank_generator.get_ranks('cp')
        for ranks in _PARALLEL_STRATEGY_GROUP[index]['cp']:
            _PARALLEL_STRATEGY_GROUP[index]['cp_group'].append(torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=global_mpu.get_nccl_options('cp', {})
            ))


def modify_parallel(strategy_idx):
    """
    Modify the current parallel strategy based on the given strategy index.
    """
    rank = torch.distributed.get_rank()
    parallel_strategy = _PARALLEL_STRATEGY_GROUP[int(strategy_idx)]
    for ranks, group, group_gloo in zip(parallel_strategy['dp'], parallel_strategy['dp_group'], parallel_strategy['dp_group_gloo']):
        if rank in ranks:
            global_mpu._DATA_PARALLEL_GROUP = group
            global_mpu._DATA_PARALLEL_GROUP_GLOO = group_gloo
            global_mpu._DATA_PARALLEL_GLOBAL_RANKS = ranks
    for ranks, group in zip(parallel_strategy['cp'], parallel_strategy['cp_group']):
        if rank in ranks:
            global_mpu._CONTEXT_PARALLEL_GROUP = group
            global_mpu._CONTEXT_PARALLEL_GLOBAL_RANKS = ranks
    torch.distributed.barrier()


def get_batch_on_this_tp_rank(data_iterator):
    """
    Get a batch from the data iterator on the current TP rank.
    """
    args = get_args()
    interleaved = args.mm.model.interleaved \
        if hasattr(args.mm.model, "interleaved") else False
    if data_iterator is not None:
        batch = next(data_iterator, None)
    else:
        return None
    # data is loaded in cpu for interleaved.
    if batch is not None and not interleaved:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(torch_npu.npu.current_device())
    return batch


def get_dpcp_batch_for_step(data_iterator):
    if CACHED_BATCH:
        batch = CACHED_BATCH.pop(0)
    else:
        batch = get_batch(data_iterator)
    return batch


def get_batch(data_iterator):
    """Generate a batch."""
    if mpu.is_pipeline_first_stage():
        batch = get_batch_on_this_tp_rank(data_iterator)
        return batch
    else:
        return None


def dynamic_dpcp_transfer_data(batch):
    """
    在CP组内广播所有rank的batch数据
    将收集到的所有batch存储到CACHED_BATCH中
    """
    context_group = global_mpu.get_context_parallel_group()
    rank = torch.distributed.get_rank()
    device = batch[VIDEO].device
    group_size = torch.distributed.get_world_size(group=context_group)

    for src_rank_local in range(group_size):
        src_rank = torch.distributed.get_global_rank(context_group, src_rank_local)

        if rank != src_rank:
            received_batch = {}
        else:
            received_batch = batch

        # Broadcast each item in batch
        for key, value in batch.items():
            if value is not None and isinstance(value, torch.Tensor):
                shape = torch.tensor(list(value.shape), dtype=torch.long, device=device)
                shape_size = torch.tensor([len(shape)], dtype=torch.long, device=device)

                torch.distributed.broadcast(shape_size, src=src_rank, group=context_group)
                if rank == src_rank:
                    torch.distributed.broadcast(shape, src=src_rank, group=context_group)
                else:
                    shape = torch.zeros(shape_size.item(), dtype=torch.long, device=device)
                    torch.distributed.broadcast(shape, src=src_rank, group=context_group)

                if rank == src_rank:
                    torch.distributed.broadcast(value.contiguous(), src=src_rank, group=context_group)
                else:
                    received_tensor = torch.zeros(tuple(shape), dtype=value.dtype, device=device)
                    torch.distributed.broadcast(received_tensor, src=src_rank, group=context_group)
                    received_batch[key] = received_tensor
            else:
                if rank == src_rank:
                    message = [value]
                    torch.distributed.broadcast_object_list(message, src=src_rank, group=context_group)
                else:
                    message = [None]
                    torch.distributed.broadcast_object_list(message, src=src_rank, group=context_group)
                    received_batch[key] = message[0]

        received_batch_on_device = move_tensors_to_device(received_batch, f"npu:{rank}")
        CACHED_BATCH.append(received_batch_on_device)

        torch.distributed.barrier(group=context_group)


def get_optimized_parallel_strategy(input_seq):
    """
    Determine the optimized parallel strategy based on the input sequence size.
    """
    args = get_args()
    dp_group = global_mpu.get_data_parallel_group()
    dp_size = global_mpu.get_data_parallel_world_size()
    local_rank = torch.distributed.get_rank()
    local_info = {
        key: {
            'shape': tensor.shape
        } for key, tensor in input_seq.items()
    }
    gather_rank_dst = torch.distributed.get_global_rank(dp_group, 0)
    if local_rank == gather_rank_dst:
        all_info = [None] * dp_size
        torch.distributed.gather_object(local_info, all_info, dst=gather_rank_dst)
    else:
        torch.distributed.gather_object(local_info, None, dst=gather_rank_dst)
    strategy_idx = torch.tensor(0).to('npu')
    if local_rank == gather_rank_dst:
        seq_size_list = [info[VIDEO]['shape'].numel() for info in all_info]
        args = get_args()
        oom_flag = True
        for idx, strategy in enumerate(_PARALLEL_STRATEGY_LIST):
            dp, cp = strategy
            if max(seq_size_list) / cp <= args.mm.model.max_seq_size:
                oom_flag = False
                strategy_idx = torch.tensor(idx).to('npu')
                break
        if oom_flag is True:
            print("max_seq_size is too small")

    torch.distributed.broadcast(strategy_idx, src=gather_rank_dst)
    return strategy_idx


def data_aware_parallel_optimize(data_iterator):
    """
    Optimize the parallel strategy based on the data iterator and apply the optimzed strategy.
    """
    args = get_args()
    if is_use_dynamic_dpcp():
        batch = get_dpcp_batch_for_step(data_iterator)
        # 0. Extract the sequence
        input_seq = {VIDEO: batch[VIDEO]}
        # 1. Initialize as full DP
        modify_parallel(0)
        # 2. Optimize the algorithm and output the new parallel strategy and data rearrangement method
        strategy_idx = get_optimized_parallel_strategy(input_seq)
        if strategy_idx > 0:
            print_rank_0("adjust [dp,cp] to {}".format(_PARALLEL_STRATEGY_LIST[strategy_idx]))
            # 3. Refresh the parallel strategy
            modify_parallel(strategy_idx)
            reconfigure_num_microbatches_calculator(
                torch.distributed.get_rank(),
                args.rampup_batch_size,
                args.global_batch_size,
                args.micro_batch_size,
                _PARALLEL_STRATEGY_LIST[strategy_idx][0],
                args.decrease_batch_size_if_needed)

            # 4. Execute data rearrangement
            dynamic_dpcp_transfer_data(batch)
        else:
            reconfigure_num_microbatches_calculator(
                torch.distributed.get_rank(),
                args.rampup_batch_size,
                args.global_batch_size,
                args.micro_batch_size,
                global_mpu.get_data_parallel_world_size(),
                args.decrease_batch_size_if_needed)

            CACHED_BATCH.append(batch)


def is_use_dynamic_dpcp():
    args = get_args()
    return hasattr(args.mm.model, "use_dynamic_dpcp") and args.mm.model.use_dynamic_dpcp


def get_max_cp_size():
    args = get_args()
    return args.mm.model.max_cp_size if is_use_dynamic_dpcp() else mpu.get_context_parallel_world_size()
