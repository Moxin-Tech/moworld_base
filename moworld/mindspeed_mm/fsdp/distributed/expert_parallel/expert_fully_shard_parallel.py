import logging

import torch
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy
from torch.distributed.tensor import Shard
from torch.distributed.distributed_c10d import ReduceOp

from mindspeed.fsdp.parallel_engine_config import EPPlanConfig
from mindspeed.fsdp.utils.log import print_rank
from mindspeed.fsdp.utils.str_match import module_name_match
from mindspeed_mm.fsdp.params.parallel_args import FSDPPlanConfig
from mindspeed_mm.fsdp.distributed.fully_shard_parallel import (
    get_mixprecision_policy,
    get_fsdp_hook_modules,
    find_hook_module,
    get_efsdp_modules,
)

logger = logging.getLogger(__name__)


def expert_fully_shard_modules(model: torch.nn.Module, efsdp_mesh, ep_plan: EPPlanConfig, fsdp_plan: FSDPPlanConfig) -> torch.nn.Module:
    efsdp_modules = get_efsdp_modules(model, ep_plan)
    efsdp_hook_modules = get_fsdp_hook_modules(model, fsdp_plan)

    # Configure mixed precision if enabled
    cpu_offload = None
    if fsdp_plan.cpu_offload:
        cpu_offload = CPUOffloadPolicy(pin_memory=True)

    config = {'mesh': efsdp_mesh,
              'mp_policy': get_mixprecision_policy(fsdp_plan),
              'shard_placement_fn': lambda x: Shard(1),
              'offload_policy': cpu_offload}

    apply_hccl_premul_sum_patch()
    for experts in efsdp_modules:
        hook_module = find_hook_module(experts, efsdp_hook_modules)
        if isinstance(experts, torch.nn.ModuleList):
            for expert in experts:
                fully_shard(expert, hook_module=hook_module, **config)
                set_gradient_divide_factor(expert, ep_plan._gradient_divide_factor)
        else:
            fully_shard(experts, hook_module=hook_module, **config)
            set_gradient_divide_factor(experts, ep_plan._gradient_divide_factor)

    return model


def set_gradient_divide_factor(module, factor):
    if hasattr(module, 'set_gradient_divide_factor'):
        module.set_gradient_divide_factor(factor)
    else:
        module.set_reduce_scatter_divide_factor(factor)


def hccl_premul_sum_wrapper(op, output_name):
    """
    A wrapper for distributed operations to handle ReduceOp.PREMUL_SUM which is not supported in Huawei HCCL.
    This wrapper intercepts operations using ReduceOp.PREMUL_SUM and converts them into equivalent
    ReduceOp.SUM operations followed by scalar multiplication.
    """

    def wrapper(*args, **kwargs):
        # Note:Although the sequence of operations(ReduceOp.SUM followed by multiplication) may differ from semantics,
        # we have verified that there is no problem with the performance and accuracy of this sequence.
        factor = None
        if "op" in kwargs and kwargs["op"] == ReduceOp.PREMUL_SUM:
            factor = kwargs["op"].__getstate__()[1]
            kwargs["op"] = ReduceOp.SUM
        handle = op(*args, **kwargs)
        if handle is not None:
            handle.wait()
        if factor is not None:
            output = args[0] if len(args) > 0 else kwargs[output_name]
            output.data.mul_(factor)
        return handle

    return wrapper


def apply_hccl_premul_sum_patch():
    torch.distributed.all_reduce = hccl_premul_sum_wrapper(torch.distributed.all_reduce, "tensor")
    torch.distributed.reduce_scatter = hccl_premul_sum_wrapper(torch.distributed.reduce_scatter, "output")
    torch.distributed.reduce_scatter_tensor = hccl_premul_sum_wrapper(
        torch.distributed.reduce_scatter_tensor, "output"
    )
