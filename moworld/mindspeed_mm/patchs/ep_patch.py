from functools import wraps

import torch
from torch.distributed.tensor import distribute_tensor

from mindspeed_mm.models.transformers.global_vars import get_ep_fsdp_group, get_check_moe_func


def finalize_model_grads_wrapper(fn):

    @wraps(fn)
    def wrapper(model, num_tokens=None):
        fn(model, num_tokens)

        ep_fsdp_group = get_ep_fsdp_group()
        check_moe_func = get_check_moe_func()
        if ep_fsdp_group is None:
            return
        for model_chunk in model:
            for name, param in model_chunk.named_parameters():
                if param.grad is None or not check_moe_func(name):
                    continue
                local_grad = param.grad.to_local()
                # All-Reduce within EP FSDP group
                torch.distributed.all_reduce(
                    local_grad,
                    op=torch.distributed.ReduceOp.AVG,
                    group=ep_fsdp_group,
                    async_op=True,
                )
                dtensor = distribute_tensor(
                    local_grad,
                    device_mesh=param.grad.device_mesh,
                    placements=param.grad.placements,
                )
                param.grad.copy_(dtensor)
        torch.npu.current_stream().synchronize()

    return wrapper
