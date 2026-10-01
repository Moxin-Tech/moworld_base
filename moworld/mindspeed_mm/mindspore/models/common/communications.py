# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from typing import List

import torch
import torch.distributed as dist


def _gather(input_: torch.Tensor,
            pg: dist.ProcessGroup,
            dim: int = -1,
            gather_sizes: List = None
            ):
    input_ = input_.contiguous()
    world_size = dist.get_world_size(pg)

    # MS adapt: Extend the input_.device.type check to include the "Ascend" device type.
    if input_.device.type not in ["cuda", "npu", "Ascend"]:
        raise AssertionError("input tensor must in cuda or npu/Ascend")

    if world_size == 1:
        return input_

    # all gather
    if gather_sizes is not None:
        tensor_list = []
        tensor_shape_base = input_.size()
        for i in range(world_size):
            tensor_shape = list(tensor_shape_base)
            tensor_shape[dim] = gather_sizes[i]
            tensor_list.append(torch.empty(tensor_shape, dtype=input_.dtype, device=input_.device))
    else:
        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]

    torch.distributed.all_gather(tensor_list, input_, group=pg)

    # concat
    output = torch.cat(tensor_list, dim=dim).contiguous()

    return output
