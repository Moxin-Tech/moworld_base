# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
from typing import Optional, List, Tuple

import torch
import torch.distributed as dist

from .utils import cal_split_sizes, reorder_output, cal_split_sizes_multi
from ...distributed.parallel_state import get_parallel_state


PERMUTE_DIMS1 = {
    4: (1, 2, 3, 0),
    5: (1, 2, 3, 0, 4),
}


PERMUTE_DIMS2 = {
    4: (1, 2, 0, 3),
    5: (1, 2, 0, 3, 4),
}


def adjust_tensor_dimensions(tensor, scatter_idx, gather_idx):
    """
    Adjusts the dimensions of a tensor to move scatter_idx and gather_idx to dim 0 and dim 1 respectively.

    Args:
        tensor (torch.Tensor): The input tensor.
        scatter_idx (int): The index of the dimension to scatter.
        gather_idx (int): The index of the dimension to gather.

    Returns:
        tuple: A tuple containing the adjusted tensor and the list of adjusted dimensions.
    """
    dims = list(range(tensor.dim()))

    if gather_idx == 0:
        if scatter_idx != 1:
            dims[1], dims[gather_idx] = dims[gather_idx], dims[1]
            dims[0], dims[scatter_idx] = dims[scatter_idx], dims[0]
        else:
            dims[scatter_idx], dims[gather_idx] = dims[gather_idx], dims[scatter_idx]

    elif gather_idx == 1:
        if scatter_idx != 0:
            # If scatter_idx is not 0, move it to 0
            dims[0], dims[scatter_idx] = dims[scatter_idx], dims[0]
    else:
        if scatter_idx == 0:
            dims[1], dims[gather_idx] = dims[gather_idx], dims[1]
        else:
            dims[0], dims[scatter_idx] = dims[scatter_idx], dims[0]
            dims[1], dims[gather_idx] = dims[gather_idx], dims[1]
    return tensor.permute(dims).contiguous(), dims


def unadjust_tensor_dimensions(tensor, adjusted_dims):
    """
    Reverses the dimension adjustments using the list of adjusted dimensions.

    Args:
        tensor (torch.Tensor): The tensor whose dimensions need to be restored.
        adjusted_dims (list): The list of adjusted dimensions used during the adjustment process.

    Returns:
        torch.Tensor: The tensor with its dimensions reverted to the original order.
    """
    inverse_dims = [0] * len(adjusted_dims)

    for new_pos, old_pos in enumerate(adjusted_dims):
        inverse_dims[old_pos] = new_pos

    # Restore the dimension order
    unadjusted_tensor = tensor.permute(inverse_dims).contiguous()
    return unadjusted_tensor


def _all_to_all(
    input_: torch.Tensor,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
    gather_size: Optional[int] = None
):
    """
    Helper function to perform the all-to-all operation. It scatters the input tensor along the specified scatter
    dimension and then gathers it along the specified gather dimension. The function supports aligned and unaligned
    data.
    Args:
        input_ (torch.Tensor): The input tensor to be processed.
        group (dist.ProcessGroup): The process group perform the operation within.
        scatter_dim (int): The index of the dimension that needs to be scattered.
        gather_dim (int): The index of the dimension that needs to be gathered.
        gather_size (Optional[int]): The total size of the output tensor along the `gather_dim`. If not provided, it
        will be calculated as the product of the original size of the `gather_dim` of the input tensor and the
        `world_size`.

    Returns:
        torch.Tensor: The resulting tensor after performing the all-to-all operation.

    Note:
        - The tensor will be split into `world_size` chunks along the `scatter_dim`. Each process will receive one
          chunk. If the total size of the `scatter_dim` is not divisible by `world_size`, the extra elements will be
          distributed to the first few processes, ensuring that no process receives more than one additional element
          compared to the others.
        - The tensor will be gathered along the `gather_dim`, with each process contributing its part to form the
          final output tensor. The gathering process also supports unaligned data, where the remainder elements
          are distributed to the first few processes.
    """

    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input_

    scatter_size = input_.size(scatter_dim)
    if gather_size is None:
        gather_size = input_.size(gather_dim) * world_size
    gather_mod = gather_size % world_size
    scatter_mod = scatter_size % world_size

    if gather_mod == 0 and scatter_mod == 0:
        # In the case of aligned data (both scatter_size and gather_size are divisible by world_size),
        # _aligned_all_to_all function performs better than _partial_unaligned_all_to_all function
        return _aligned_all_to_all(input_, group, scatter_dim, gather_dim)
    elif gather_mod != 0 and scatter_mod != 0:
        return _full_unaligned_all_to_all(input_, group, scatter_dim, gather_dim, gather_size)
    else:
        return _partial_unaligned_all_to_all(input_, group, scatter_dim, gather_dim, gather_size)


def _full_unaligned_all_to_all(
    input_: torch.Tensor,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
    gather_size: Optional[int] = None
):
    """
    Helper function to perform the all-to-all operation. It scatters the input tensor along the specified scatter
    dimension and then gathers it along the specified gather dimension. This function supports unaligned scatter
    and gather sizes.

    Args:
        input_ (torch.Tensor): The input tensor to be processed.
        world_size (int): The number of processes in the process group.
        group (dist.ProcessGroup): The process group to perform the operation within.
        scatter_dim (int): The index of the dimension that needs to be scattered.
        gather_dim (int): The index of the dimension that needs to be gathered.
        gather_size (Optional[int]): The total size of the output tensor along the `gather_dim`. If not provided, it
        will be calculated as the product of the original size of the `gather_dim` of the input tensor and the
        `world_size`.

    Returns:
        torch.Tensor: The resulting tensor after performing the all-to-all operation.
    """
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)

    scatter_sizes = cal_split_sizes(dim_size=input_.size(scatter_dim), world_size=world_size)
    input_list = [t.contiguous() for t in torch.split(input_, scatter_sizes, scatter_dim)]

    gather_sizes = cal_split_sizes(dim_size=gather_size, world_size=world_size)
    output_list = []
    tensor_shape_base = input_list[rank].size()
    for i in range(world_size):
        tensor_shape = list(tensor_shape_base)
        tensor_shape[gather_dim] = gather_sizes[i]
        output_list.append(torch.empty(tensor_shape, dtype=input_.dtype, device=input_.device))

    dist.all_to_all(output_list, input_list, group=group)

    return torch.cat(output_list, dim=gather_dim).contiguous()


def _aligned_all_to_all(
    input_: torch.Tensor,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
):
    """
    Helper function to perform the all-to-all operation. It scatters the input tensor along the specified scatter
    dimension and then gathers it along the specified gather dimension.
    Special note: The function only supports aligned data (both scatter_size and gather_size are divisible by
    world_size)
    """
    world_size = dist.get_world_size(group)
    inp_shape = list(input_.shape)
    inp_shape[scatter_dim] = inp_shape[scatter_dim] // world_size
    if scatter_dim == 0:
        input_t = input_.reshape([world_size] + inp_shape).contiguous()
    else:
        input_t = input_.reshape([-1, world_size] + inp_shape[scatter_dim:]).transpose(0, 1).contiguous()

    output = torch.empty_like(input_t)

    dist.all_to_all_single(output, input_t, group=group)

    output = output.view([world_size] + inp_shape).contiguous()
    output_dim = output.dim()
    if gather_dim == 1:
        # the shape of input_t is (world_size, inp_shape[0], inp_shape[gather_dim], *inp_shape[2:])
        output = output.transpose(0, 1).contiguous()
        # the shape of output is (inp_shape[0], world_size, inp_shape[gather_dim], *inp_shape[2:])
    elif gather_dim == 2:
        # the shape of input_t is (world_size, inp_shape[0], inp_shape[1], *inp_shape[gather_dim:])
        output = output.permute(*PERMUTE_DIMS2[output_dim]).contiguous()
        # the shape of output is (inp_shape[0], inp_shape[1], world_size, *inp_shape[gather_dim:])
    elif gather_dim == 3:
        # the shape of input_t is (world_size, inp_shape[0], inp_shape[1], inp_shape[2], inp_shape[gather_dim])
        output = output.permute(*PERMUTE_DIMS1[output_dim]).contiguous()
        # the shape of output is (inp_shape[0], inp_shape[1], inp_shape[2], world_size, inp_shape[gather_dim])
    # The last case: gather_dim == 0:
    # the shape of input_t is (world_size, inp_shape[gather_dim], inp_shape[0], *inp_shape[1:])
    # output requires no action
    # the shape of output is (world_size, inp_shape[gather_dim], inp_shape[0], *inp_shape[1:])
    output = output.view(inp_shape[:gather_dim] + [inp_shape[gather_dim] * world_size, ] + inp_shape[gather_dim + 1:]
                         ).contiguous()

    return output


def _partial_unaligned_all_to_all(
    input_: torch.Tensor,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
    gather_size: Optional[int] = None
):
    """
    Helper function to perform the all-to-all operation. It scatters the input tensor along the specified scatter
    dimension and then gathers it along the specified gather dimension. The function supports aligned and unaligned
    data.
    Special note: In the case of aligned data (both scatter_size and gather_size are divisible by world_size),
    _partial_unaligned_all_to_all function performs worse than _aligned_all_to_all function. Therefore, in the case of
    aligning data, it is recommended to use _aligned_all_to_all function.
    """
    world_size = dist.get_world_size(group)
    input_ = input_.contiguous()
    rank = dist.get_rank(group=group)

    scatter_size = input_.size(scatter_dim)
    if gather_size is None:
        gather_size = input_.size(gather_dim) * world_size


    scatter_size_per_rank = scatter_size // world_size
    scatter_size_remainder = scatter_size % world_size
    input_split_sizes = [scatter_size_per_rank + (1 if i < scatter_size_remainder else 0) for i in range(world_size)]

    gather_size_per_rank = gather_size // world_size
    gather_size_remainder = gather_size % world_size
    output_split_sizes = [gather_size_per_rank + (1 if i < gather_size_remainder else 0) for i in range(world_size)]

    # Adjusts the dimensions of a tensor to move scatter_idx and gather_idx to dim 0 and dim 1 respectively.
    reshaped_input, reshaped_input_dims = adjust_tensor_dimensions(input_, scatter_dim, gather_dim)
    reshaped_input_shape = list(reshaped_input.shape)
    # the shape of reshaped_input is (input_.size(scatter_dim), input_.size(gather_dim), *reshaped_input_shape[2:])

    if scatter_size % world_size == 0:
        reshaped_input = reshaped_input.view(
            [world_size, input_.size(scatter_dim) // world_size, input_.size(gather_dim)] + reshaped_input_shape[2:]
        ).transpose(1, 2).contiguous()

    output_dims = reshaped_input_dims
    # Relative to reshaped_input(the return value of adjust_tensor_dimensions func),
    # which shape is (input_.size(scatter_dim), input_.size(gather_dim), *reshaped_input_shape[2:]),
    # output just swaps the 0th and 1st axes.
    output_dims[1], output_dims[0] = output_dims[0], output_dims[1]
    output = torch.empty((gather_size, input_split_sizes[rank], *reshaped_input_shape[2:]),
                         dtype=input_.dtype, device=input_.device)
    output_shape = list(output.shape)

    dist.all_to_all_single(
        output,
        reshaped_input,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes if scatter_size % world_size != 0 else [1 for _ in range(world_size)],
        group=group,
    )

    if gather_size % world_size == 0 and scatter_size % world_size != 0:
        output = output.view(
            [world_size, input_split_sizes[rank], gather_size // world_size] + reshaped_input_shape[2:]
        ).transpose(1, 2).reshape(output_shape).contiguous()

    # Reverses the dimension adjustments using the list of adjusted dimensions.
    unadjust_output_ = unadjust_tensor_dimensions(output, output_dims)

    return unadjust_output_


class _AllToAll(torch.autograd.Function):
    """Custom autograd function that performs an all-to-all communication.
    This function supports both aligned and unaligned data.
    """
    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim, gather_size=None):
        """
        Forward pass: Perform all-to-all communication by scattering the input tensor along the specified scatter
        dimension and then gathering it along the specified gather dimension.

        Args:
            input_ (torch.Tensor): The input tensor to be processed.
            process_group (dist.ProcessGroup): The process group to perform the operation within.
            scatter_dim (int): The index of the dimension that needs to be scattered.
            gather_dim (int): The index of the dimension that needs to be gathered.
            gather_size (int): The size of the gather dimension.

        Returns:
            torch.Tensor: The resulting tensor after performing the all-to-all operation.
        """
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.scatter_size = input_.size(scatter_dim)
        ctx.gather_dim = gather_dim
        ctx.gather_size = gather_size
        output = _all_to_all(
            input_, process_group, scatter_dim, gather_dim, gather_size
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Perform the reverse all-to-all communication

        Args:
            grad_output (torch.Tensor): The gradient of the output with respect to the loss.

        Returns:
            tuple: The gradient of the input with respect to the loss and `None` for other arguments.
        """
        grad_output = _all_to_all(
            grad_output,
            ctx.process_group,
            ctx.gather_dim,
            ctx.scatter_dim,
            ctx.scatter_size
        )
        return (
            grad_output,
            None,
            None,
            None,
            None,
            None
        )


def _split(
        input_: torch.Tensor,
        pg: dist.ProcessGroup,
        dim: int = -1,
        split_sizes: Optional[List[int]] = None
) -> torch.Tensor:
    """
    Splits a tensor across the specified dimension and returns the part corresponding to the current rank,
    supporting aligned and unaligned data.

    Args:
        input_ (torch.Tensor): The input tensor to be split.
        pg (dist.ProcessGroup): The process group to perform the operation within.
        dim (int, optional): The dimension along which to split the tensor. Defaults to -1 (last dimension).
        split_sizes (Optional[List[int]], optional): A list of sizes for each part of the tensor to be split.
            If not provided, the tensor will be split equally among the processes, with the remainder
            distributed to the first few processes. Defaults to None.

    Returns:
        torch.Tensor: The part of the tensor corresponding to the current rank in the process group.
    """
    # Ensure split_sizes is a list if provided
    if split_sizes is not None and not isinstance(split_sizes, list):
        raise ValueError("split_sizes must be a list if provided.")

    # skip if only one rank involved
    world_size = dist.get_world_size(pg)

    if world_size == 1:
        return input_

    # Calculate split sizes if not provided
    if split_sizes is None:
        dim_size = input_.size(dim)
        base_size = dim_size // world_size
        remainder = dim_size % world_size

        # Calculate the size for each process
        split_sizes = [base_size + 1 if i < remainder else base_size for i in range(world_size)]

    tensor_list = torch.split(input_, split_sizes, dim=dim)

    # Get the part corresponding to the current rank
    rank = dist.get_rank(pg)
    output = tensor_list[rank].contiguous()

    return output


def _gather(input_: torch.Tensor,
            pg: dist.ProcessGroup,
            dim: int = -1,
            gather_sizes: Optional[List[int]] = None):
    """
    Gathers tensors from all processes in the process group and concatenates them along the specified dimension,
    supporting aligned and unaligned data.

    Args:
        input_ (torch.Tensor): The input tensor to be gathered.
        pg (dist.ProcessGroup): The process group to perform the operation within.
        dim (int, optional): The dimension along which to concatenate the gathered tensors. Defaults to -1 (last dimension).
        gather_sizes (Optional[List[int]], optional): A list of sizes for each part of the tensor to be gathered.
            If not provided, it is assumed that all tensors have the same shape as the input tensor. Defaults to None.

    Returns:
        torch.Tensor: The concatenated tensor after gathering from all processes in the process group.
    """
    # Ensure gather_sizes is a list if provided
    if gather_sizes is not None and not isinstance(gather_sizes, list):
        raise ValueError("gather_sizes must be a list if provided.")

    # Skip if only one rank is involved
    world_size = dist.get_world_size(pg)
    if world_size == 1:
        return input_

    input_ = input_.contiguous()

    # Prepare the output list with appropriate shapes
    if gather_sizes:
        tensor_list = []
        tensor_shape_base = input_.size()
        for i in range(world_size):
            tensor_shape = list(tensor_shape_base)
            tensor_shape[dim] = gather_sizes[i]
            tensor_list.append(torch.empty(tensor_shape, dtype=input_.dtype, device=input_.device))
    else:
        tensor_list = [torch.empty_like(input_, dtype=input_.dtype, device=input_.device) for _ in range(world_size)]

    torch.distributed.all_gather(tensor_list, input_, group=pg)

    # concat
    output = torch.cat(tensor_list, dim=dim).contiguous()
    return output


class _GatherForwardSplitBackward(torch.autograd.Function):
    """
    Custom autograd function that gathers the input tensor from all processes in the model parallel region and
    concatenates them.
    During the backward pass, it splits the gradients and scales them according to the gradient scaling mode.

    """

    @staticmethod
    def symbolic(graph, input_, process_group, dim, gather_sizes):
        """
        Define the symbolic representation of the custom operation.
        """
        return _gather(input_, process_group, dim, gather_sizes)

    @staticmethod
    def forward(ctx, input_, process_group, dim, gather_sizes, grad_scale="up"):
        """
        Forward pass: Gathers tensors from all processes in the specified process group and concatenates them along the specified dimension.

        Args:
            input_ (torch.Tensor): The input tensor to be processed.
            process_group (dist.ProcessGroup): The process group to perform the operation within.
            dim (int): The dimension along which to concatenate the gathered tensors.
            gather_sizes (Optional[List[int]], optional): A list of sizes for each part of the tensor to be gathered.
            grad_scale (str, optional): Gradient scaling mode. Can be "up", "down", or None. Defaults to "up".

        Returns:
            torch.Tensor: The resulting tensor after gathering and concatenating.
        """
        ctx.mode = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale

        ctx.gather_sizes = gather_sizes
        return _gather(input_, process_group, dim, ctx.gather_sizes)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Distribute the gradients to the input tensors and scales them according to the gradient scaling mode.

        Args:
            grad_output (torch.Tensor): The gradient of the output.

        Returns:
            torch.Tensor: The gradient of the input with respect to the loss.
        """
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.mode)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.mode)

        return _split(grad_output, ctx.mode, ctx.dim, ctx.gather_sizes), None, None, None, None


class _SplitForwardGatherBackward(torch.autograd.Function):
    """
    Custom autograd function that splits the input tensor and keeps only the corresponding chunk for the current rank.
    During the backward pass, it gathers the gradients and scales them according to the gradient scaling mode.

    """
    @staticmethod
    def symbolic(graph, input_, process_group, dim, split_sizes):
        return _split(input_, process_group, dim, split_sizes)

    @staticmethod
    def forward(ctx, input_, process_group, dim, split_sizes, grad_scale):
        ctx.mode = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale

        ctx.split_sizes = split_sizes

        return _split(input_, process_group, dim, ctx.split_sizes)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.mode)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.mode)
        return _gather(grad_output, ctx.mode, ctx.dim, ctx.split_sizes), None, None, None, None


def all_to_all(
        input_: torch.Tensor,
        process_group: dist.ProcessGroup,
        scatter_dim: int = 2,
        gather_dim: int = 1,
        gather_size: Optional[int] = None
):
    """
    Performs an all-to-all operation on the input tensor. The input tensor is scattered along the specified scatter
    dimension and then gathered along the specified gather dimension.
    This function supports both aligned and unaligned data.

    Args:
        input_ (torch.Tensor): The input tensor to be processed.
        process_group (dist.ProcessGroup): The process group to perform the operation within.
        scatter_dim (int, optional): The index of the dimension that needs to be scattered. Defaults to 2.
        gather_dim (int, optional): The index of the dimension that needs to be gathered. Defaults to 1.
        gather_size (Optional[int]): The total size of the output tensor along the `gather_dim`. If not provided, it
        will be calculated as the product of the original size of the `gather_dim` of the input tensor and the
        `world_size`.

    Returns:
        torch.Tensor: The resulting tensor after performing the all-to-all operation.
    """
    return _AllToAll.apply(input_, process_group, scatter_dim, gather_dim, gather_size)


def split_forward_gather_backward(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    dim: int,
    split_sizes: Optional[List[int]] = None,
    grad_scale: str = "down"

) -> torch.Tensor:
    """
    Splits the input tensor and keeps only the corresponding chunk for the current rank.
    During the backward pass, it gathers the gradients and scales them according to the gradient scaling mode.
    This function supports both aligned and unaligned data.
    Args:
        input_ (torch.Tensor): The input tensor to be processed.
        process_group (dist.ProcessGroup): The process group to perform the operation within.
        dim (int): The dimension along which to split the tensor.
        split_sizes (Optional[List[int]], optional): A list of sizes for each part of the tensor to be split.
            If not provided, the tensor will be split equally among the processes. Defaults to None.
        grad_scale (str, optional): Gradient scaling mode. Can be "up", "down", or None. Defaults to "down".

    Returns:
        torch.Tensor: The resulting tensor after splitting and keeping only the corresponding chunk.
    """
    return _SplitForwardGatherBackward.apply(input_, process_group, dim, split_sizes, grad_scale)


def gather_forward_split_backward(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    dim: int,
    gather_sizes: Optional[List[int]] = None,
    grad_scale: str = "up"
) -> torch.Tensor:
    """
    Gathers the input tensor from all processes in the model parallel region and concatenates them along the specified
    dimension. During the backward pass, it splits the gradients and scales them according to the gradient scaling mode.
    This function handles both aligned and unaligned data during the gather and scatter operations.
    Args:
        input_ (torch.Tensor): The input tensor to be processed.
        process_group (dist.ProcessGroup): The process group to perform the operation within.
        dim (int): The dimension along which to concatenate the gathered tensors.
        gather_sizes (Optional[List[int]], optional): A list of sizes for each part of the tensor to be gathered.
            If not provided, it is assumed that all tensors have the same shape as the input tensor. Defaults to None.
        grad_scale (str, optional): Gradient scaling mode. Can be "up", "down", or None. Defaults to "up".

    Returns:
        torch.Tensor: The resulting tensor after gathering and concatenating.
    """
    return _GatherForwardSplitBackward.apply(input_, process_group, dim, gather_sizes, grad_scale)


class _SplitForwardGatherBackWardWithMegatronCP(torch.autograd.Function):
    '''
    Split the input tensor in the forward pass and gather the gradients in the backward pass.
    It will be implemented in Mindspeed in the future.
    '''
    @staticmethod
    def forward(ctx, val, cp_rank, cp_size, seq_dim, cp_group=None):
        val = val.view(
            *val.shape[0:seq_dim],
            2 * cp_size,
            val.shape[seq_dim] // (2 * cp_size),
            *val.shape[(seq_dim + 1):],
        )
        index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device=val.device)
        val = val.index_select(seq_dim, index)
        val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2):])

        ctx.cp_group = cp_group
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.seq_dim = seq_dim

        return val

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = {}
        grad_input = _gather(grad_output, ctx.cp_group, dim=ctx.seq_dim) / ctx.cp_size
        grad_input = reorder_output(grad_input, ctx.cp_rank, ctx.cp_size, ctx.cp_group, dim=ctx.seq_dim)
        return grad_input, None, None, None, None


class _GatherForwardSplitBackWardWithMegatronCP(torch.autograd.Function):
    '''
    Split the input tensor in the forward pass and gather the gradients in the backward pass with megatron cp(Ring Attention)
    It will be implemented in Mindspeed in the future.
    '''
    @staticmethod
    def forward(ctx, val, cp_rank, cp_size, seq_dim, cp_group=None):
        # Step 1: All-gather shards from all CP ranks along the sequence dimension
        val = _gather(val, cp_group, dim=seq_dim)
        # Step 2: Reorder the gathered tensor
        val = reorder_output(val, cp_rank, cp_size, cp_group, dim=seq_dim)

        ctx.cp_group = cp_group
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.seq_dim = seq_dim

        return val

    @staticmethod
    def backward(ctx, grad_output):
        cp_group = ctx.cp_group
        cp_rank = ctx.cp_rank
        cp_size = ctx.cp_size
        seq_dim = ctx.seq_dim

        grad_output = grad_output.view(
            *grad_output.shape[0:seq_dim],
            2 * cp_size,
            grad_output.shape[seq_dim] // (2 * cp_size),
            *grad_output.shape[(seq_dim + 1):],
        ) * cp_size  # Scale gradients up by cp_size
        # Select the two chunks that belong to the current rank:
        # - One from the forward direction (index = cp_rank)
        # - One from the backward direction (index = 2*cp_size - cp_rank - 1)
        index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device=grad_output.device)
        grad_output = grad_output.index_select(seq_dim, index)

        # Collapse the two selected chunks back into a single contiguous local sequence
        grad_input = grad_output.view(*grad_output.shape[0:seq_dim], -1, *grad_output.shape[(seq_dim + 2):])

        return grad_input, None, None, None, None


def load_balanced_split_forward_gather_backward(
        input_: torch.Tensor,
        process_group: torch.distributed.ProcessGroup,
        dim: int = 0
) -> torch.Tensor:
    cp_size = torch.distributed.get_world_size(group=process_group)
    cp_rank = torch.distributed.get_rank(group=process_group)

    return _SplitForwardGatherBackWardWithMegatronCP.apply(input_, cp_rank, cp_size, dim, process_group)


def load_balanced_gather_forward_split_backward(
    input_: torch.Tensor,
    process_group: torch.distributed.ProcessGroup,
    dim: int = 0
) -> torch.Tensor:
    cp_size = torch.distributed.get_world_size(group=process_group)
    cp_rank = torch.distributed.get_rank(group=process_group)

    return _GatherForwardSplitBackWardWithMegatronCP.apply(input_, cp_rank, cp_size, dim, process_group)


def split_forward_gather_backward_with_cp(
    input_: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """
    Perform a context-parallel-aware tensor split during forward pass and gather during backward pass.

    This function supports multiple context parallel (CP) algorithms:
      - Ulysses-style CP: uniform or non-uniform split across CP ranks.
      - Ring Attention: typically used with sequence parallelism and ring-based communication.
      - Hybrid CP: combines ring and Ulysses-style splitting in nested groups.
    """
    ps = get_parallel_state()
    seq_len = input_.shape[dim]
    if ps.is_ring_enable():
        if seq_len % (2 * ps.get_ring_group_size()) != 0:
            raise ValueError(f"Seq lens should be multiple of 2 * ring_size, but got seq_len: {seq_len}, ring_size: {ps.get_ring_group_size()}")
        input_ = load_balanced_split_forward_gather_backward(input_, ps.get_ring_group(), dim=dim)
        seq_len = input_.shape[dim]
    if ps.is_ulysses_enable():
        split_gather_sizes = cal_split_sizes(seq_len, ps.get_ulysses_group_size())
        input_ = split_forward_gather_backward(input_, ps.get_ulysses_group(), dim=dim, split_sizes=split_gather_sizes)

    return input_


def gather_forward_split_backward_with_cp(
    input_: torch.Tensor,
    dim: int,
    gather_size: int
):
    """
    Perform a context-parallel-aware tensor gather during forward pass and split during backward pass.
    This is the reverse operation of split_forward_gather_backward_with_cp.
    """
    ps = get_parallel_state()
    if ps.is_ring_enable():
        if gather_size % (2 * ps.get_ring_group_size()) != 0:
            raise ValueError(f"Total gather size should be multiple of 2 * ring_size, but got total gather size: {gather_size}, ring_size: {ps.get_ring_group_size()}")
        # Calculate the sequence length per ring CP group for Ulysses processing.
        # Since padding is applied in ring groups, the division yields an integer.
        gather_size = gather_size // ps.get_ring_group_size()

    if ps.is_ulysses_enable():
        gather_size_list = cal_split_sizes(gather_size, ps.get_ulysses_group_size())
        input_ = gather_forward_split_backward(input_, ps.get_ulysses_group(), dim=dim, gather_sizes=gather_size_list)
    if ps.is_ring_enable():
        input_ = load_balanced_gather_forward_split_backward(input_, ps.get_ring_group(), dim=dim)

    return input_


def packed_data_split_forward_gather_backward(
    hidden_states: torch.Tensor,  # Concatenated sequences: s1+s2+s3+...
    process_group: torch.distributed.ProcessGroup,
    sequence_lengths: List[int],  # List of individual sequence lengths: [s1_len, s2_len, s3_len, ...]
    dim: int = 0
) -> torch.Tensor:
    """
    Split packed hidden states for sequence parallelism, handling multiple variable-length sequences.

    This function:
    1. Splits each sequence across all processes in the process group
    2. Returns the local chunks for the current rank

    Args:
        hidden_states: Concatenated hidden states of all sequences [total_seq_len, hidden_dim]
        process_group: Distributed process group for parallelism
        sequence_lengths: List of lengths for each individual sequence
        dim: Dimension along which to split (default: 0 - sequence dimension)

    Returns:
        local_hidden_states: Local chunks of hidden states for current rank
    """

    world_size = torch.distributed.get_world_size(process_group)
    local_sequence_chunks = []     # Local chunks for current rank

    current_position = 0

    # Process each sequence individually
    for _, seq_len in enumerate(sequence_lengths):
        # Extract the current sequence from packed hidden states
        sequence_end = current_position + seq_len
        current_sequence = hidden_states.narrow(dim=dim, start=current_position, length=seq_len)
        current_position = sequence_end

        # Calculate how to split this sequence across all ranks
        sequence_split_sizes = cal_split_sizes(seq_len, world_size)

        # Split sequence and get local chunk for current rank
        local_chunk = split_forward_gather_backward(
            current_sequence,
            process_group=process_group,
            dim=dim,
            grad_scale="down",  # Scale gradients during backward
            split_sizes=sequence_split_sizes
        )
        local_sequence_chunks.append(local_chunk)

    # Concatenate all local chunks along sequence dimension
    local_hidden_states = torch.cat(local_sequence_chunks, dim=dim)

    return local_hidden_states


def packed_data_gather_forward_split_backward(
    local_hidden_states: torch.Tensor,
    all_split_sizes: torch.Tensor,  # [world_size, num_sequences]
    process_group: torch.distributed.ProcessGroup,
    dim: int = 0
) -> torch.Tensor:
    """
    Gather distributed sequence chunks and reconstruct original packed sequences.

    Reconstruction process:
    1. For each sequence: gather chunks from all ranks and concatenate
    2. Concatenate all reconstructed sequences along sequence dimension

    Args:
        local_hidden_states: Local hidden states [local_seq_len, hidden_dim]
        all_split_sizes: Split sizes tensor [world_size, num_sequences] showing how each sequence
                        was distributed across ranks
        process_group: Distributed process group for communication
        dim: Dimension along which sequences were split (default: 0 - sequence dimension)

    Returns:
        reconstructed_sequences: Reconstructed packed sequences [total_seq_len, hidden_dim]
    """
    world_size, num_sequences = all_split_sizes.shape
    rank = torch.distributed.get_rank(process_group)

    # Calculate total sequence length handled by each rank
    # This sums up all sequence chunks that belong to each rank
    rank_total_sizes = []
    for r in range(world_size):
        total_size_this_rank = all_split_sizes[r].sum().item()  # Sum across all sequences
        rank_total_sizes.append(total_size_this_rank)

    # Step 1: Gather all local chunks from all ranks into a single tensor
    all_gathered_tensor = gather_forward_split_backward(
        local_hidden_states,
        process_group=process_group,
        dim=dim,
        grad_scale="up",  # Scale gradients during backward
        gather_sizes=rank_total_sizes
    )

    # Step 2: Split the gathered tensor into chunks corresponding to each rank's contribution
    # This separates the gathered data back into per-rank chunks for processing
    rank_chunks = torch.split(all_gathered_tensor, rank_total_sizes, dim=dim)
    rank_seq_chunks = [list(torch.split(rank_chunks[i], all_split_sizes[i].tolist(), dim=dim)) for i in range(world_size)]

    # Step 3: Reconstruct each original sequence by collecting chunks from all ranks
    reconstructed_sequences = []

    for _ in range(num_sequences):
        # For current sequence, collect chunks from all ranks
        sequence_chunks = [rank_seq_chunks[i].pop(0) for i in range(world_size)]
        # Concatenate all chunks for this sequence
        reconstructed_sequences.extend(sequence_chunks)

    # Step 4: Concatenate all reconstructed sequences along sequence dimension
    return torch.cat(reconstructed_sequences, dim=dim)


def packed_data_split_forward_gather_backward_with_cp(
    x: torch.Tensor,
    dim: int,
    seq_lens: List[int]
):
    """
    Split packed sequences across context parallel (CP) ranks during the forward pass,
    and gather full gradients during the backward pass.

    This function supports three CP strategies:
      - **Ulysses CP**: Splits the entire packed sequence uniformly (or near-uniformly) across all CP ranks.
      - **Ring Attention**: Splits *each individual sample sequence* (e.g., image tokens) across CP ranks,
        then concatenates the resulting shards to form a new packed tensor.
      - **Hybrid CP**: First applies ring-based splitting per sample, then further splits the result
        using Ulysses within each ring subgroup.
    Args:
        x: Concatenated sequences: s1+s2+s3+...
    """
    ps = get_parallel_state()
    if ps.is_ring_enable():
        x = packed_data_split_forward_gather_backward(x, ps.get_ring_group(), seq_lens, dim)
    if ps.is_ulysses_enable():
        split_gather_sizes = cal_split_sizes(x.shape[dim], ps.get_ulysses_group_size())
        x = split_forward_gather_backward(x, ps.get_ulysses_group(), dim=dim, split_sizes=split_gather_sizes)
    return x


def packed_data_gather_forward_split_backward_with_cp(
    x: torch.Tensor,  # Concatenated sequences: s1+s2+s3+...
    dim: int,
    seq_lens: List[int]
):
    """
    Gather visual sequences across context parallel (CP) ranks during the forward pass,
    and split gradients back during the backward pass.

    This function supports multiple CP strategies:
      - **Ulysses CP**: All-gather full sequence using precomputed per-rank sequence lengths.
      - **Ring Attention**: Reconstruct packed tensor from sequence chunks distributed across CP ranks.
      - **Hybrid CP**: First gather within Ulysses subgroups, then across ring-based CP groups.
    """
    ps = get_parallel_state()
    # Step 1: Gather within Ulysses subgroups (inner CP group)
    # First, compute how packed seqs are distributed across ring CP ranks
    all_split_sizes_tensor = cal_split_sizes_multi(seq_lens, ps.get_ring_group_size())
    gather_sizes = cal_split_sizes(all_split_sizes_tensor[ps.get_ring_rank()].sum(), ps.get_ulysses_group_size())
    if ps.is_ulysses_enable():
        x = gather_forward_split_backward(x, ps.get_ulysses_group(), dim=dim, gather_sizes=gather_sizes)
    # Step 2: Gather across ring CP ranks
    if ps.is_ring_enable():
        x = packed_data_gather_forward_split_backward(x, all_split_sizes_tensor, ps.get_ring_group(), dim=dim)

    return x
