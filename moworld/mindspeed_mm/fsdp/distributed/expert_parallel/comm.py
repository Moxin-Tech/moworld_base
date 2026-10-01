from typing import List
import torch
import torch.distributed as dist
from mindspeed.fsdp.distributed.dist_ops import all_to_all as _all_to_all

EP_RANK_SEQ_LENS = None


def get_ep_rank_seq_lens() -> List:
    """Get the sequence lengths of all ranks in the Expert Parallel (EP) group.

    Returns:
        List: A list of sequence lengths from each rank in the EP group.
    """
    global EP_RANK_SEQ_LENS
    return EP_RANK_SEQ_LENS


def set_ep_rank_seq_lens(seq_len: int, ep_group: dist.ProcessGroup, device):
    """Set and synchronize the sequence length across all ranks in the EP group.

    This function performs an All-Gather operation to collect the sequence length
    from each rank, ensuring all ranks know the token count of every other rank.

    Args:
        seq_len (int): The sequence length of the current rank.
        ep_group (dist.ProcessGroup): The Expert Parallel communication group.
        device: The device (e.g., NPU/CUDA) where the tensor should be placed.
    """
    global EP_RANK_SEQ_LENS

    # Create a tensor for the local sequence length
    local_seq_len = torch.tensor([seq_len], dtype=torch.long, device=device)

    # Prepare a tensor to receive lengths from all ranks
    gathered_seq_lens = torch.empty(dist.get_world_size(ep_group), dtype=torch.long, device=device)

    # Perform All-Gather to collect sequence lengths
    dist.all_gather_into_tensor(gathered_seq_lens, local_seq_len, group=ep_group, async_op=False)

    # Convert to Python list for easier handling
    EP_RANK_SEQ_LENS = gathered_seq_lens.tolist()


def _gather_along_first_dim(
    local_tokens: torch.Tensor, ep_group: dist.ProcessGroup, async_op: bool = False
) -> torch.Tensor:
    """Gather tokens from all ranks in the Expert Parallel (EP) group.

    This function concatenates local token tensors from all ranks along the first dimension (batch).
    It handles both equal-length and variable-length (ragged) inputs efficiently.

    Args:
        local_tokens (torch.Tensor): Local token tensor of the current rank, shape=(N_local, D).
        ep_group (dist.ProcessGroup): The EP communication group.
        async_op (bool): Whether to perform the operation asynchronously. Defaults to False.

    Returns:
        torch.Tensor: The concatenated global token tensor, shape=(N_total, D),
                      where N_total = sum(N_local over all ranks in EP group).
    """
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed is not initialized.")

    ep_size = dist.get_world_size(group=ep_group)

    # Optimization: If EP group size is 1, no communication needed
    if ep_size == 1:
        return local_tokens

    # --- Handle Variable-Length (Ragged) Inputs ---
    # Check if sequence lengths are pre-set (for variable lengths) or assume equal length
    tokens_per_rank = get_ep_rank_seq_lens()

    if tokens_per_rank:
        token_counts = tokens_per_rank
    else:
        # Assume equal length if no global info is set
        token_counts = [local_tokens.shape[0] for _ in range(ep_size)]

    # Optimization: If all ranks have the same number of tokens, use efficient all_gather_into_tensor
    if len(set(token_counts)) == 1:
        total_tokens = token_counts[0] * ep_size
        # Construct output shape: [total_tokens, hidden_dim, ...]
        token_shape = [total_tokens] + list(local_tokens.shape)[1:]

        gathered_tokens = torch.empty(*token_shape, dtype=local_tokens.dtype, device=local_tokens.device)

        dist.all_gather_into_tensor(gathered_tokens, local_tokens, group=ep_group, async_op=async_op)
        return gathered_tokens
    else:
        # --- Variable-Length Handling (List-based AllGather) ---
        # When token counts vary across ranks, we must use list-based all_gather
        # and then concatenate manually.

        # Pre-allocate tensors for each rank based on their token count
        gathered_list: List[torch.Tensor] = [
            torch.empty(count, local_tokens.size(1), dtype=local_tokens.dtype, device=local_tokens.device)
            for count in token_counts
        ]

        # Perform All-Gather into the list
        dist.all_gather(gathered_list, local_tokens, group=ep_group, async_op=async_op)

        # Concatenate the list of tensors into a single tensor
        return torch.cat(gathered_list, dim=0)


def _reduce_scatter_along_first_dim(
    full_tokens: torch.Tensor, ep_group: dist.ProcessGroup, async_op: bool = False
) -> torch.Tensor:
    """Reduce-scatter operation for the output of MoE experts.

    This function performs a sum-reduction on the gathered results and scatters
    the chunks back to the corresponding ranks. It is the inverse operation of _gather_along_first_dim.

    Args:
        full_tokens (torch.Tensor): The complete token tensor after expert computation, shape=(N_total, D).
        ep_group (dist.ProcessGroup): The EP communication group.
        async_op (bool): Whether to perform the operation asynchronously. Defaults to False.

    Returns:
        torch.Tensor: The local output tensor for the current rank, shape=(N_local, D).
    """
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed is not initialized.")

    ep_size = dist.get_world_size(group=ep_group)

    if ep_size == 1:
        return full_tokens

    total_tokens, hidden_dim = full_tokens.shape

    # --- Input Validation ---
    # For standard reduce_scatter, the total token count must be divisible by EP group size
    # (Note: This is a limitation; some implementations use padding or all-to-all for raggedness)
    if total_tokens % ep_size != 0:
        raise ValueError(
            f"total_tokens ({total_tokens}) must be divisible by "
            f"EP group size ({ep_size}) for standard reduce_scatter. "
            "Consider padding or using all_to_all for variable lengths."
        )

    # --- Determine Local Output Size ---
    # Use pre-set sequence lengths if available (for variable-length handling)
    tokens_per_rank = get_ep_rank_seq_lens()

    if tokens_per_rank:
        local_tokens_size = tokens_per_rank[dist.get_rank(ep_group)]
        token_counts = tokens_per_rank
    else:
        # Fallback to equal division
        local_tokens_size = total_tokens // ep_size
        token_counts = [local_tokens_size] * ep_size

    # --- Execute Communication ---
    # Optimization: If all ranks output the same number of tokens
    if len(set(token_counts)) == 1:
        local_output = torch.empty(local_tokens_size, hidden_dim, dtype=full_tokens.dtype, device=full_tokens.device)
        # Perform Reduce-Scatter (Sum reduction by default)
        work = dist.reduce_scatter_tensor(output=local_output, input=full_tokens, group=ep_group, async_op=async_op)
    else:
        # --- Variable-Length Handling ---
        # Split the full tensor according to the token counts
        input_list = list(torch.split(full_tokens, tokens_per_rank, dim=0))

        local_output = torch.empty(local_tokens_size, hidden_dim, dtype=full_tokens.dtype, device=full_tokens.device)

        # Perform List-based Reduce-Scatter
        work = dist.reduce_scatter(local_output, input_list, group=ep_group, async_op=async_op)

    # --- Handle Asynchronous Operation ---
    if async_op:
        return work  # Return the Work handle for the caller to .wait()
    else:
        return local_output


class _GatherTokens(torch.autograd.Function):
    """Custom autograd function for gathering tokens.

    Forward:  AllGather tokens from EP group.
    Backward: ReduceScatter gradients.
    """

    @staticmethod
    def forward(ctx, input_, group):
        """Forward pass: Gather tokens."""
        ctx.group = group
        return _gather_along_first_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: Scatter gradients (inverse of gather)."""
        # During backward, we reduce-scatter the gradient
        return (_reduce_scatter_along_first_dim(grad_output, ctx.group), None)


class _ReduceScatterTokens(torch.autograd.Function):
    """Custom autograd function for reducing and scattering tokens.

    Forward:  ReduceScatter tokens.
    Backward: AllGather gradients.
    """

    @staticmethod
    def forward(ctx, input_, group):
        """Forward pass: Reduce and scatter tokens."""
        ctx.group = group
        return _reduce_scatter_along_first_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: Gather gradients (inverse of reduce-scatter)."""
        # During backward, we all-gather the gradient
        return (
            _gather_along_first_dim(grad_output, ctx.group),
            None,
        )


# --- Public API Wrappers ---
def allgather_tokens_in_ep(input_, ep_group=None):
    """Wrapper function: Forward performs AllGather, Backward performs ReduceScatter."""
    return _GatherTokens.apply(input_, ep_group)


def reduce_scatter_tokens_in_ep(input_, ep_group=None):
    """Wrapper function: Forward performs ReduceScatter, Backward performs AllGather."""
    return _ReduceScatterTokens.apply(input_, ep_group)


def all_to_all(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    scatter_dim: int = 2,
    gather_dim: int = 1,
    scatter_sizes: List = None,
    gather_sizes: List = None
):
    return _all_to_all(process_group, input_, gather_sizes, scatter_sizes)
