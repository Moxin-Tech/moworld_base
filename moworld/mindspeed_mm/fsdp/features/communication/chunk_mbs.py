from typing import List, Any, Optional
from functools import wraps
import logging
import torch

from mindspeed.fsdp.utils.log import print_rank
from mindspeed.fsdp.utils.str_match import module_name_match


# Create a logger instance for this module
logger = logging.getLogger(__name__)


def get_chunkmbs_modules(modules, plan):
    """
    Retrieve modules from a model whose names match a specified plan pattern.

    Args:
        modules (nn.Module): The parent module to search within.
        plan (str): The target module name or pattern to match.

    Returns:
        List[Tuple[str, nn.Module]]: A list of (name, module) pairs that matched the plan.

    Raises:
        RuntimeError: If no modules match the given plan name.
    """
    matched_modules = []
    for plan_name in plan:
        for name, module in modules.named_modules():
            if module_name_match(plan_name, name):
                matched_modules.append((name, module))

    if len(matched_modules) == 0:
        raise RuntimeError(f'[ChunkMBS] No module named {plan}.')

    return matched_modules


def apply_chunkmbs_module(chunk_mbs_modules, chunkmbs_cfg):
    """
    Apply the ChunkMBS micro-batching wrapper to a list of modules.

    This function monkey-patches the 'forward' method of the target modules.
    It wraps the original forward pass with a decorator that splits the batch
    into smaller micro-batches.

    Args:
        chunk_mbs_modules (List[Tuple[str, nn.Module]]): List of modules to modify.
        chunkmbs_cfg (object): Configuration object containing chunking parameters.
    """
    for name, module in chunk_mbs_modules:
        # Log which module is being modified (only on the main rank)
        print_rank(logger.info, f'Applying chunkmbs to module: {name}')

        # Replace the forward method with the wrapped version
        module.forward = chunk_mbs_forward(
            chunk_mbs=chunkmbs_cfg.chunk_mbs,
            batch_dim=chunkmbs_cfg.batch_dim,
            chunk_arg_indexs=chunkmbs_cfg.chunk_arg_indexs,
            chunk_kwarg_names=chunkmbs_cfg.chunk_kwarg_names,
        )(module.forward)


def _slice_batch_recursive(
    data: Any,
    start: int,
    end: int,
    batch_dim: int = 0
) -> Any:
    """
    Recursively slice tensors within a nested data structure along the batch dimension.

    This utility handles complex input/output structures (tuples, lists, dicts).
    Non-tensor types (int, str, None, etc.) are returned unchanged.

    Args:
        data: The input data (Tensor, list, tuple, dict, or primitive).
        start (int): Start index for slicing.
        end (int): End index for slicing.
        batch_dim (int): The dimension along which to slice tensors.

    Returns:
        Sliced data with the same structure as input.
    """
    if isinstance(data, torch.Tensor):
        # Create slice object for all dimensions, modifying only batch_dim
        slices = [slice(None)] * data.ndim
        slices[batch_dim] = slice(start, end)
        return data[tuple(slices)]

    elif isinstance(data, (tuple, list)):
        # Recursively slice each item in the sequence
        return type(data)(
            _slice_batch_recursive(item, start, end, batch_dim)
            for item in data
        )

    elif isinstance(data, dict):
        # Recursively slice each value in the dictionary
        return {
            key: _slice_batch_recursive(value, start, end, batch_dim)
            for key, value in data.items()
        }

    else:
        # Return non-container, non-tensor types unchanged
        return data


def chunk_mbs_forward(
    chunk_mbs: int = 1,
    batch_dim: int = 0,
    chunk_arg_indexs: Optional[List[int]] = None,
    chunk_kwarg_names: Optional[List[str]] = None
):
    """
    Decorator factory to enable chunk Micro-Batch on a forward pass.

    This decorator splits a large input batch into smaller micro-batches.
    It processes them sequentially and concatenates the results.

    Args:
        chunk_mbs (int): Micro-batch size (default: 1).
        batch_dim (int): Dimension of the batch in the tensor (default: 0).
        chunk_arg_indexs (List[int]): Indices of positional args to chunk.
        chunk_kwarg_names (List[str]): Names of keyword args to chunk.

    Returns:
        Callable: A decorator that wraps a forward function.
    """
    def decorator(forward_func):
        @wraps(forward_func)
        def wrapper(*args, **kwargs):
            # --- Determine Batch Size ---
            # Try to infer the full batch size from the first specified input tensor
            if chunk_arg_indexs and len(chunk_arg_indexs) > 0:
                full_batch_size = args[chunk_arg_indexs[0]].shape[batch_dim]
            elif chunk_kwarg_names and len(chunk_kwarg_names) > 0:
                full_batch_size = kwargs[chunk_kwarg_names[0]].shape[batch_dim]
            else:
                raise ValueError("No tensor input found to infer batch size.")

            # --- Skip if Batch is Small ---
            if full_batch_size <= chunk_mbs:
                # If the input is smaller than the micro-batch, run normally
                return forward_func(*args, **kwargs)

            else:
                # --- Process Micro-Batches ---
                # Calculate the number of micro-batches needed
                num_micros = (full_batch_size + chunk_mbs - 1) // chunk_mbs
                outputs = []

                for i in range(num_micros):
                    start = i * chunk_mbs
                    end = min(start + chunk_mbs, full_batch_size)

                    # Prepare micro-batch for positional arguments
                    micro_args = []
                    for arg_idx, arg in enumerate(args):
                        if arg_idx in chunk_arg_indexs:
                            # Slice tensor arguments
                            micro_args.append(_slice_batch_recursive(arg, start, end, batch_dim))
                        else:
                            # Pass non-tensor arguments unchanged
                            micro_args.append(arg)

                    # Prepare micro-batch for keyword arguments
                    micro_kwargs = {}
                    for kwarg_name, kwarg_value in kwargs.items():
                        if kwarg_name in chunk_kwarg_names:
                            micro_kwargs[kwarg_name] = _slice_batch_recursive(kwarg_value, start, end, batch_dim)
                        else:
                            micro_kwargs[kwarg_name] = kwarg_value

                    # Execute the forward pass on the micro-batch
                    out = forward_func(*micro_args, **micro_kwargs)
                    outputs.append(out)

                # --- Concatenate Results ---
                # Handle different output types (Tensor, Tuple/List, or unsupported)
                if isinstance(outputs[0], torch.Tensor):
                    return torch.cat(outputs, dim=batch_dim)
                elif isinstance(outputs[0], (tuple, list)):
                    # Concatenate each element of the sequence separately
                    return type(outputs[0])(
                        torch.cat([out[i] for out in outputs], dim=batch_dim)
                        for i in range(len(outputs[0]))
                    )
                else:
                    raise TypeError(f"Unsupported output type: {type(outputs[0])}")

        return wrapper

    return decorator
