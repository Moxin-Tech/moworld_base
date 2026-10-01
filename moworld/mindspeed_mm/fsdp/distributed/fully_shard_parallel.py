# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import logging
from typing import Set, List, Any, Dict, Optional, Union
from collections import OrderedDict

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard, CPUOffloadPolicy
from torch.nn.parallel import DistributedDataParallel as DDP

from mindspeed.fsdp.utils.log import print_rank
from mindspeed.fsdp.utils.str_match import module_name_match
from mindspeed.fsdp.parallel_engine_config import EPPlanConfig
from mindspeed_mm.fsdp.distributed.parallel_state import get_parallel_state
from mindspeed_mm.fsdp.params.parallel_args import FSDPPlanConfig
from mindspeed_mm.fsdp.utils.device import get_torch_device, get_device_type
from mindspeed_mm.fsdp.utils.dtype import get_dtype
from mindspeed_mm.fsdp.params.training_args import TrainingArguments


logger = logging.getLogger(__name__)


def pregather_fsdp_params(model: torch.nn.Module):
    """
    Pre-gather FSDP2 parameters before forward pass.
    This ensures all ranks have parameters ready before timed computation,
    reducing straggler effects caused by uneven allGather times.

    Args:
        model: The model with FSDP2 applied modules.
    """
    for name, module in model.named_modules():
        if hasattr(module, 'unshard') and callable(getattr(module, 'unshard')):
            try:
                module.unshard()
            except Exception as e:
                logging.debug("Failed to unshard module %s: %s", name, e)
    get_torch_device().synchronize()


def fully_shard_parallel_modules(model: torch.nn.Module, fsdp_mesh: DeviceMesh, fsdp_plan: FSDPPlanConfig, training_config: TrainingArguments, **kwargs):
    """
    Apply Fully Sharded Data Parallelism (FSDP) to specified modules in the model.

    Args:
        model: The neural network model to apply FSDP to.
        fsdp_mesh: Device mesh defining the FSDP process group.
        fsdp_plan: Configuration specifying which modules to apply FSDP to and mixed precision settings.
        **kwargs: Additional keyword arguments.

    Returns:
        The model with FSDP applied to specified modules.
    """

    ps = get_parallel_state()

    if ps.fully_shard_parallel_size == 1 and not training_config.init_model_with_meta_device:
        # Background: In DDP mode, model is loaded in float32 by default.
        # Qwen3.5's ChunkGatedDeltaRuleFunction requires bfloat16, so we need to
        # convert the model to the target dtype specified in fsdp_plan.
        target_dtype = get_dtype(fsdp_plan.param_dtype) if fsdp_plan.param_dtype else None
        if target_dtype is not None:
            for name, param in model.named_parameters():
                if "lora" not in name:
                    param.data = param.data.to(dtype=target_dtype)

        # wrap model in DDP
        dp_group = ps.get_dp_group()
        model = DDP(
            model.to(get_device_type()),
            process_group=dp_group,
            find_unused_parameters=True,
            device_ids=[get_torch_device()],
        )

        print_rank(logger.info,
                   "DDP mode is enabled (fully_shard_parallel_size=1) instead of FSDP wrapping")
        return model

    if hasattr(model, 'fully_shard') and callable(getattr(model, 'fully_shard')):
        execute_result = model.fully_shard(fsdp_plan=fsdp_plan)
        if execute_result:
            return model

    # Get modules and parameters that should be ignored for FSDP
    ignored_modules, ignored_params = get_ignored_modules(model, fsdp_plan)
    # Get modules that should have FSDP applied
    fsdp_modules = get_fsdp_modules(model, fsdp_plan, ignored_modules)
    # Get modules that FSDP hook add
    hook_modules = get_fsdp_hook_modules(model, fsdp_plan)

    # Configure mixed precision if enabled
    cpu_offload = None
    if fsdp_plan.cpu_offload:
        cpu_offload = CPUOffloadPolicy(pin_memory=True)
    config = {'mesh': fsdp_mesh, 'ignored_params': ignored_params, "reshard_after_forward": fsdp_plan.reshard_after_forward, "offload_policy": cpu_offload}
    config["mp_policy"] = get_mixprecision_policy(fsdp_plan)
    # Apply FSDP to specific child modules first
    for module in fsdp_modules:
        hook_module = find_hook_module(module, hook_modules)
        if hook_module is None:
            fully_shard(module, **config)
        else:
            fully_shard(module, hook_module=hook_module, **config)
    # Apply FSDP to the entire model
    fully_shard(model, **config)

    return model


def get_mixprecision_policy(fsdp_plan: FSDPPlanConfig):
    """Construct the MixedPrecisionPolicy object."""
    param_dtype = get_dtype(fsdp_plan.param_dtype) if fsdp_plan.param_dtype else None
    reduce_dtype = get_dtype(fsdp_plan.reduce_dtype) if fsdp_plan.reduce_dtype else None
    output_dtype = get_dtype(fsdp_plan.output_dtype) if fsdp_plan.output_dtype else None

    return MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=output_dtype,
        cast_forward_inputs=fsdp_plan.cast_forward_inputs
    )


def _post_order_traverse(model: torch.nn.Module, parent_path: str = ""):
    """
    Perform post-order traversal of model submodules.

    Post-order traversal ensures child modules are visited before their parents,
    which is important for FSDP to properly handle nested modules.

    Args:
        model: The model to traverse.
        parent_path: The path to the current module in the hierarchy.

    Yields:
        Tuple of (module_path, module) for each module in the model.
    """
    for name, child in model.named_children():
        child_path = f"{parent_path}.{name}" if parent_path else name
        yield from _post_order_traverse(child, child_path)
    yield parent_path, model


def get_fsdp_modules(model: torch.nn.Module, fsdp_plan: FSDPPlanConfig, ignored_modules: Set[str]) -> List[Any]:
    fsdp_modules = []
    if fsdp_plan.apply_modules is None:
        return fsdp_modules
    # Traverse all modules in the model
    if fsdp_plan.apply_modules:
        for name, module in _post_order_traverse(model):
            # Check if module matches any pattern in the FSDP plan
            for pattern in fsdp_plan.apply_modules:
                if module_name_match(pattern, name) and name not in ignored_modules:
                    print_rank(logger.debug, f'[FSDP2]: Apply fsdp2 to module <{name}>')
                    fsdp_modules.append(module)
        # Ensure at least one module matches the FSDP plan
        if len(fsdp_modules) == 0:
            raise RuntimeError(f'[FSDP2] No module named {fsdp_plan.apply_modules}.')
    return fsdp_modules


def get_fsdp_hook_modules(model: torch.nn.Module, fsdp_plan: FSDPPlanConfig) -> List[Any]:
    fsdp_hook_modules = []
    if fsdp_plan.apply_modules is None:
        return fsdp_hook_modules

    # Traverse all modules in the model
    if fsdp_plan.hook_modules:
        for name, module in _post_order_traverse(model):
            # Check if module matches any pattern in the FSDP plan
            for pattern in fsdp_plan.hook_modules:
                if module_name_match(pattern, name):
                    print_rank(logger.debug, f'[FSDP2]: Apply fsdp2 hook to hook_module <{name}>')
                    fsdp_hook_modules.append(module)
        # Ensure at least one module matches the FSDP plan
        if len(fsdp_hook_modules) == 0:
            raise RuntimeError(f'[FSDP2] No module named {fsdp_plan.hook_modules}.')

    return fsdp_hook_modules


def get_efsdp_modules(modules: torch.nn.Module, plan: EPPlanConfig):
    efsdp_modules = []
    for plan_name in plan.apply_efsdp_modules:
        for name, module in modules.named_modules():
            if module_name_match(plan_name, name):
                print_rank(logger.debug, f'[Expert Fully Shard]: Apply efsdp to module <{name}>')
                efsdp_modules.append(module)
    if len(efsdp_modules) == 0:
        raise RuntimeError(f'[Expert Fully Shard] No module named {plan} or not be ModuleList')
    return efsdp_modules


def find_hook_module(target_module: torch.nn.Module, hook_module_list: List[torch.nn.Module]) -> Optional[torch.nn.Module]:
    for hook_module in hook_module_list:
        for _, sub_mod in hook_module.named_modules():
            if sub_mod is target_module:
                return hook_module
    return None


def get_ignored_modules(model: torch.nn.Module, fsdp_plan: FSDPPlanConfig):
    ignored_modules = set()
    ignored_params = set()
    if fsdp_plan.ignored_modules is None:
        return ignored_modules, ignored_params
    for name, module in model.named_modules():
        for pattern in fsdp_plan.ignored_modules:
            if module_name_match(pattern, name):
                print_rank(logger.debug, f'[FSDP2]: Ignored module to apply fsdp2 <{name}>')
                ignored_modules.add(name)
                ignored_params.update(list(module.parameters(recurse=True)))
    return ignored_modules, ignored_params


def _match_pattern_with_reversed_order(patterns: List[str], name: str) -> int:
    """
    Iterates through the pattern list in reverse order to find a match for the given name
    and return its original forward index.

    This function is typically used in FSDP (Fully Sharded Data Parallel) prefetch settings.
    It searches from the end of the list in the beginning, and implies that patterns at the end
    of the list may have higher priority.

    Args:
        patterns (List[str]): A list of string patterns to match against the module name.
        name (str): The module name to be matched.

    Returns:
        int: The index of the matched pattern in the original `patterns` list.

    Raises:
        RuntimeError: If no matching pattern is found after checking the entire list.
    """

    patterns_num = len(patterns)
    for reversed_order_id, pattern in enumerate(reversed(patterns)):
        if module_name_match(pattern, name):
            return patterns_num - reversed_order_id - 1
    raise RuntimeError(f"Cannot find parent module for module '{name}' in FSDP prefetch setting patterns: {patterns}")


def _get_layer_path(model: torch.nn.Module, target_layer) -> Optional[str]:
    """
    Retrieves the full path (name) of a specific target layer within the model.

    This function traverses all named modules in the model to locate the target layer
    by object identity and returns its dot-separated path string.

    Args:
        model (torch.nn.Module): The top-level model to search within.
        target_layer (torch.nn.Module): The specific module instance to find.

    Returns:
        str | None: The path of the target layer (e.g., 'layer1.conv1') if found,
                    otherwise None.
    """

    for name, module in model.named_modules():
        if module is target_layer:
            return name
    return None


def _is_submodule(child: torch.nn.Module, parent: torch.nn.Module) -> bool:
    return any(m is child for m in parent.modules())


def _order_sub_modules_by_hierarchy(sub_modules: List[Union[torch.nn.Module, List[torch.nn.Module]]], parent_first: bool = False) -> List[Union[torch.nn.Module, List[torch.nn.Module]]]:
    """
    Reorder a list of sub-modules based on their hierarchical relationship.

    This function supports sorting elements that are either single Modules or
    Lists of Modules. If an element is a list, the first item (index 0) is used
    for hierarchy comparison.

    Args:
        sub_modules: A list of torch modules (or lists of modules) to be ordered.
        parent_first: A boolean flag to control sort direction.
                      - If True: Parents are placed before children (Parent -> Child).
                      - If False: Children are placed before parents (Child -> Parent).

    Returns:
        The list of modules sorted according to the specified hierarchy.
    """

    # Create a copy of the list to avoid modifying the original input
    ordered_sub_modules = sub_modules.copy()
    n = len(sub_modules)

    # Bubble sort to order modules by hierarchy
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            curr_mod = ordered_sub_modules[j]
            next_mod = ordered_sub_modules[j + 1]

            curr_compare = curr_mod[0] if isinstance(curr_mod, list) else curr_mod
            next_compare = next_mod[0] if isinstance(next_mod, list) else next_mod

            should_swap = False

            if parent_first:
                # Target: Parent -> Child
                # If 'curr' is a child of 'next', then 'curr' should come after 'next'.
                # So we swap to move 'next' (parent) to the left.
                if _is_submodule(curr_compare, next_compare):
                    should_swap = True
            else:
                # Target: Child -> Parent
                # If 'next' is a child of 'curr', then 'next' should come before 'curr'.
                # So we swap to move 'next' (child) to the left.
                if _is_submodule(next_compare, curr_compare):
                    should_swap = True

            if should_swap:
                ordered_sub_modules[j], ordered_sub_modules[j + 1] = ordered_sub_modules[j + 1], ordered_sub_modules[j]
                swapped = True

        # Optimization: if no swaps occurred in a pass, the list is already sorted
        if not swapped:
            break
    return ordered_sub_modules


def set_modules_to_prefetch(
    model: torch.nn.Module,
    fsdp_plan: FSDPPlanConfig,
    ep_plan: Optional[EPPlanConfig] = None
):
    """
    Configure forward and backward prefetching.

    This function automatically determines the module execution order based on
    `fsdp_plan.apply_modules` and sets up prefetching accordingly.

    Note:
    1. This interface is not very generic. For high-performance prefetching requirements,
    it is strongly recommended to implement a custom `set_modules_to_prefetch` interface in the model.
    2. If you use the automatic setup method, it is recommended to check the relevant settings in the logs.

    e.g.
    fsdp_plan
    - apply_modules (must in order):
        - model.visual
        - model.visual.blocks.{*}
        - model.language_model
        - model.language_model.embed_tokens
        - model.language_model.layers.{*}
        - model.language_model.layers.{*}.attn
        - lm_head

    ep_plan
    - apply_modules:
        - model.language_model.layers.{*}.mlp.experts

    The setting result is:
    [forward]:
    model.visual -> model.visual.blocks[0]
    model.visual.blocks[i] -> model.visual.blocks[i+1]
    model.visual.blocks[-1] -> model.language_model
    model.language_model -> model.language_model.embed_tokens
    model.language_model.embed_tokens -> [model.language_model.layers[0], model.language_model.layers[0].attn, model.language_model.layers[0].mlp.experts]
    model.language_model.layers[i].mlp.experts -> [model.language_model.layers[i+1], model.language_model.layers[i+1].attn, model.language_model.layers[i+1].mlp.experts]
    model.language_model.layers[-1].mlp.experts -> lm_head

    [backward]:
    lm_head -> model.language_model
    model.language_model -> [model.language_model.layers[-1], model.language_model.layers[-1].attn, model.language_model.layers[-1].mlp.experts]
    model.language_model.layers[i] -> [model.language_model.layers[i-1], model.language_model.layers[i-1].attn, model.language_model.layers[i-1].mlp.experts]
    model.language_model.layers[0] -> model.language_model.embed_tokens
    model.language_model.embed_tokens -> model.visual
    model.visual -> model.visual_blocks[-1]
    model.visual_blocks[i] -> model.visual_blocks[i-1]
    """

    if hasattr(model, 'set_modules_to_prefetch') and callable(getattr(model, 'set_modules_to_prefetch')):
        execute_result = model.set_modules_to_prefetch(fsdp_plan=fsdp_plan, ep_plan=ep_plan)
        if execute_result:
            return model

    # Get all modules that need to be wrapped by FSDP based on the plan
    ignore_modules, _ = get_ignored_modules(model, fsdp_plan)
    fsdp_modules = get_fsdp_modules(model, fsdp_plan, ignore_modules)
    # Get modules that have explicit hook points (used to determine prefetch boundaries)
    hook_modules_in_order = get_fsdp_hook_modules(model, fsdp_plan)
    # Initialize a list of OrderedDicts to group modules by their application order
    wrapped_modules: List[Dict[torch.nn.Module]] = [OrderedDict() for _ in range(len(fsdp_plan.apply_modules))] # [hook_module/default_hook_module: List(torch.nn.Module)]
    # Get E-FSDP modules if EP plan is provided
    efsdp_modules = get_efsdp_modules(model, ep_plan) if ep_plan else []

    order_num = 0
    # --- Phase 1: Traverse the model to group modules by their hook/order ---
    for name, sub_module in model.named_modules():
        # Handle Hook Modules: If this module is a designated hook point
        if any(sub_module is target_module for target_module in hook_modules_in_order):
            # Find the reverse order index based on pattern matching
            forward_order = _match_pattern_with_reversed_order(fsdp_plan.apply_modules, name)

            # Register the hook module if not already present
            if sub_module not in wrapped_modules[forward_order].keys():
                wrapped_modules[forward_order][sub_module] = []
                order_num += 1

            # Special handling for E-FSDP (Expert Parallelism)
            # Only effective if explicit hook_module is set to avoid memory overhead
            for efsdp_module in efsdp_modules:
                hook_module = find_hook_module(efsdp_module, hook_modules_in_order)
                # Associate E-FSDP module with this hook point
                if hook_module is sub_module:
                    wrapped_modules[forward_order][sub_module].append(efsdp_module)

        # Handle General FSDP Modules
        if any(sub_module is target_module for target_module in fsdp_modules):
            hook_module = find_hook_module(sub_module, hook_modules_in_order)
            if hook_module:
                # If an explicit hook_module is set, follow its order
                forward_order = 0
                for i, wrapped_module_order_dict in enumerate(wrapped_modules):
                    if hook_module in wrapped_module_order_dict.keys():
                        forward_order = i
                        break

            else:
                # Default behavior: use pattern matching to determine order
                hook_module = sub_module
                forward_order = _match_pattern_with_reversed_order(fsdp_plan.apply_modules, name)
                if hook_module not in wrapped_modules[forward_order]:
                    wrapped_modules[forward_order][hook_module] = []
                    order_num += 1

            # Add the current module to the list under its hook
            wrapped_modules[forward_order][hook_module].append(sub_module)


    # --- Phase 2: Flatten the grouped modules into a strict execution order ---
    # Initialize the final ordered list
    wrapped_modules_in_order: List[List[torch.nn.Module]] = [[] for _ in range(order_num)]
    order_id = 0
    apply_module_num = len(fsdp_plan.apply_modules)

    def _insert_child_modules(curr_order, curr_fsdp_module, order_id):
        """
        Recursively insert child modules that belong to deeper levels of the module hierarchy.
        This ensures that nested FSDP modules are scheduled correctly.
        """
        if curr_order < apply_module_num - 1:
            # Look for modules in subsequent orders that are submodules of the current one
            for search_order in range(curr_order + 1, len(wrapped_modules)):
                if fsdp_plan.apply_modules[curr_order] in fsdp_plan.apply_modules[search_order]:
                    fsdp_module_list = list(wrapped_modules[search_order].keys())
                    for fsdp_module in fsdp_module_list:
                        if _is_submodule(fsdp_module, curr_fsdp_module):
                            wrapped_modules_in_order[order_id] = wrapped_modules[search_order][fsdp_module]
                            order_id += 1
                            # Recurse to find deeper children
                            order_id = _insert_child_modules(search_order, fsdp_module, order_id)
                            # Remove processed module to avoid duplication
                            wrapped_modules[search_order].pop(fsdp_module)
        return order_id

    # Build the final ordered list by iterating through the grouped modules
    for apply_match_order in range(apply_module_num):
        wrapped_module_order_dict = wrapped_modules[apply_match_order]
        for hook_module, modules in wrapped_module_order_dict.items():
            wrapped_modules_in_order[order_id] = modules
            order_id += 1
            order_id = _insert_child_modules(apply_match_order, hook_module, order_id)

    # --- Phase 3: Configure Forward Prefetching ---
    if fsdp_plan.num_to_forward_prefetch > 0:
        for i, layer_modules in enumerate(wrapped_modules_in_order):
            # Determine the range of modules to prefetch
            j_end = min(len(wrapped_modules_in_order), i + 1 + fsdp_plan.num_to_forward_prefetch)
            layers_to_prefetch = wrapped_modules_in_order[i + 1: j_end]
            if layers_to_prefetch:
                # Flatten the list of modules from the prefetch layers
                modules_to_prefetch = [module_to_prefetch for layer_modules in layers_to_prefetch for module_to_prefetch in layer_modules]

                # Sort to find the first FSDP module that will be executed in this group
                layer_modules = _order_sub_modules_by_hierarchy(layer_modules)
                # Set the prefetch modules on the first module in the current group
                layer_modules[0].set_modules_to_forward_prefetch(modules_to_prefetch)

                # Logging: Print the prefetch configuration
                print_rank(
                    logger.info,
                    f"{_get_layer_path(model, layer_modules[0])} set forward prefetch: {[_get_layer_path(model, module_to_prefetch) for module_to_prefetch in modules_to_prefetch]}"
                )

    # --- Phase 4: Configure Backward Prefetching ---
    if fsdp_plan.num_to_backward_prefetch > 0:
        # Reverse the order for backward pass
        rev_wrapped_modules_in_order = list(reversed(wrapped_modules_in_order))
        rev_wrapped_modules_in_order = _order_sub_modules_by_hierarchy(rev_wrapped_modules_in_order, parent_first=True)
        for i, layer_modules in enumerate(rev_wrapped_modules_in_order):
            # Determine the range for backward prefetch
            j_end = min(len(rev_wrapped_modules_in_order), i + 1 + fsdp_plan.num_to_backward_prefetch)
            layers_to_prefetch = rev_wrapped_modules_in_order[i + 1: j_end]
            if layers_to_prefetch:
                # Flatten the list
                modules_to_prefetch = [module_to_prefetch for layer_modules in layers_to_prefetch for module_to_prefetch in layer_modules]
                # Sort to find the last FSDP module that was executed in this group
                layer_modules = _order_sub_modules_by_hierarchy(layer_modules)
                # Set the prefetch modules on the last module in the current group
                layer_modules[-1].set_modules_to_backward_prefetch(modules_to_prefetch)

                # Logging: Print the prefetch configuration
                print_rank(
                    logger.info,
                    f"{_get_layer_path(model, layer_modules[-1])} set backward prefetch: {[_get_layer_path(model, module_to_prefetch) for module_to_prefetch in modules_to_prefetch]}"
                )

    return model
