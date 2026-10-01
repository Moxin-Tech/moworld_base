from typing import Optional, Iterable, Union
import copy
import dataclasses
import re
import types
import functools
from functools import partial

import yaml
import torch
import torch.nn as nn
from torch.distributed._tensor import Shard, Replicate, DTensor
from torch.distributed.fsdp import fully_shard, CPUOffloadPolicy, OffloadPolicy
from torch.distributed.fsdp._fully_shard._fsdp_init import _get_device_from_mesh
from torch.distributed import DeviceMesh
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing
)
from torch.distributed import ProcessGroup, get_process_group_ranks
from torch.distributed.fsdp import MixedPrecisionPolicy
from megatron.training import get_args
from megatron.training.utils import unwrap_model

from mindspeed.utils import _get_dtype
from mindspeed_mm.models.transformers.global_vars import (
    set_ep_size,
    set_ep_rank,
    set_check_moe_func,
    set_ep_group,
    set_ep_fsdp_group,
    get_ep_size,
    get_check_moe_func
)


def get_nested_module(model, path):
    """
    Get nested module in PyTorch model by dot-separated path.

    Args:
        model: PyTorch model instance
        path: Dot-separated path string, e.g., "model.visual.blocks.0"

    Returns:
        Found module object, or None if path doesn't exist
    """
    if path == "":
        return model

    modules = path.split('.')
    current_module = model
    for module_name in modules:
        if hasattr(current_module, module_name):
            current_module = getattr(current_module, module_name)
        else:
            return None
    return current_module


def expand_wildcard_pattern(model, module_path):
    """
    Expand wildcard pattern {*} in module path based on actual model structure.

    Args:
        model: The target model
        module_path: Module path string containing {*} pattern

    Returns:
        List of expanded module paths
    """
    wildcard_pattern = r'\{\*\}'
    re_match = re.search(wildcard_pattern, module_path)
    if not re_match:
        return [module_path]

    wildcard_start = re_match.start()
    wildcard_end = re_match.end()

    # Get path before {*}
    base_path = module_path[:wildcard_start].rstrip('.')
    remaining_path = module_path[wildcard_end:].lstrip('.')

    base_module = get_nested_module(model, base_path)
    expanded_paths = []
    if base_module is not None and hasattr(base_module, '__len__'):
        total = len(base_module)
        for idx in range(total):
            new_path = f"{base_path}.{idx}"
            if remaining_path:
                new_path += f".{remaining_path}"
            expanded_paths.append(new_path)
    else:
        raise ValueError(f"Module at path '{base_path}' is None or does not have length attribute")

    return expanded_paths


def expand_range_pattern(module_path):
    """
    Expand range pattern like {0-20,25,30-40} in module path.

    Args:
        module_path: Module path string containing range pattern

    Returns:
        List of expanded module paths
    """
    pattern_regex = re.compile(r'^(.*?){(.*?)}(.*)$')
    re_match = pattern_regex.match(module_path)

    before_pattern = re_match.group(1).rstrip('.')
    pattern_content = re_match.group(2)
    after_pattern = re_match.group(3).lstrip('.')

    indices = set()
    parts = pattern_content.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            # Handle range like 0-20
            start, end = part.split('-')
            start = int(start.strip())
            end = int(end.strip())
            indices.update(range(start, end + 1))
        else:
            # Single number
            indices.add(int(part))
    indices = [str(i) for i in sorted(indices)]

    expanded_paths = []
    for idx in indices:
        new_path = f"{before_pattern}.{idx}"
        if after_pattern:
            new_path += f".{after_pattern}"
        expanded_paths.append(new_path)

    return expanded_paths


def _validate_pattern(module_path):
    """
    Validate pattern string to ensure at most one pair of braces {}.

    Args:
        module_path: Module path string to validate
    """
    start_count = module_path.count("{")
    end_count = module_path.count("}")

    if start_count > 1 or end_count > 1:
        raise ValueError(f"Configuration string '{module_path}' contains multiple brace pairs. Only one pair is allowed per line")

    if start_count != end_count:
        raise ValueError(f"Configuration string '{module_path}' has mismatched braces")

    if start_count == 1:
        start_idx = module_path.find('{')
        end_idx = module_path.find('}')
        if end_idx <= start_idx:
            raise ValueError(f"Configuration string '{module_path}' has invalid brace order")


def get_submodules_by_path(model, config):
    """
    Get submodules from model based on configuration paths.

    Args:
        model: The target model
        config: Configuration list containing module paths with patterns

    Returns:
        List of unique submodules in order of appearance
    """
    if config is None:
        return None
    model = unwrap_model(model)

    submodules = []

    for module_path in config:
        _validate_pattern(module_path)

        # Handle simple paths without patterns
        if "{" not in module_path and "}" not in module_path:
            target_module = get_nested_module(model, module_path)
            if target_module is not None:
                submodules.append(target_module)
            else:
                print(f"Warning: Module not found at path '{module_path}'")
        else:
            if "{*}" in module_path:
                expanded_paths = expand_wildcard_pattern(model, module_path)
            else:
                expanded_paths = expand_range_pattern(module_path)

            # Get corresponding modules
            for path in expanded_paths:
                target_module = get_nested_module(model, path)
                if target_module is not None:
                    submodules.append(target_module)
                else:
                    print(f"Warning: Module not found at expanded path '{path}'")

    # Remove duplicates while preserving order
    unique_submodules = []
    seen_modules = set()
    for module in submodules:
        module_id = id(module)
        if module_id not in seen_modules:
            seen_modules.add(module_id)
            unique_submodules.append(module)

    return unique_submodules


@dataclasses.dataclass
class Fsdp2Config:
    sharding_size: Optional[int] = None
    sub_modules_to_wrap: Optional[Iterable[torch.nn.Module]] = None
    reshard_after_forward: Union[bool, int] = True
    # mp_policy
    param_dtype: Optional[torch.dtype] = None
    reduce_dtype: Optional[torch.dtype] = None
    output_dtype: Optional[torch.dtype] = None
    cast_forward_inputs: bool = True

    # offload
    offload_to_cpu: bool = False
    pin_memory: bool = True # pin_memory is effective exclusively when offload_to_cpu is True

    # prefetch setting
    num_to_forward_prefetch: Optional[int] = 0
    num_to_backward_prefetch: Optional[int] = 0

    ignored_modules: Optional[Iterable[torch.nn.Module]] = None

    recompute_modules: Optional[Iterable[torch.nn.Module]] = None
    use_reentrant: bool = True

    # If True, each FSDP parameter group within a block contains only Linear layer parameters,
    # enabling aligned sharding for improved communication efficiency. Set to False to disable this optimization.
    align_fsdp_param_groups: bool = False

    expert_parallel_size: Optional[int] = 1
    reshard_local_experts: Union[bool, int] = True
    moe_modules: Optional[Iterable[torch.nn.Module]] = None

    def to_dict(self):
        mp_policy = self._mp_policy()
        offload_policy = None
        if self.offload_to_cpu:
            offload_policy = CPUOffloadPolicy(pin_memory=self.pin_memory)
        else:
            offload_policy = OffloadPolicy()  # means no offloading

        kwargs = {
            "mp_policy": mp_policy,
            "reshard_after_forward": self.reshard_after_forward,
            "offload_policy": offload_policy,
        }
        return kwargs

    def _mp_policy(self):
        param_dtype = _get_dtype(self.param_dtype) if self.param_dtype else None
        reduce_dtype = _get_dtype(self.reduce_dtype) if self.reduce_dtype else None
        output_dtype = _get_dtype(self.output_dtype) if self.output_dtype else None
        return MixedPrecisionPolicy(param_dtype=param_dtype,
                                    reduce_dtype=reduce_dtype,
                                    output_dtype=output_dtype,
                                    cast_forward_inputs=self.cast_forward_inputs)

    @classmethod
    def load_from_yaml(cls, yml_file: str):
        with open(yml_file, 'r') as f:
            config = yaml.safe_load(f)
        kwargs = {}
        for f in dataclasses.fields(cls):
            if f.name in config:
                kwargs[f.name] = config[f.name]
        return cls(**kwargs)


def _create_device_mesh(sharding_size: Optional[int], process_group: ProcessGroup) -> DeviceMesh:
    """
    Create a DeviceMesh for FSDP (Fully Sharded Data Parallel).

    Args:
        sharding_size (int): Number of processes in each FSDP group (sharding dimension)
        process_group (ProcessGroup): The process group containing all participating ranks

    Returns:
        DeviceMesh: A 1D or 2D device mesh for parallel training
    """
    if sharding_size == "auto":
        sharding_size = torch.distributed.get_world_size(process_group)
    elif sharding_size is None:
        sharding_size = 1
    # Get total number of processes in the group
    world_size = torch.distributed.get_world_size(process_group)

    # Get global ranks of all processes in this group
    group_global_ranks = torch.tensor(
        get_process_group_ranks(process_group),
        device="cpu",
        dtype=torch.int
    )

    # Calculate DDP group size (data parallel dimension)
    replicating_size = world_size // sharding_size

    # Validate configuration
    if replicating_size * sharding_size != world_size:
        raise ValueError(
            f"World size {world_size} must be divisible by sharding_size {sharding_size}. "
            f"Current configuration would leave {world_size % sharding_size} ranks unassigned."
        )

    # Create 1D mesh (FSDP-only) or 2D mesh (FSDP+DDP hybrid)
    if replicating_size == 1:
        # Pure FSDP case - single dimension mesh
        mesh = group_global_ranks
        device_mesh = DeviceMesh.from_group(
            process_group,
            "npu", # NPU device type (change to "cuda" for GPUs)
            mesh_dim_names=["Shard"]
        )
    else:
        # Hybrid FSDP+DDP case - two dimensional mesh
        mesh = group_global_ranks.view(replicating_size, sharding_size)
        device_mesh = DeviceMesh(
            "npu",
            mesh,
            mesh_dim_names=["Replicate", "Shard"]  # [data_parallel, model_sharding]
        )

    return device_mesh


def initialize_fsdp2_config(fsdp2_config_path, module, process_group):
    """Initialize and configure FSDP2 settings.

    Args:
        fsdp2_config_path (str): Path to the FSDP2 configuration YAML file.
        module (torch.nn.Module): The neural network module to be wrapped with FSDP2.
        process_group: Process group for distributed communication.

    Returns:
        tuple: A tuple containing:
            - fsdp2_config: The loaded FSDP2 configuration object
            - fsdp2_kwargs (dict): Dictionary of fully_shard parameters
    """
    fsdp2_kwargs = {}

    if fsdp2_config_path:
        fsdp2_config = Fsdp2Config.load_from_yaml(fsdp2_config_path)
        fsdp2_kwargs.update(fsdp2_config.to_dict())
    else:
        # Use default configuration
        fsdp2_config = Fsdp2Config()
        fsdp2_kwargs.update(fsdp2_config.to_dict())

    device_mesh = _create_device_mesh(fsdp2_config.sharding_size, process_group)
    fsdp2_kwargs["mesh"] = device_mesh

    # Collect ignored parameters
    ignored_params = get_ignored_params(module, device_mesh, fsdp2_config.ignored_modules)
    if ignored_params:
        fsdp2_kwargs["ignored_params"] = ignored_params

    return fsdp2_config, fsdp2_kwargs


def create_ep_device_mesh(ep_size, reshard_local_experts=False):
    ep_device_mesh = None
    unit_device_mesh = None
    if ep_size > 1:
        world_size = torch.distributed.get_world_size()
        if world_size % ep_size != 0:
            raise ValueError(
                f"World size {world_size} must be divisible by expert_parallel_size {ep_size}."
            )
        ep_replicate = world_size // ep_size

        ep_device_mesh = init_device_mesh(
            device_type="npu",
            mesh_shape=[ep_replicate, ep_size],
            mesh_dim_names=["EP_Replicate", "EP"]
        )
        if not reshard_local_experts:
            global_rank = torch.tensor([torch.distributed.get_rank()], device="cpu", dtype=torch.int)
            unit_device_mesh = DeviceMesh("npu", global_rank)
    return ep_device_mesh, unit_device_mesh


def get_moe_modules_and_param_name(module, moe_modules_config):
    moe_modules = get_submodules_by_path(module, moe_modules_config)
    if not moe_modules:
        return [], set()
    moe_param_names = set()
    for module_name, sub_module in module.named_modules():
        if any([sub_module == moe_module for moe_module in moe_modules]):
            for local_name, _ in sub_module.named_parameters(recurse=False):
                moe_param_names.add(f"{module_name}.{local_name}")
    return moe_modules, moe_param_names


def set_module_from_path(model, path, value):
    attrs = path.split(".")
    if len(attrs) == 1:
        setattr(model, attrs[0], value)
    else:
        next_obj = getattr(model, attrs[0])
        set_module_from_path(next_obj, ".".join(attrs[1:]), value)


def check_moe_by_param_name(moe_param_names, param_name):
    recompute_prefix = "_checkpoint_wrapped_module."
    if recompute_prefix in param_name:
        param_name = param_name.replace(recompute_prefix, "")
    return any([param_name.endswith(moe_param) for moe_param in moe_param_names])


def get_ignored_params(module, device_mesh, ignored_modules_config):
    """Identify and collect parameters that should be ignored by FSDP2 sharding.

    Args:
        module: The root module to search for ignored submodules
        device_mesh: The device mesh for distributed training
        ignored_modules_config: Configuration specifying which modules to ignore

    Returns:
        Set of parameters that should be excluded from FSDP2 sharding
    """
    ignored_modules = get_submodules_by_path(module, ignored_modules_config)
    ignored_params = set()
    if ignored_modules:
        for sub_module in module.modules():
            if any(sub_module is target_module for target_module in ignored_modules):
                if not get_args().init_model_with_meta_device:
                    sub_module.to(_get_device_from_mesh(device_mesh))

                ignored_params.update(sub_module.parameters())

    return ignored_params


def set_recompute_modules_to_wrap(module, recompute_modules_config, use_reentrant=True):
    """Apply activation checkpointing to specified modules for memory optimization.

    Args:
        module: The root module to apply activation checkpointing to
        recompute_modules_config: Configuration specifying which modules to checkpoint
    """
    recompute_modules = get_submodules_by_path(module, recompute_modules_config)
    if recompute_modules:
        apply_activation_checkpointing(
            module,
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper, checkpoint_impl=CheckpointImpl.REENTRANT if use_reentrant else CheckpointImpl.NO_REENTRANT
            ),
            check_fn=lambda module: any(module is target_module for target_module in recompute_modules)
        )

    return


def set_fullyshard_modules_to_wrap(module, fullyshard_modules_config, **fsdp2_kwargs):
    """Apply FSDP2 wrapping to specified submodules in post-order traversal.

    Args:
        module: The root module to wrap with FSDP2
        fullyshard_modules_config: Configuration specifying which modules to shard
        **fsdp2_kwargs: Additional fully_shard arguments
    """

    def _post_order_traverse(model: torch.nn.Module):
        """Post-order traversal of model submodules (recursive implementation).

        Yields child modules before their parents.
        """
        for child in model.children():
            yield from _post_order_traverse(child)
        yield model

    sub_modules_to_wrap = get_submodules_by_path(module, fullyshard_modules_config)
    for sub_module in _post_order_traverse(module):
        # Wrap individual submodules to fetch parameters just-in-time rather than
        # conservatively fetching all parameters at the start of each iteration.
        if any(sub_module is target_module for target_module in sub_modules_to_wrap):
            fully_shard(sub_module, **fsdp2_kwargs)

    # Wrap the root module as required by the FSDP API.
    fully_shard(module, **fsdp2_kwargs)

    return


def set_modules_to_prefetch(module, fullyshard_modules_config, num_to_forward_prefetch, num_to_backward_prefetch):
    """Configure forward and backward prefetching for communication-computation overlap.

    Args:
        module: The root module to configure prefetching for
        fullyshard_modules_config: Configuration specifying which modules are sharded
        num_to_forward_prefetch: Number of layers to prefetch during forward pass
        num_to_backward_prefetch: Number of layers to prefetch during backward pass
    """
    sub_modules_to_wrap = get_submodules_by_path(module, fullyshard_modules_config)
    # Pre-order traversal to collect sub-modules for prefetching
    wrapped_modules_in_order: list[torch.nn.Module] = []
    for sub_module in module.modules():# pre-order
        if any(sub_module is target_module for target_module in sub_modules_to_wrap):
            wrapped_modules_in_order.append(sub_module)

    # Configure forward prefetching to overlap communication with computation
    if num_to_forward_prefetch > 0:
        for i, layer in enumerate(wrapped_modules_in_order):
            j_end = min(len(wrapped_modules_in_order), i + 1 + num_to_forward_prefetch)
            layers_to_prefetch = wrapped_modules_in_order[i + 1:j_end]
            if layers_to_prefetch:
                layer.set_modules_to_forward_prefetch(layers_to_prefetch)

    # Configure backward prefetching for gradient communication overlapping
    if num_to_backward_prefetch > 0:
        rev_wrapped_modules_in_order = list(reversed(wrapped_modules_in_order))
        for i, layer in enumerate(rev_wrapped_modules_in_order):
            j_end = min(len(rev_wrapped_modules_in_order), i + 1 + num_to_backward_prefetch)
            layers_to_prefetch = rev_wrapped_modules_in_order[i + 1:j_end]
            if layers_to_prefetch:
                layer.set_modules_to_backward_prefetch(layers_to_prefetch)


class FSDP2Mixin:
    """
    Mixin class for FSDP2 (Fully Sharded Data Parallel v2) functionality.

    Important: Classes using this mixin MUST inherit from torch.nn.Module.

    Example:
        class MyModel(nn.Module, FSDP2Mixin):
            ...
    """
    def __init__(self):
        self.fsdp2_config = None
        self.fsdp2_kwargs = None
        self.ep_device_mesh = None
        self.unit_device_mesh = None

    def freeze(self, config):
        pass

    def post_meta_init(self):
        """
        Hook method called after meta device initialization.

        This method can be overridden by subclasses to perform additional
        initialization steps after weights are loaded from meta device.
        """
        pass

    def _pre_fully_shard(self, process_group, fsdp2_config_path, **kwargs):
        """
        Pre-processing step before applying FSDP2 sharding.

        Args:
            process_group: torch.distributed.ProcessGroup for communication
            fsdp2_config_path: Path to YAML configuration file for FSDP2 settings
            **kwargs: Additional arguments for FSDP2 initialization

        Returns:
            tuple: (fsdp2_kwargs, fsdp2_config) - Configuration parameters for FSDP2
        """
        self.fsdp2_config, self.fsdp2_kwargs = initialize_fsdp2_config(fsdp2_config_path, self, process_group)
        self.ep_device_mesh, self.unit_device_mesh = create_ep_device_mesh(self.fsdp2_config.expert_parallel_size, self.fsdp2_config.reshard_local_experts)

    def to_empty_if_needed(self, *, device: torch.device | str | int | None, recurse: bool = True):
        """Move the parameters and buffers to the specified device without copying storage if they are not already on that device.

        Args:
            module: The module whose parameters and buffers to (maybe) move.
            device: The desired device of the parameters and buffers in the module. If `None`, the default device is used.
            recurse: Whether parameters and buffers of submodules should be recursively moved to the specified device.

        Returns:
            The (maybe) moved module.
        """
        device = torch.empty((), device=device).device

        def _replace_tensor(t):
            if isinstance(t, torch.nn.Parameter):
                return torch.empty_like(t, device=device) if t.device != device else t
            else:
                # we do not offload buffer to cpu when enable FSDP2 offload_to_cpu function.
                return t.to(device=torch.npu.current_device()) if t.device == torch.device('cpu') else t
        return self._apply(_replace_tensor, recurse=recurse)

    def _post_fully_shard(self):
        """
        Post-processing step after FSDP2 sharding is applied.

        Handles meta device initialization, weight initialization for FSDP2 setup.
        """

        # Initialize model with meta device for memory-efficient large model loading
        args = get_args()
        if args.init_model_with_meta_device:
            if self.fsdp2_config.offload_to_cpu:
                self.to_empty_if_needed(device="cpu")
            else:
                self.to_empty_if_needed(device="cuda")

            # Check if the unwrapped model has init_weights method
            if not hasattr(self, 'init_weights') or not callable(self.init_weights):
                raise AttributeError(
                    f"The model {type(self).__name__} does not have an 'init_weights' method. "
                    "This is required when using meta device initialization. "
                    "Please implement an 'init_weights' method in your model class to initialize "
                    "the weights after loading from meta device."
                )

            self.init_weights()
            self.post_meta_init()

        # distinguish between MOE parameters and non-MOE parameters
        if get_ep_size() > 1:
            check_moe_fn = get_check_moe_func()
            for name, param in self.named_parameters():
                if check_moe_fn(name):
                    setattr(param, "allreduce", False)

    def _fully_shard(self, fsdp2_kwargs, fsdp2_config):
        """
        Core FSDP2 sharding logic - applies wrappers and configurations.

        Args:
            fsdp2_kwargs: Keyword arguments for fully_shard
            fsdp2_config: Configuration object containing FSDP2 settings
        """

        # recompute modules to wrap
        set_recompute_modules_to_wrap(self, fsdp2_config.recompute_modules, fsdp2_config.use_reentrant)

        # Apply fsdp2 wrapping to specified sub-modules
        set_fullyshard_modules_to_wrap(self, fsdp2_config.sub_modules_to_wrap, **fsdp2_kwargs)

        # Configure forward and backward prefetching for performance optimization
        num_to_forward_prefetch = getattr(self.fsdp2_config, "num_to_forward_prefetch", 0)
        num_to_backward_prefetch = getattr(self.fsdp2_config, "num_to_backward_prefetch", 0)
        set_modules_to_prefetch(self, self.fsdp2_config.sub_modules_to_wrap, num_to_forward_prefetch, num_to_backward_prefetch)

    def _init_ep_model(self):
        """
        Initialize expert parallel (EP) model components for mixture-of-experts (MoE) training.
        """
        if self.ep_device_mesh is None:
            return
        ep_group = self.ep_device_mesh["EP"].get_group()
        ep_rank = torch.distributed.get_rank(ep_group)
        ep_size = torch.distributed.get_world_size(ep_group)
        moe_modules, moe_param_names = get_moe_modules_and_param_name(self, self.fsdp2_config.moe_modules)

        set_ep_size(ep_size)
        set_ep_rank(ep_rank)
        set_ep_group(ep_group)
        check_moe_fn = partial(check_moe_by_param_name, moe_param_names)
        set_check_moe_func(check_moe_fn)

        if not (ep_size > 1 and len(moe_modules) > 0):
            return

        ep_fsdp2_kwargs = copy.deepcopy(self.fsdp2_kwargs)

        if self.fsdp2_config.reshard_local_experts:
            ep_fsdp2_kwargs["mesh"] = self.ep_device_mesh["EP_Replicate"]

            def shard_placement_fn(*args, **kwargs):
                return Shard(1)

            # if experts shape: [num_experts, input_dim, output_dim], shard_placement_fn = Shard(1)
            # elif experts shape: [num_experts * input_dim, output_dim], shard_placement_fn = Shard(0) (default)
            if any(p.ndim == 3 for p in moe_modules[0].parameters()):
                ep_fsdp2_kwargs["shard_placement_fn"] = shard_placement_fn
        else:
            ep_fsdp2_kwargs["mesh"] = self.unit_device_mesh
            if torch.distributed.get_world_size(self.ep_device_mesh["EP_Replicate"].get_group()) != 1:
                set_ep_fsdp_group(self.ep_device_mesh["EP_Replicate"].get_group())

        for sub_module in moe_modules:
            for name, param in sub_module.named_parameters():
                ori_dtensor = DTensor.from_local(
                    local_tensor=param.data,
                    device_mesh=self.ep_device_mesh,
                    placements=[Replicate(), Replicate()]
                )
                new_dtensor = ori_dtensor.redistribute(
                    device_mesh=self.ep_device_mesh,
                    placements=[Replicate(), Shard(0)]
                )
                local_chunk = torch.nn.Parameter(new_dtensor.to_local(), requires_grad=param.requires_grad)
                set_module_from_path(sub_module, name, local_chunk)

            if hasattr(sub_module, "ep_forward") and callable(getattr(sub_module, "ep_forward")):
                forward_fn = partial(sub_module.ep_forward, ep_group)
                sub_module.forward = types.MethodType(forward_fn, sub_module)
            else:
                raise AssertionError(f"'Moe module {sub_module.__class__.__name__}' must implement 'ep_forward' method.")

        self._ep_fully_shard(ep_fsdp2_kwargs, moe_modules)

        # bugfix for HCCL premul sum issue, will be fixed in future torch release
        from mindspeed_mm.patchs.premul_sum_patch import apply_hccl_premul_sum_patch
        apply_hccl_premul_sum_patch()

        scale_factor = torch.distributed.get_world_size()
        for sub_module in moe_modules:
            if hasattr(sub_module, "set_gradient_divide_factor"): # torch>=2.8
                sub_module.set_gradient_divide_factor(scale_factor)
            else: # torch==2.7.1
                sub_module.set_reduce_scatter_divide_factor(scale_factor)

    def _ep_fully_shard(self, ep_fsdp2_kwargs, moe_modules):
        """
        Apply FSDP2 sharding specifically for expert parallel modules.

        Args:
            ep_fsdp2_kwargs: Keyword arguments for expert parallel fully_shard
            moe_modules: List of mixture-of-experts modules to shard
        """
        for sub_module in moe_modules:
            fully_shard(sub_module, **ep_fsdp2_kwargs)

    def fully_shard(self, process_group, fsdp2_config_path, **kwargs):
        """
        Applies Fully Sharded Data Parallel v2 (FSDP2) wrapping to the model for distributed training.

        Args:
            process_group (torch.distributed.ProcessGroup):
                The process group used for communication within a shard group.

            fsdp2_config_path (str):
                Path to the YAML configuration file for FSDP2 settings. Must include:
                - sharding_size (int or null): Number of devices in a shard group.
                  If null, defaults to world size.
                - param_dtype (str, optional): Data type for parameters (e.g., "bf16", "fp16", "fp32").
                - reduce_dtype (str, optional): Data type for gradient reduction.
                - cast_forward_inputs (bool, optional): Whether to cast forward inputs to param_dtype.
                - offload_to_cpu (bool, optional): Whether to enable CPU offloading.
                - pin_memory (bool, optional): Whether to use pinned memory for offloaded tensors.

            **kwargs:
                Additional keyword arguments for potential future FSDP options (currently unused).

        Returns:
            bool: True if sharding and setup completed successfully.

        Example:
            >>> model = MyModel(nn.Module, FSDP2Mixin)
            >>> model.fully_shard(
            ...     process_group=pg,
            ...     fsdp2_config_path="configs/fsdp2.yaml"
            ... )
            True

        Note:
            - This method modifies the model in-place by applying checkpoint wrappers and FSDP wrappers.
            - `post_meta_init` is only called if `args.init_model_with_meta_device` is True,
              to handle models initialized on meta device.
        """
        self._pre_fully_shard(process_group, fsdp2_config_path, **kwargs)
        self._init_ep_model()
        self._fully_shard(self.fsdp2_kwargs, self.fsdp2_config)
        self._post_fully_shard()
        return True


class WeightInitMixin:
    """
    Weight Initialization Mixin Class

    Provides general model weight initialization functionality, supporting multiple layer types
    and composite model structures. Can be used as a mixin class with other torch.nn.Module subclasses.
    """
    def _init_weights(self, module, std=0.02):
        """
        Initialize the weights. This is quite general on purpose, in the spirit of what we usually do. For more complex
        initialization scheme, it should be overridden by the derived `PreTrainedModel` class. In case a model adds an explicit
        `nn.Parameter`, this method should also be overridden in order to initialize it correctly.
        """
        if getattr(module, "_is_initialized", False):
            return

        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding) and module.padding_idx is None:
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.MultiheadAttention):
            # This uses torch's original init
            module._reset_parameters()
        # We cannot use `isinstance` on the RMSNorms or LayerNorms, as they usually are custom modules which change names
        # between modelings (because they are prefixed with the model name)
        elif (
            isinstance(module, (nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
            or "norm" in module.__class__.__name__.lower()
        ):
            # Norms can exist without weights (in which case they are None from torch primitives)
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.fill_(1.0)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
        # 3. Added: Generic parameter scanning and initialization for unhandled module types
        else:
            # Scan all Parameter attributes of the module
            for name, param in module.named_parameters(recurse=False):
                # Only process parameters that directly belong to this module (not recursive to submodules)
                if "weight" in name.lower():
                    param.data.normal_(mean=0.0, std=std)
                elif "bias" in name.lower():
                    param.data.zero_()
                else:
                    # Use default initialization for unknown parameter types
                    param.data.normal_(mean=0.0, std=std)

        module._is_initialized = True

    @torch.no_grad()
    def init_weights(self):
        """
        This is equivalent to calling `self.apply(self._initialize_weights)`, but correctly handles composite models.
        This function dynamically dispatches the correct `init_weights` function to the modules as we advance in the
        module graph along the recursion. It can handle an arbitrary number of sub-models. Without it, every composite
        model would have to recurse a second time on all sub-models explicitly in the outer-most `_init_weights`, which
        is extremely error prone and inefficient.

        Note that the `torch.no_grad()` decorator is very important as well, as most of our `_init_weights` do not use
        `torch.nn.init` functions (which are all no_grad by default), but simply do in-place ops such as
        `module.weight.data.zero_()`.
        """
        # This function is equivalent to `torch.nn.Module.apply`, except that it dynamically adjust the function
        # to apply as we go down the graph
        def smart_apply(self, fn):
            for module in self.children():
                module.smart_apply(fn)
            fn(self)
            return self

        torch.nn.Module.smart_apply = smart_apply

        # Let the magic happen with this simple call
        self.smart_apply(self._init_weights)
