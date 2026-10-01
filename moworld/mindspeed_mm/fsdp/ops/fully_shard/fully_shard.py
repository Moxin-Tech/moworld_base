"""
Optimized native PyTorch FSDP2 for enhanced performance and ease of use. Key highlights include:

---------
Optimization 1:

FSDP2 (fully_shard) Custom Patch for Layer-wise Hook Management and Multi-Stream Communication.

This module extends PyTorch's native FSDP2 implementation to support:
1. hook_module: Attaching all FSDP hooks to a specific parent module (e.g., a Transformer block)
   instead of individual sub-modules, facilitating layer-wise management.

   BEST PRACTICE: The `hook_module` should generally be the same module wrapped by
   `checkpoint_wrapper`. This alignment specifically resolves bugs where activation
   checkpointing conflicts with having separate FSDPStates for sub-modules within a
   wrapped block, ensuring hooks are correctly triggered at the block level.

Usage:
    Apply the patch before initializing your model:
    >>> from mindspeed_mm.fsdp.ops.fully_shard.fully_shard import apply_fully_shard_patch
    >>> apply_fully_shard_patch()
    >>> from torch.distributed.fsdp import fully_shard
    >>> model = fully_shard(model, hook_module=layer_block)

--------
Optimization 2:

Refined FSDP2 multi-stream event dependencies to resolve the issue of an
extra block being prefetched in the timeline when prefetching is enabled,
resulting in a more rational pipeline layout.

In scenarios like EP, this prevents bandwidth contention caused by the
overlap between FSDP2 unshard communication and token dispatch communication.

"""


import weakref
import functools
from typing import (
    Any,
    Callable,
    Optional,
    Union,
    NamedTuple
)

import torch
import torch.nn as nn
from torch.utils._pytree import tree_map
from torch.distributed._composable import contract
from torch.profiler import record_function
from torch.distributed._composable_state import _insert_module_state
from torch.distributed.utils import _get_root_modules
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.tensor import DeviceMesh, Shard

# FSDP Internal Imports
# Note: These imports rely on internal PyTorch APIs which may change between versions.
from torch.distributed.fsdp._fully_shard._fsdp_api import MixedPrecisionPolicy, OffloadPolicy
from torch.distributed.fsdp._fully_shard._fsdp_common import (
    _cast_fp_tensor,
    FSDPMeshInfo,
    HSDPMeshInfo,
    compiled_autograd_enabled,
    TrainingState
)
from torch.distributed.fsdp._fully_shard._fsdp_init import (
    _get_device_from_mesh,
    _get_managed_modules,
    _get_managed_states,
    _get_post_forward_mesh_info,
    _init_default_fully_shard_mesh,
    _move_states_to_device,
)
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup, FSDPCommContext
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam, alloc_storage
from torch.distributed.fsdp._fully_shard._fsdp_state import (
    FSDPState,
    logger,
    disable_if_config_true,
    _register_group_forward_hooks
)
from torch.distributed.fsdp._fully_shard._fully_shard import _unimplemented_deepcopy, FSDPModule
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    foreach_all_gather_copy_out,
    foreach_reduce,
    AllGatherResult
)

# -----------------------------------------------------------------------------
# Global State & Type Definitions
# -----------------------------------------------------------------------------

# Mapping from original module class to the dynamically created FSDP-wrapped class
cls_to_fsdp_cls: dict[type, type] = {}

# Tracks the number of communication contexts assigned to each hook module.
# Key: hook_module, Value: count of contexts used (used to generate next index)
HOOK_MODULE_COMM_CTX_COUNT: weakref.WeakKeyDictionary[nn.Module, int] = weakref.WeakKeyDictionary()


class AllGatherState(NamedTuple):
    all_gather_result: AllGatherResult
    event: torch.Event  # all-gather copy-out
    hook_module: nn.Module


class ReduceScatterState(NamedTuple):
    reduce_scatter_input: torch.Tensor
    event: torch.Event  # reduce-scatter event
    hook_module: nn.Module


class AllReduceState(NamedTuple):
    all_reduce_input: torch.Tensor
    event: torch.Event  # all-reduce event


# -----------------------------------------------------------------------------
# Core API: fully_shard
# -----------------------------------------------------------------------------

@contract(state_cls=FSDPState)
def fully_shard(
    module,
    *,
    mesh: Optional[DeviceMesh] = None,
    reshard_after_forward: Union[bool, int] = True,
    shard_placement_fn: Optional[Callable[[nn.Parameter], Optional[Shard]]] = None,
    mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),
    offload_policy: OffloadPolicy = OffloadPolicy(),
    ignored_params: Optional[set[nn.Parameter]] = None,
    hook_module: Optional[nn.Module] = None,
):
    """
    Applies Fully Sharded Data Parallel (FSDP2) to a module with custom hook and stream management.

    Args:
        module: The module to shard.
        mesh: The device mesh for sharding. If None, a default 1D mesh is created.
        reshard_after_forward: Whether to reshard parameters after forward pass.
        shard_placement_fn: Custom function to determine shard placement.
        mp_policy: Mixed precision policy.
        offload_policy: CPU offload policy.
        ignored_params: Set of parameters to ignore during sharding.
        hook_module:
            The specific module to register forward/pre-forward hooks on.
            If None, hooks are registered on the 'module' itself.
            This allows grouping multiple FSDP units under a single logical layer hook.
    Returns:
        The sharded module.
    """

    if isinstance(module, (nn.ModuleList, nn.ModuleDict)):
        raise ValueError(
            f"fully_shard does not support containers that do not implement forward: {module}"
        )
    mesh = mesh or _init_default_fully_shard_mesh()
    if mesh.ndim not in (1, 2):
        raise ValueError(f"fully_shard expects a 1D or 2D DeviceMesh but got {mesh}")
    elif mesh.ndim == 1:
        mesh_info = FSDPMeshInfo(mesh, shard_mesh_dim=0)
    else:
        if mesh.mesh_dim_names is None:
            raise AssertionError(
                "Please init the 2D mesh for HSDP with mesh_dim_names specified"
            )
        mesh_info = HSDPMeshInfo(mesh, shard_mesh_dim=1, replicate_mesh_dim=0)
    device = _get_device_from_mesh(mesh)
    post_forward_mesh_info = _get_post_forward_mesh_info(
        reshard_after_forward, mesh_info
    )

    arg_module = module
    modules = (
        (module,) if isinstance(module, nn.Module) else tuple(_get_root_modules(module))
    )
    state = fully_shard.state(modules[0])  # type: ignore[attr-defined] # see [1]

    # Determine hook_module
    if hook_module:
        _hook_module = hook_module
    else:
        _hook_module = (modules[0] if len(modules) > 0 else modules)

    # Auto-increment comm_ctx_index
    if _hook_module not in HOOK_MODULE_COMM_CTX_COUNT:
        HOOK_MODULE_COMM_CTX_COUNT[_hook_module] = 0
    comm_ctx_index = HOOK_MODULE_COMM_CTX_COUNT.get(_hook_module)
    HOOK_MODULE_COMM_CTX_COUNT[_hook_module] = comm_ctx_index + 1

    # Initialize state with custom parameters
    state.init(
        modules, device, mp_policy,
        hook_module=hook_module,
        comm_ctx_index=comm_ctx_index,
    )

    managed_modules = _get_managed_modules(modules, ignored_params)
    params, buffers = _get_managed_states(managed_modules, ignored_params)

    _move_states_to_device(params, buffers, device)
    if params:
        state._fsdp_param_group = FSDPParamGroup(
            params,
            modules,
            mesh_info,
            post_forward_mesh_info,
            device,
            shard_placement_fn,
            mp_policy,
            offload_policy,
        )

    # For Dynamo
    for managed_module in managed_modules:
        managed_module._is_fsdp_managed_module = True  # type: ignore[assignment]
        managed_module._fsdp_use_orig_params = True  # type: ignore[assignment]

    # Place FSDP leftmost for highest priority in the method resolution order
    for module in modules:
        cls = module.__class__
        new_cls = cls_to_fsdp_cls.get(cls, None)
        if not new_cls:
            dct = {"__deepcopy__": _unimplemented_deepcopy}
            new_cls = type(f"FSDP{cls.__name__}", (FSDPModule, cls), dct)
            cls_to_fsdp_cls[cls] = new_cls
        module.__class__ = new_cls
    return arg_module


# -----------------------------------------------------------------------------
# Patched FSDPState Methods
# -----------------------------------------------------------------------------

def hook_module_init(
    self,
    modules: tuple[nn.Module, ...],
    device: torch.device,
    mp_policy: MixedPrecisionPolicy,
    hook_module: Optional[nn.Module] = None,
    comm_ctx_index: int = 0,
) -> None:
    """
    Custom initialization for FSDPState.

    Extends the default init to:
    1. Register hooks on a specific 'hook_module' (if provided) instead of the first managed module.
    2. Store the 'comm_ctx_index' for multi-stream management.
    """

    for module in modules:
        _insert_module_state(module, self)
    self._modules = modules
    self._device = device
    self._device_handle = _get_device_handle(device.type)
    self._mp_policy = mp_policy
    self.comm_ctx_index = comm_ctx_index

    # Register Hooks
    if hook_module:
        # Register hooks on the user-specified hook_module
        self._pre_forward_hook_handle = hook_module.register_forward_pre_hook(
            self._pre_forward, prepend=True, with_kwargs=True
        )
        self._post_forward_hook_handle = hook_module.register_forward_hook(
            self._post_forward, prepend=False
        )
        self.hook_module = weakref.ref(hook_module)
    else:
        # Fallback to default behavior if no hook_module is specified
        if len(modules) == 1:
            self._pre_forward_hook_handle = modules[0].register_forward_pre_hook(
                self._pre_forward, prepend=True, with_kwargs=True
            )
            self._post_forward_hook_handle = modules[0].register_forward_hook(
                self._post_forward, prepend=False
            )
        else:
            hook_handle = _register_group_forward_hooks(
                modules,
                self._pre_forward,
                self._post_forward,
                self._modules_to_run_forward,
            )
            self._pre_forward_hook_handle = hook_handle
            self._post_forward_hook_handle = hook_handle

        self.hook_module = weakref.ref(modules[0])


def copy_fsdp_comm_ctx(new_comm_ctx: FSDPCommContext, comm_ctx: FSDPCommContext) -> FSDPCommContext:
    """
    Copies critical stream and state attributes from one communication context to another.
    Used to initialize additional global communication contexts based on the root context.
    """

    new_comm_ctx.device_handle = comm_ctx.device_handle

    # # Copy streams
    new_comm_ctx.all_gather_copy_in_stream = comm_ctx.all_gather_copy_in_stream
    new_comm_ctx.all_gather_stream = comm_ctx.all_gather_stream
    new_comm_ctx.reduce_scatter_stream = comm_ctx.reduce_scatter_stream
    new_comm_ctx.all_reduce_stream = comm_ctx.all_reduce_stream

    # Copy state placeholders
    new_comm_ctx.all_gather_state = comm_ctx.all_gather_state
    new_comm_ctx.reduce_scatter_state = comm_ctx.reduce_scatter_state
    new_comm_ctx.post_forward_order = comm_ctx.post_forward_order

    return new_comm_ctx


def hook_module_init_shared_state(self) -> None:
    """
    Initializes shared state across all FSDP states in the context.

    Creates a global list of communication contexts (global_comm_ctx) to manage
    multiple streams. It ensures that every unique comm_ctx_index used by any
    state in the group has a corresponding initialized FSDPCommContext.
    """

    self._comm_ctx.lazy_init(self._device)
    if not hasattr(self, "global_comm_ctx"):
        self.global_comm_ctx = [self._comm_ctx]

    # Collect all unique comm_ctx_indices used in this state context
    global_comm_ctx_list = [0]
    for state in self._state_ctx.all_states:
        if state.comm_ctx_index not in global_comm_ctx_list:
            global_comm_ctx_list.append(state.comm_ctx_index)
            new_comm_ctx = FSDPCommContext()
            new_comm_ctx = copy_fsdp_comm_ctx(new_comm_ctx, self._comm_ctx)
            self.global_comm_ctx.append(new_comm_ctx)

    # Assign the correct comm_ctx to each state based on its index
    for state in self._state_ctx.all_states:
        state._state_ctx = self._state_ctx

        # set comm_ctx_index
        _comm_ctx = self.global_comm_ctx[global_comm_ctx_list.index(state.comm_ctx_index)]

        setattr(state, "global_comm_ctx", self.global_comm_ctx)

        state._comm_ctx = _comm_ctx
        if fsdp_param_group := state._fsdp_param_group:
            fsdp_param_group.comm_ctx = _comm_ctx
            setattr(fsdp_param_group, "hook_module", state.hook_module)
            setattr(fsdp_param_group, "global_comm_ctx", self.global_comm_ctx)


def _root_post_backward_final_callback(self) -> None:
    """
    Custom callback executed after the final backward pass.

    Ensures that the main stream waits for ALL reduce-scatter events from
    ALL communication contexts (global_comm_ctx) to complete before finishing.
    This is crucial for correctness when using multiple streams.
    """

    if not compiled_autograd_enabled():
        logger.debug("FSDP::root_post_backward")
    with torch.profiler.record_function("FSDP::root_post_backward_callback"):
        for state in self._state_ctx.all_states:
            fsdp_param_group = state._fsdp_param_group
            if (
                fsdp_param_group
                and fsdp_param_group._training_state != TrainingState.POST_BACKWARD
            ):
                # Run post-backward in case forward inputs did not require
                # gradient so the autograd backward did not run
                fsdp_param_group.post_backward()
            state._training_state = TrainingState.IDLE
            if fsdp_param_group:
                fsdp_param_group._training_state = TrainingState.IDLE
            if self._state_ctx.is_last_backward:
                state._finalize_backward()
        if self._state_ctx.is_last_backward:
            self._comm_ctx.post_forward_order.clear()
            if self._comm_ctx.reduce_scatter_state is not None:
                self._device_handle.current_stream().wait_event(
                    self._comm_ctx.reduce_scatter_state.event
                )
                self._comm_ctx.reduce_scatter_state = None

            # WAIT FOR ALL GLOBAL COMM CONTEXTS
            # This ensures synchronization across all custom streams
            if hasattr(self, "global_comm_ctx"):
                for _comm_ctx in self.global_comm_ctx:
                    _comm_ctx.post_forward_order.clear()
                    if _comm_ctx.reduce_scatter_state is not None:
                        self._device_handle.current_stream().wait_event(
                            _comm_ctx.reduce_scatter_state.event
                        )
                    _comm_ctx.reduce_scatter_state = None

        self._state_ctx.post_backward_final_callback_queued = False


@disable_if_config_true
def _pre_forward(
    self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    # When composing with module-hook-based activation checkpointing, the
    # the pre-backward hook is responsible for the unshard
    if self._training_state == TrainingState.PRE_BACKWARD:
        return args, kwargs
    self._training_state = TrainingState.FORWARD
    args, kwargs = self._root_pre_forward(module, args, kwargs)
    if self._mp_policy.cast_forward_inputs and self._mp_policy.param_dtype:
        with torch.profiler.record_function("FSDP::cast_forward_inputs"):
            cast_fn = functools.partial(
                _cast_fp_tensor, self._mp_policy.param_dtype
            )
            args, kwargs = tree_map(cast_fn, args), tree_map(cast_fn, kwargs)
    if self._fsdp_param_group:
        args, kwargs = self._fsdp_param_group.pre_forward(module, args, kwargs)
    for fsdp_state in self._states_to_forward_prefetch:
        if (target_param_group := fsdp_state._fsdp_param_group) is not None:
            prefetch_all_gather_copy_in_stream = target_param_group.comm_ctx.all_gather_copy_in_stream

            # Notice:
            # [Optimization 2] Refine multi-stream event dependencies to optimize
            # pipeline scheduling.
            # Explicitly wait for the previous global communication event to complete
            # before triggering a new prefetch. This prevents an extra block from being
            # prefetched and avoids bandwidth contention (e.g., between FSDP unshard
            # and token dispatch in EP scenarios).
            for comm_ctx in self.global_comm_ctx:
                if comm_ctx.all_gather_state and comm_ctx.all_gather_state.event:
                    prefetch_all_gather_copy_in_stream.wait_event(comm_ctx.all_gather_state.event)

            FSDPParamGroup._prefetch_unshard(target_param_group, "forward")
    return args, kwargs


@disable_if_config_true
def _post_forward(self, module: nn.Module, input: Any, output: Any) -> Any:
    """
    Custom post-forward hook.

    Waits for all All-Gather operations from ALL communication contexts to complete
    and frees their events before proceeding. This prevents memory leaks and ensures
    data readiness for subsequent operations.
    """

    if self._training_state == TrainingState.PRE_BACKWARD:
        return output
    if self._fsdp_param_group:
        output = self._fsdp_param_group.post_forward(module, input, output)
    output = self._register_pre_backward_hook(output)
    self._training_state = TrainingState.IDLE
    # Wait and free All-Gather states for ALL global communication contexts
    if self._state_ctx.iter_forward_root is self:
        for comm_ctx in self.global_comm_ctx:
            # Free the last all-gather result if needed; refer to
            # [Note: Overlapping all-gather copy-in and all-gather]
            if comm_ctx.all_gather_state:
                # Wait for the copy-in and main all-gather streams
                self._comm_ctx.all_gather_copy_in_stream.wait_event(comm_ctx.all_gather_state.event)
                self._comm_ctx.all_gather_stream.wait_event(comm_ctx.all_gather_state.event)
            # free the all-gather result
            comm_ctx.all_gather_state = None

        self._state_ctx.iter_forward_root = None

    if self._mp_policy.output_dtype is not None:
        with torch.profiler.record_function("FSDP::cast_forward_outputs"):
            output = tree_map(
                functools.partial(_cast_fp_tensor, self._mp_policy.output_dtype),
                output,
            )
    return output


# -----------------------------------------------------------------------------
# Patched FSDPParamGroup Methods
# -----------------------------------------------------------------------------

def param_group_wait_for_unshard_pt27(self):
    """
    Waits for preceding All-Gather operations to complete before unsharding.

    Specifically checks global_comm_ctx for events generated by DIFFERENT hook_modules.
    This enables overlapping communication: Layer N can start computing while Layer N+1
    is still gathering, provided they use different streams/contexts.
    """

    if not self._all_gather_result:
        return  # no preceding unshard
    async_op = self._all_gather_result.all_gather_work is not None
    if self._training_state == TrainingState.FORWARD:  # implicit prefetch

        for comm_ctx in self.global_comm_ctx:
            if prev_all_gather_state := comm_ctx.all_gather_state:
                # Only wait if the previous operation belongs to a DIFFERENT hook_module.
                # If it's the same module, dependencies are handled differently or sequentially.
                # Note: Logic assumes self.hook_module is available on the param group
                if prev_all_gather_state.hook_module != self.hook_module:
                    self._wait_all_gather_streams_on_event(prev_all_gather_state.event)
                    comm_ctx.all_gather_state = None # free the all-gather result

    with record_function(self._with_fqn("FSDP::all_gather_copy_out")):
        foreach_all_gather_copy_out(
            self._all_gather_result,
            self.fsdp_params,
            self._all_gather_process_group,
        )
    for fsdp_param in self.fsdp_params:
        fsdp_param.init_unsharded_param()
    self._to_unsharded()
    all_gather_copy_out_event = self.device_handle.Event()
    all_gather_copy_out_event.record()
    if not async_op and self._training_state == TrainingState.FORWARD:
        # Defer free to allow for overlap of this copy-out with next
        # all-gather collective
        self.comm_ctx.all_gather_state = AllGatherState(
            self._all_gather_result, all_gather_copy_out_event, self.hook_module
        )
    else:
        self._wait_all_gather_streams_on_event(all_gather_copy_out_event)
    self._all_gather_result = None  # free unless saved in `all_gather_state`


def param_group_wait_for_unshard_pt29(self):
    """
    Waits for preceding All-Gather operations to complete before unsharding.

    Specifically checks global_comm_ctx for events generated by DIFFERENT hook_modules.
    This enables overlapping communication: Layer N can start computing while Layer N+1
    is still gathering, provided they use different streams/contexts.
    """

    if not self._all_gather_result:
        return  # no preceding unshard
    async_op = self._all_gather_result.all_gather_work is not None
    if self._training_state == TrainingState.FORWARD:  # implicit prefetch
        for comm_ctx in self.global_comm_ctx:
            if prev_all_gather_state := comm_ctx.all_gather_state:
                # Only wait if the previous operation belongs to a DIFFERENT hook_module.
                # If it's the same module, dependencies are handled differently or sequentially.
                # Note: Logic assumes self.hook_module is available on the param group
                if prev_all_gather_state.hook_module != self.hook_module:
                    self._wait_all_gather_streams_on_event(prev_all_gather_state.event)
                    comm_ctx.all_gather_state = None  # free the all-gather result
    world_size = self._all_gather_process_group.size()
    if world_size == 1:
        # directly initialize unsharded parameters from sharded parameters

        for fsdp_param in self.fsdp_params:
            # Use all_gather_inputs which already handles conversion to param_dtype
            # This is consistent with the world_size > 1 path
            all_gather_input = fsdp_param.all_gather_inputs[0]

            # Make sure the all_gather_outputs has proper storage size before using it
            # First ensure we have at least one tensor in all_gather_outputs
            fsdp_param.init_all_gather_outputs(
                [all_gather_input.numel()],
                [all_gather_input.dtype],
                world_size,
                self.device,
                force_recreate=False,
            )

            tensor = fsdp_param.all_gather_outputs[0]
            alloc_storage(tensor)

            # find alternative way to check if tensor.is_inference
            with torch.autograd._unsafe_preserve_version_counter(tensor):
                tensor.copy_(all_gather_input)

    else:
        with record_function(self._with_fqn("FSDP::all_gather_copy_out")):
            foreach_all_gather_copy_out(
                self._all_gather_result,
                self.fsdp_params,
                self._all_gather_process_group,
            )

    for fsdp_param in self.fsdp_params:
        fsdp_param.init_unsharded_param()

    self._to_unsharded()
    all_gather_copy_out_event = self.device_handle.Event()
    all_gather_copy_out_event.record()

    if (
        not async_op
        and self._training_state == TrainingState.FORWARD
        and world_size > 1
    ):
        # Defer free to allow for overlap of this copy-out with next
        # all-gather collective
        self.comm_ctx.all_gather_state = AllGatherState(
            self._all_gather_result, all_gather_copy_out_event, self.hook_module
        )
    else:
        self._wait_all_gather_streams_on_event(all_gather_copy_out_event)

    self._all_gather_result = None  # free unless saved in `all_gather_state`


def param_group_post_backward_pt27(self, *unused: Any):
    """
    Custom post-backward logic for gradient reduction and resharding.

    Ensures that the current stream waits for Reduce-Scatter events from
    OTHER communication contexts (different hook_modules) before starting
    its own reduction. This maintains correctness in multi-stream setups.
    """

    if not compiled_autograd_enabled():
        logger.debug("%s", self._with_fqn("FSDP::post_backward"))
    self._training_state = TrainingState.POST_BACKWARD
    with record_function(self._with_fqn("FSDP::post_backward_accumulate")):
        for fsdp_param in self.fsdp_params:
            fsdp_param.accumulate_unsharded_grad_if_needed()

    with record_function(self._with_fqn("FSDP::post_backward_reshard")):
        if not self.reduce_grads:
            if self.reshard_after_backward:
                self.reshard()
            for fsdp_param in self.fsdp_params:
                fsdp_param.to_accumulated_grad_if_needed()
            return
        # Save the autograd-computed gradients before resharding to only
        # access the unsharded parameters when their data is present
        fsdp_params_with_grad: list[FSDPParam] = []
        unsharded_grads: list[torch.Tensor] = []
        for fsdp_param in self.fsdp_params:
            if not hasattr(fsdp_param, "_unsharded_param"):
                continue
            # May have an accumulated gradient of the reduce dtype if the
            # previous backward did not reduce-scatter
            if fsdp_param.unsharded_accumulated_grad is not None:
                fsdp_params_with_grad.append(fsdp_param)
                unsharded_grads.append(fsdp_param.unsharded_accumulated_grad_data)
                fsdp_param.unsharded_accumulated_grad = None
            elif fsdp_param.unsharded_param.grad is not None:
                fsdp_params_with_grad.append(fsdp_param)
                unsharded_grads.append(fsdp_param.unsharded_grad_data)
                fsdp_param.unsharded_param.grad = None
        if self.reshard_after_backward:
            self.reshard()

    if len(fsdp_params_with_grad) == 0:
        return

    with record_function(self._with_fqn("FSDP::post_backward_reduce")):
        # Wait for local context reduce-scatter
        if self.comm_ctx.reduce_scatter_state is not None:
            self.device_handle.current_stream().wait_event(
                self.comm_ctx.reduce_scatter_state.event
            )
            self.comm_ctx.reduce_scatter_state = None

        # Wait for GLOBAL context reduce-scatters from DIFFERENT hook modules
        for comm_ctx in self.global_comm_ctx:
            if comm_ctx.reduce_scatter_state and comm_ctx.reduce_scatter_state.hook_module != self.hook_module:
                self.device_handle.current_stream().wait_event(comm_ctx.reduce_scatter_state.event)
                comm_ctx.reduce_scatter_state = None

        all_reduce_pg = self._all_reduce_process_group if self._is_hsdp else None
        all_reduce_stream: torch.cuda.Stream
        if all_reduce_pg is None and self._all_reduce_hook_stream is not None:
            # this means the native HSDP is not enabled,
            # but user may want to have a custom HSDP setup
            if self._all_reduce_hook is None:
                raise AssertionError(
                    "all reduce hook stream is specified but hook itself is missing."
                )
            all_reduce_stream = self._all_reduce_hook_stream
        else:
            all_reduce_stream = self.comm_ctx.all_reduce_stream

        self._wait_for_post_backward()
        (
            reduce_scatter_input,
            reduce_scatter_event,
            self._post_reduce_event,
            all_reduce_input,
            all_reduce_event,
            self._partial_reduce_output,
        ) = foreach_reduce(
            fsdp_params_with_grad,
            unsharded_grads,
            self._reduce_scatter_process_group,
            self.comm_ctx.reduce_scatter_stream,
            self._orig_dtype,
            self._reduce_dtype,
            self.device,
            self.reduce_scatter_reduce_op,
            self._all_reduce_process_group if self._is_hsdp else None,
            all_reduce_stream,
            self.all_reduce_grads,
            self._partial_reduce_output,
            self._all_reduce_hook,
        )

        # Record the new reduce-scatter state
        self.comm_ctx.reduce_scatter_state = ReduceScatterState(
            reduce_scatter_input, reduce_scatter_event, self.hook_module
        )
        if all_reduce_input is not None:
            if all_reduce_event is None:
                raise AssertionError("all_reduce_event cannot be None.")
            self._all_reduce_state = AllReduceState(
                all_reduce_input, all_reduce_event
            )


def param_group_post_backward_pt29(self, *unused: Any):
    """
    Custom post-backward logic for gradient reduction and resharding.

    Ensures that the current stream waits for Reduce-Scatter events from
    OTHER communication contexts (different hook_modules) before starting
    its own reduction. This maintains correctness in multi-stream setups.
    """

    if not compiled_autograd_enabled():
        logger.debug("%s", self._with_fqn("FSDP::post_backward"))
    self._training_state = TrainingState.POST_BACKWARD
    with record_function(self._with_fqn("FSDP::post_backward_accumulate")):
        for fsdp_param in self.fsdp_params:
            fsdp_param.accumulate_unsharded_grad_if_needed()

    with record_function(self._with_fqn("FSDP::post_backward_reshard")):
        if not self.reduce_grads:
            if self.reshard_after_backward:
                self.reshard()
            for fsdp_param in self.fsdp_params:
                fsdp_param.to_accumulated_grad_if_needed()
            return
        # Save the autograd-computed gradients before resharding to only
        # access the unsharded parameters when their data is present
        fsdp_params_with_grad: list[FSDPParam] = []
        unsharded_grads: list[torch.Tensor] = []
        for fsdp_param in self.fsdp_params:
            if not hasattr(fsdp_param, "_unsharded_param"):
                continue
            # May have an accumulated gradient of the reduce dtype if the
            # previous backward did not reduce-scatter
            if fsdp_param.unsharded_accumulated_grad is not None:
                fsdp_params_with_grad.append(fsdp_param)
                unsharded_grads.append(fsdp_param.unsharded_accumulated_grad_data)
                fsdp_param.unsharded_accumulated_grad = None
            elif fsdp_param.unsharded_param.grad is not None:
                fsdp_params_with_grad.append(fsdp_param)
                unsharded_grads.append(fsdp_param.unsharded_grad_data)
                fsdp_param.unsharded_param.grad = None
        if self.reshard_after_backward:
            self.reshard()

    if len(fsdp_params_with_grad) == 0:
        return

    with record_function(self._with_fqn("FSDP::post_backward_reduce")):
        # Wait for local context reduce-scatter
        if self.comm_ctx.reduce_scatter_state is not None and self.comm_ctx.reduce_scatter_state.event is not None:
            self.device_handle.current_stream().wait_event(
                self.comm_ctx.reduce_scatter_state.event
            )
            self.comm_ctx.reduce_scatter_state = None

        # Wait for GLOBAL context reduce-scatters from DIFFERENT hook modules
        for comm_ctx in self.global_comm_ctx:
            if comm_ctx.reduce_scatter_state and comm_ctx.reduce_scatter_state.hook_module != self.hook_module:
                self.device_handle.current_stream().wait_event(comm_ctx.reduce_scatter_state.event)
                comm_ctx.reduce_scatter_state = None

        all_reduce_pg = self._all_reduce_process_group if self._is_hsdp else None
        all_reduce_stream: torch.cuda.Stream
        if all_reduce_pg is None and self._all_reduce_hook_stream is not None:
            # this means the native HSDP is not enabled,
            # but user may want to have a custom HSDP setup
            if self._all_reduce_hook is None:
                raise AssertionError(
                    "all reduce hook stream is specified but hook itself is missing."
                )
            all_reduce_stream = self._all_reduce_hook_stream
        else:
            all_reduce_stream = self.comm_ctx.all_reduce_stream

        self._wait_for_post_backward()
        (
            reduce_scatter_input,
            reduce_scatter_event,
            self._post_reduce_event,
            all_reduce_input,
            all_reduce_event,
            self._partial_reduce_output,
        ) = foreach_reduce(
            fsdp_params_with_grad,
            unsharded_grads,
            self._reduce_scatter_process_group,
            self.comm_ctx.reduce_scatter_stream,
            self._reduce_scatter_comm,
            self._orig_dtype,
            self._reduce_dtype,
            self.device,
            self.gradient_divide_factor,
            self._all_reduce_process_group if self._is_hsdp else None,
            all_reduce_stream,
            self.all_reduce_grads,
            self._partial_reduce_output,
            self._all_reduce_hook,
            self.force_sum_reduction_for_comms,
        )

        # Record the new reduce-scatter state
        self.comm_ctx.reduce_scatter_state = ReduceScatterState(
            reduce_scatter_input, reduce_scatter_event, self.hook_module
        )
        if all_reduce_input is not None:
            if self.device.type != "cpu":
                raise AssertionError("all_reduce_event cannot be None.")
            self._all_reduce_state = AllReduceState(
                all_reduce_input, all_reduce_event
            )


# -----------------------------------------------------------------------------
# Patch Application
# ----
def apply_fully_shard_patch() -> None:
    """
    Applies all custom patches to the FSDPState and FSDPParamGroup classes.
    Call this function once at the beginning of your training script.
    """
    # Patch FSDPState methods
    FSDPState.init = hook_module_init
    FSDPState._init_shared_state = hook_module_init_shared_state
    FSDPState._root_post_backward_final_callback = _root_post_backward_final_callback
    FSDPState._pre_forward = _pre_forward
    FSDPState._post_forward = _post_forward

    # Patch FSDPParamGroup methods
    if "2.7.1" in torch.__version__:
        FSDPParamGroup.wait_for_unshard = param_group_wait_for_unshard_pt27
        FSDPParamGroup.post_backward = param_group_post_backward_pt27
    elif "2.9.0" in torch.__version__:
        FSDPParamGroup.wait_for_unshard = param_group_wait_for_unshard_pt29
        FSDPParamGroup.post_backward = param_group_post_backward_pt29
    else:
        raise ValueError(f"The torch{torch.__version__} is not supported now.")

    # Override the public fully_shard API
    from torch.distributed import fsdp
    fsdp.fully_shard = fully_shard
