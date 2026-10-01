from functools import wraps
from typing import List

import torch

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.pipeline_parallel.schedules import custom_backward
from megatron.core.transformer.module import MegatronModule

from mindspeed.core.performance.auto_pipeline_perf.schedules import (
    backward_step_decorator,
)
from mindspeed.megatron_adaptor import get_mindspeed_args
from mindspeed.patch_utils import MindSpeedPatchesManager as mspm


def _get_megatron_optimizer_based_on_param_groups_wrapper(fn):
    @wraps(fn)
    def wrapper(
        config: OptimizerConfig,
        model_chunks: List[MegatronModule],
        param_groups: List,
        *args,
        **kwargs
    ):
        if len(param_groups) == 0:
            param_groups = [
                {
                    "params": torch.nn.parameter.Parameter(
                        data=torch.Tensor(), requires_grad=True
                    ),
                    "is_expert_parallel": False,
                    "is_decoupled_lr": False,
                }
            ]
        return fn(config, model_chunks, param_groups, *args, **kwargs)

    return wrapper


def backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage).
    """

    # NOTE: This code currently can handle at most one skip connection. It
    # needs to be modified slightly to support arbitrary numbers of skip
    # connections.

    if config.timers is not None:
        config.timers("backward-compute", log_level=2).start()

    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # Backward pass.
    if output_tensor_grad[0] is None and config.grad_scale_func is not None:
        output_tensor[0] = config.grad_scale_func(output_tensor[0])

    # Skip backward pass if grad_fn is None
    if output_tensor[0].grad_fn is not None:
        if config.deallocate_pipeline_outputs:
            custom_backward(output_tensor[0], output_tensor_grad[0])
        else:
            torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0])

    # Collect the grad of the input_tensor.
    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)

    # Handle single skip connection if it exists (encoder_hidden_state in
    # model with encoder and decoder).
    if (
        parallel_state.get_pipeline_model_parallel_world_size() > 1
        and parallel_state.is_pipeline_stage_after_split()
        and model_type == ModelType.encoder_and_decoder
    ):
        if output_tensor_grad[1] is not None:
            input_tensor_grad[-1].add_(output_tensor_grad[1])
    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    if config.timers is not None:
        config.timers("backward-compute").stop()

    return input_tensor_grad


mindspeed_args = get_mindspeed_args()
if (
    hasattr(mindspeed_args, "enable_dummy_optimizer")
    and mindspeed_args.enable_dummy_optimizer
):
    mspm.register_patch(
        "megatron.core.optimizer._get_megatron_optimizer_based_on_param_groups",
        _get_megatron_optimizer_based_on_param_groups_wrapper,
    )

    mspm.register_patch(
        "megatron.core.pipeline_parallel.schedules.backward_step",
        backward_step,
        force_patch=True,
    )

    mspm.apply_patches()
