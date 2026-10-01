# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import Optional
import functools
from dataclasses import dataclass
from contextlib import contextmanager

import torch
from torch.distributed import ProcessGroup
from megatron.core import mpu, tensor_parallel, parallel_state
from megatron.training import get_args
from megatron.core.enums import ModelType

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_model_config
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.transformer.module import Float16Module

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, MixedPrecision, ShardingStrategy, BackwardPrefetch
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy


@dataclass
class FSDPConfig:
    sharding_strategy: str = "fsdp"
    param_dtype: str = "bf16"
    reduce_dtype: str = "fp32"
    forward_prefetch: bool = False
    backward_prefetch: str = "backward_pre"


class _BaseDataParallel(MegatronModule):
    """A template class for DistributedDataParallel implementations."""

    def __init__(self, config: TransformerConfig, module: torch.nn.Module):
        super().__init__(config=config)
        self.module = module

    def forward(self, *inputs, **kwargs):
        """
        Calls the wrapped module's forward() method.
        """
        return self.module(*inputs, **kwargs)

    @contextmanager
    def no_sync(self):
        """
        Context manager that turns off gradient synchronization.
        """
        try:
            yield
        finally:
            pass

    def start_grad_sync(self, *unused):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients.

        When overlap_grad_reduce is set to True, dispatches asynchronous communication
        calls. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        pass

    def scale_gradients(self, scaling_factor: float) -> None:
        """Scale all gradients inside the buffers by `scaling_factor`."""
        pass

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        calls to complete. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        pass

    def zero_grad_buffer(self):
        """
        Zeros out all grad buffers. Needs to be called at the beginning of each
        training iteration.
        """
        pass

    def broadcast_params(self):
        """
        Syncs parameters across all DP ranks.
        """
        pass

    def state_dict(self, prefix='', keep_vars=False, destination=None):
        """
        Returns a dictionary containing references to the whole state of the
        wrapped module.

        Both parameters and persistent buffers (e.g. running averages) are included.
        Keys are corresponding parameter and buffer names. Parameters and buffers
        set to None are not included.
        """
        return self.module.state_dict(prefix=prefix, keep_vars=keep_vars, destination=destination)

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """
        Returns wrapped module's state_dict for checkpoint saving.
        """
        return self.module.state_dict_for_save_checkpoint(prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        """
        Copies parameters and buffers from state_dict into the wrapped module and its
        descendants. If strict is True, then the keys of state_dict must exactly match
        the keys returned by this module’s state_dict() function.
        """
        self.module.load_state_dict(state_dict, strict=strict)


class TorchFullyShardedDataParallel(_BaseDataParallel):
    def __init__(
        self,
        config: TransformerConfig,
        ddp_config: DistributedDataParallelConfig,
        module: torch.nn.Module,
        process_group: Optional[ProcessGroup] = None,
        fsdp_config: Optional[FSDPConfig] = None,
        **kwargs
    ):
        super().__init__(config=config, module=module)

        if process_group is None:
            self.process_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
        else:
            self.process_group = process_group

        if ddp_config.bucket_size is None:
            ddp_config.bucket_size = max(
                40000000, 1000000 * parallel_state.get_data_parallel_world_size()
            )
        # Set bucket_size to infinity if overlap_grad_reduce is False.
        if not ddp_config.overlap_grad_reduce:
            ddp_config.bucket_size = None

        self.ddp_config = ddp_config
        self.bucket_size = self.ddp_config.bucket_size
        self.expert_parallel_buffers = None

        def save_custom_attrs(module):
            custom_attrs = {}
            for name, param in module.named_parameters():
                attrs = vars(param)
                custom_attrs[name] = {k: v for k, v in attrs.items()}
            return custom_attrs

        def restore_custom_attrs(module, custom_attrs):
            for name, param in module.named_parameters():
                if name in custom_attrs:
                    for attr_name, attr_value in custom_attrs[name].items():
                        setattr(param, attr_name, attr_value)

        attrs = save_custom_attrs(self.module)
        fsdp_wrap_model_list = self.module.module.get_fsdp_wrap_module_list()
        for module_name, fsdp_wrap_module_list in fsdp_wrap_model_list:
            fsdp_wrap_model = getattr(self.module.module, module_name)
            fsdp_wrap_model = FSDP(
                fsdp_wrap_model,
                auto_wrap_policy=functools.partial(
                    lambda_auto_wrap_policy,
                    lambda_fn=lambda m: m in fsdp_wrap_module_list,
                ),
                process_group=process_group,
                sharding_strategy={
                    "fsdp": ShardingStrategy.FULL_SHARD,
                    "sdp": ShardingStrategy.SHARD_GRAD_OP,
                    "no": ShardingStrategy.NO_SHARD,
                    "hybrid": ShardingStrategy.HYBRID_SHARD,
                    "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2
                }[fsdp_config.sharding_strategy],
                mixed_precision=MixedPrecision(
                    param_dtype={
                        "fp32": torch.float,
                        "tf32": torch.float,
                        "bf16": torch.bfloat16,
                        "fp16": torch.float16,
                    }[fsdp_config.param_dtype],
                    reduce_dtype={
                        "fp32": torch.float,
                        "tf32": torch.float,
                        "bf16": torch.bfloat16,
                        "fp16": torch.float16,
                    }[fsdp_config.reduce_dtype],  # fp32
                ),
                device_id=torch.cuda.current_device(),
                sync_module_states=True,
                limit_all_gathers=True,
                use_orig_params=True,
                forward_prefetch=fsdp_config.forward_prefetch,
                backward_prefetch={
                    "backward_pre": BackwardPrefetch.BACKWARD_PRE,
                    "backward_post": BackwardPrefetch.BACKWARD_POST,
                }[fsdp_config.backward_prefetch],
            )
            setattr(self.module.module, module_name, fsdp_wrap_model)
        restore_custom_attrs(self.module, attrs)


def check_args(args):
    if args.pipeline_model_parallel_size > 1:
        return "pipeline_model_parallel_size"
    if args.expert_model_parallel_size > 1:
        return "expert_model_parallel_size"
    if args.tensor_model_parallel_size > 1:
        return "tensor_model_parallel_size"
    if args.use_distributed_optimizer:
        return "use_distributed_optimizer"
    if args.gradient_accumulation_fusion:
        return "gradient_accumulation_fusion"
    return ""


def fsdp1_get_model(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
    args = get_args()
    args.model_type = model_type

    # check args
    invalid_param = check_args(args)
    if invalid_param:
        raise AssertionError(f"Model splitting and distributed optimizer are not supported, check param {invalid_param}.")

    model = model_provider_func(
        pre_process=True,
        post_process=True
    )
    model.model_type = model_type

    if not isinstance(model, list):
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on (tensor, pipeline) '
              'model parallel rank ({}, {}): {}'.format(
            mpu.get_tensor_model_parallel_rank(),
            mpu.get_pipeline_model_parallel_rank(),
            sum([sum([p.nelement() for p in model_module.parameters()])
                 for model_module in model])), flush=True)

     # Fp16 conversion.
    if args.fp16 or args.bf16:
        config = get_model_config(model[0])
        model = [Float16Module(config, model_module) for model_module in model]

    if wrap_with_ddp:
        config = get_model_config(model[0])
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=args.accumulate_allreduce_grads_in_fp32,
            overlap_grad_reduce=args.overlap_grad_reduce,
            use_distributed_optimizer=args.use_distributed_optimizer,
            check_for_nan_in_grad=args.check_for_nan_in_loss_and_grad,
            bucket_size=args.ddp_bucket_size,
            average_in_collective=args.ddp_average_in_collective)

        fsdp_config_dict = get_args().mm.model.patch.use_fsdp1.to_dict()
        fsdp_config = FSDPConfig(**fsdp_config_dict)

        model = [TorchFullyShardedDataParallel(
            config=config,
            ddp_config=ddp_config,
            module=model_chunk,
            fsdp_config=fsdp_config,
        )
        for (model_chunk_idx, model_chunk) in enumerate(model)]

        torch.cuda.empty_cache()
        # Broadcast params from data parallel src rank to other data parallel ranks.
        if args.data_parallel_random_init:
            for model_module in model:
                model_module.broadcast_params()

    return model
