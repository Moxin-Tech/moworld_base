# This code file is from [https://github.com/hao-ai-lab/FastVideo], which is licensed under Apache License 2.0.


import functools
from functools import partial

import torch
from peft.utils.other import fsdp_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper)
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, BackwardPrefetch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)


def apply_fsdp_checkpointing(model, no_split_modules, p=1):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    print("--> applying fdsp activation checkpointing...")
    block_idx = 0
    cut_off = 1 / 2
    # when passing p as a fraction number (e.g. 1/3), it will be interpreted
    # as a string in argv, parse it safely instead of using eval()
    if isinstance(p, str):
        if '/' in p:
            numerator, denominator = p.split('/')
            p = float(numerator) / float(denominator)
        else:
            p = float(p)

    def selective_checkpointing(submodule):
        nonlocal block_idx
        nonlocal cut_off

        if isinstance(submodule, no_split_modules):
            block_idx += 1
            if block_idx * p >= cut_off:
                cut_off += 1
                return True
        return False

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=selective_checkpointing,
    )


def get_mixed_precision(master_weight_type="fp32"):
    weight_type = torch.float32 if master_weight_type == "fp32" else torch.bfloat16
    mixed_precision = MixedPrecision(
        param_dtype=weight_type,
        # Gradient communication precision.
        reduce_dtype=weight_type,
        # Buffer precision.
        buffer_dtype=weight_type,
        cast_forward_inputs=False,
    )
    return mixed_precision


def get_dit_fsdp_kwargs(
        dancegrpo_model,
        sharding_strategy,
        use_lora=False,
        cpu_offload=False,
        master_weight_type="fp32",
):
    split_modules = dancegrpo_model.get_split_modules()
    if use_lora:
        auto_wrap_policy = fsdp_auto_wrap_policy
    else:
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=split_modules,
        )

    # we use float32 for fsdp but autocast during training
    mixed_precision = get_mixed_precision(master_weight_type)

    if sharding_strategy == "full":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif sharding_strategy == "hybrid_full":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    elif sharding_strategy == "none":
        sharding_strategy = ShardingStrategy.NO_SHARD
        auto_wrap_policy = None
    elif sharding_strategy == "hybrid_zero2":
        sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2

    device_id = torch.cuda.current_device()
    cpu_offload = (torch.distributed.fsdp.CPUOffload(
        offload_params=True) if cpu_offload else None)
    fsdp_kwargs = {
        "auto_wrap_policy": auto_wrap_policy,
        "mixed_precision": mixed_precision,
        "sharding_strategy": sharding_strategy,
        "device_id": device_id,
        "limit_all_gathers": True,
        "cpu_offload": cpu_offload,
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "forward_prefetch": True,
    }

    # Add LoRA-specific settings when LoRA is enabled
    if use_lora:
        fsdp_kwargs.update({
            "use_orig_params": False,  # Required for LoRA memory savings
            "sync_module_states": True,
        })

    return fsdp_kwargs, split_modules
