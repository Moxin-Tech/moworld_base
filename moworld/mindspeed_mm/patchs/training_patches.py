# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright 2024 Huawei Technologies Co., Ltd

import os
import dataclasses

import torch
from accelerate import init_empty_weights
from megatron.core.enums import ModelType
from megatron.training.utils import print_rank_0
from megatron.training.global_vars import get_args
from megatron.core import mpu, tensor_parallel
from megatron.core.transformer.module import Float16Module
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.training.checkpointing import load_checkpoint, get_checkpoint_tracker_filename, read_metadata, get_checkpoint_name
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.utils import get_model_config
from megatron.core.fp8_utils import correct_amax_history_if_needed
from megatron.core.distributed.custom_fsdp import FullyShardedDataParallel as custom_FSDP
try:
    from megatron.core.distributed import TorchFullyShardedDataParallel as torch_FSDP
    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False

from mindspeed_mm.utils.utils import ensure_valid
from mindspeed_mm.tasks.finetune.lora.utils import is_enable_lora
from mindspeed_mm.tasks.finetune.lora.lora_patch import model_provider_func_wrapper


def get_dist_model_load_from_pt(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
    """
    This function is copied from Megatron's get_model function with one key modification:
    Added functionality to load model weights from .pt checkpoint files at the end of build_model().
    This enables distributed training scenarios where weights need to be loaded from .pt format checkpoints.
    """
    args = get_args()
    args.model_type = model_type

    if is_enable_lora():
        model_provider_func = model_provider_func_wrapper(model_provider_func)

    # Build model.
    def build_model():
        if mpu.get_pipeline_model_parallel_world_size() > 1 and \
        args.virtual_pipeline_model_parallel_size is not None:
            if model_type == ModelType.encoder_and_decoder:
                ensure_valid(
                    args.encoder_pipeline_model_parallel_size == 0,
                    "Interleaved schedule not supported for model with encoder on separate PP rank"
                )
            model = []
            for i in range(args.virtual_pipeline_model_parallel_size):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                # Set pre_process and post_process only after virtual rank is set.
                pre_process = mpu.is_pipeline_first_stage()
                post_process = mpu.is_pipeline_last_stage()
                this_model = model_provider_func(
                    pre_process=pre_process,
                    post_process=post_process
                )
                this_model.model_type = model_type
                model.append(this_model)
        else:
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            add_encoder = True
            add_decoder = True
            if model_type == ModelType.encoder_and_decoder:
                if mpu.get_pipeline_model_parallel_world_size() > 1:
                    rank = mpu.get_pipeline_model_parallel_rank()
                    first_decoder_rank = args.encoder_pipeline_model_parallel_size
                    world_size = mpu.get_pipeline_model_parallel_world_size()
                    pre_process = rank == 0 or rank == first_decoder_rank
                    post_process = (rank == (first_decoder_rank - 1)) or (rank == (world_size - 1))
                    add_encoder = mpu.is_inside_encoder(rank)
                    add_decoder = mpu.is_inside_decoder(rank)
                model = model_provider_func(
                    pre_process=pre_process,
                    post_process=post_process,
                    add_encoder=add_encoder,
                    add_decoder=add_decoder)
            else:
                model = model_provider_func(
                    pre_process=pre_process,
                    post_process=post_process
                )
            model.model_type = model_type

        # =========================load checkpoint===============
        # Additional functionality added: Load weights from .pt checkpoint files
        # This enables loading model weights in distributed training scenarios
        load_dir = args.load
        if args.use_dist_ckpt and load_dir is not None:
            iteration, release = -1, False
            tracker_filename = get_checkpoint_tracker_filename(load_dir)
            if os.path.isfile(tracker_filename):
                iteration, release = read_metadata(tracker_filename)

            # return_base_dir is False，return the `.pt` file path
            checkpoint_name = get_checkpoint_name(load_dir, iteration, release, return_base_dir=False)

            if not os.path.exists(checkpoint_name):
                return model

            return_list = True
            if not isinstance(model, list):
                model = [model]
                return_list = False
            print_rank_0(
                f' loading checkpoint from {checkpoint_name} at iteration {iteration}'
            )
            ori_ckpt_format = args.ckpt_format
            args.ckpt_format = "torch"
            load_checkpoint(model, None, None)
            args.ckpt_format = ori_ckpt_format

            if not return_list:
                model = model[0]
            args.load = None
        # ==========================================================

        return model
    if args.init_model_with_meta_device:
        with init_empty_weights():
            model = build_model()
    else:
        model = build_model()

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
    num_parameters = sum(
        [sum([p.nelement() for p in model_module.parameters()])
         for model_module in model]
    )
    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on (tensor, pipeline) '
              'model parallel rank ({}, {}): {}'.format(
            mpu.get_tensor_model_parallel_rank(),
            mpu.get_pipeline_model_parallel_rank(),
            num_parameters), flush=True)

    # GPU allocation.
    # For FSDP2, we don't allocate GPU memory here. We allocate GPU memory
    # in the fully_shard function of FSDP2 instead.
    if not (args.use_torch_fsdp2 and args.use_cpu_initialization) and not args.init_model_with_meta_device:
        for model_module in model:
            model_module.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16 or args.bf16:
        config = get_model_config(model[0])
        model = [Float16Module(config, model_module) for model_module in model]

    # Before TE2.x: The model_module.bfloat16()/model_module.half() above will call the inplace
    #               copy of TE's Float8Tensor, which will write an unwanted value (amax calculated
    #               from the current fp8 param) to its amax_history. The below function will correct
    #               the amax_history back.
    # After TE2.x: Below function is an empty function and does nothing.
    correct_amax_history_if_needed(model)

    if wrap_with_ddp:
        if args.use_torch_fsdp2:
            ensure_valid(
                HAVE_FSDP2,
                "Torch FSDP2 requires torch>=2.4.0"
            )

            DP = torch_FSDP
        elif args.use_custom_fsdp:
            DP = custom_FSDP
        else:
            DP = DDP

        config = get_model_config(model[0])

        kwargs = {}
        for f in dataclasses.fields(DistributedDataParallelConfig):
            if hasattr(args, f.name):
                kwargs[f.name] = getattr(args, f.name)
        kwargs['grad_reduce_in_fp32'] = args.accumulate_allreduce_grads_in_fp32
        kwargs['check_for_nan_in_grad'] = args.check_for_nan_in_loss_and_grad
        kwargs['check_for_large_grads'] = args.check_for_large_grads
        if args.ddp_num_buckets is not None:
            ensure_valid(
                args.ddp_bucket_size is None,
                "Cannot specify both --ddp-num-buckets and --ddp-bucket-size"
            )
            ensure_valid(
                args.ddp_num_buckets > 0,
                "--ddp-num-buckets must be greater than 0"
            )
            kwargs['bucket_size'] = num_parameters // args.ddp_num_buckets
        else:
            kwargs['bucket_size'] = args.ddp_bucket_size
        kwargs['pad_buckets_for_high_nccl_busbw'] = args.ddp_pad_buckets_for_high_nccl_busbw
        kwargs['average_in_collective'] = args.ddp_average_in_collective
        if args.use_custom_fsdp and args.use_precision_aware_optimizer:
            kwargs["preserve_fp32_weights"] = False
        ddp_config = DistributedDataParallelConfig(**kwargs)

        if not getattr(args, "use_torch_fsdp2", False):
            # In the custom FSDP and DDP use path, we need to initialize the bucket size.

            # If bucket_size is not provided as an input, use sane default.
            # If using very large dp_sizes, make buckets larger to ensure that chunks used in NCCL
            # ring-reduce implementations are large enough to remain bandwidth-bound rather than
            # latency-bound.
            if ddp_config.bucket_size is None:
                ddp_config.bucket_size = max(
                    40000000, 1000000 * mpu.get_data_parallel_world_size(with_context_parallel=True)
                )
            # Set bucket_size to infinity if overlap_grad_reduce is False.
            if not ddp_config.overlap_grad_reduce:
                ddp_config.bucket_size = None

        model = [DP(config=config,
                     ddp_config=ddp_config,
                     module=model_chunk,
                     # Turn off bucketing for model_chunk 2 onwards, since communication for these
                     # model chunks is overlapped with compute anyway.
                     disable_bucketing=(model_chunk_idx > 0) or args.overlap_param_gather_with_optimizer_step)
                 for (model_chunk_idx, model_chunk) in enumerate(model)]

        # Broadcast params from data parallel src rank to other data parallel ranks.
        if args.data_parallel_random_init:
            for model_module in model:
                model_module.broadcast_params()

    return model
