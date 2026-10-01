# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import os
import dataclasses
from copy import deepcopy
from functools import wraps
import contextlib
import random
import sys

import numpy as np

import torch

from megatron.core import mpu, tensor_parallel
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
from megatron.core.transformer.moe import upcycling_utils
from megatron.core.num_microbatches_calculator import update_num_microbatches
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.training import get_args, print_rank_0, one_logger_utils, wandb_utils, ft_integration
from megatron.training.training import get_model, get_optimizer_param_scheduler, preprocess_common_state_dict
from megatron.training.global_vars import get_timers, get_one_logger
from megatron.training.utils import unwrap_model, update_use_dist_ckpt, is_last_rank
from megatron.training.checkpointing import (
    CheckpointType,
    checkpoint_exists,
    check_checkpoint_args,
    fix_fp8_params_lose_precision_when_loading_dist_ckpt,
    fix_query_key_value_ordering,
    generate_state_dict,
    get_checkpoint_tracker_filename,
    get_checkpoint_version,
    get_checkpoint_name,
    get_distributed_optimizer_checkpoint_name,
    get_rng_state,
    read_metadata,
    set_checkpoint_version,
    save_checkpoint,
    _load_base_checkpoint,
    _to_dtensor
)
try:
    from megatron.core.distributed import TorchFullyShardedDataParallel as torch_FSDP

    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False

try:
    from modelopt.torch.opt.plugins import (
        save_modelopt_state,
        save_sharded_modelopt_state,
        restore_modelopt_state,
        restore_sharded_modelopt_state,
    )
    has_nvidia_modelopt = True
except Exception:
    has_nvidia_modelopt = False

from mindspeed_mm.utils.transformer_model_config import get_model_config
from mindspeed_mm.models.common.mm_gpt_model import MMGPTModel
from mindspeed_mm.models.common.module_spec.get_layer_spec import get_llm_layer_spec
from mindspeed_mm.tasks.finetune.lora.utils import is_enable_lora


def setup_model_and_optimizer_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        args = (model_provider,) + args[1:]
        return func(*args, **kwargs)

    return wrapper


def setup_model_and_optimizer(model_provider_func,
                              model_type,
                              no_wd_decay_cond=None,
                              scale_lr_cond=None,
                              lr_mult=1.0,
                              checkpointing_context=None):
    """Setup model and optimizer."""
    args = get_args()
    timers = get_timers()
    one_logger = get_one_logger()

    model = get_model(model_provider, model_type)
    unwrapped_model = unwrap_model(model)

    kwargs = {}
    for f in dataclasses.fields(OptimizerConfig):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    config = OptimizerConfig(**kwargs)
    config.timers = timers
    optimizer = get_megatron_optimizer(config, model, no_wd_decay_cond,
                                       scale_lr_cond, lr_mult,
                                       use_gloo_process_groups=args.enable_gloo_process_groups)
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

    if args.moe_use_upcycling:
        torch.distributed.barrier()
        if not checkpoint_exists(args.save):
            raise AssertionError(
                "The upcycling destination directory already exists. "
                "Please check if --moe-use-upcycling is mistakenly enabled. "
                "Upcycling should only be set for the first run when converting the dense model. "
                "All subsequent runs should remove this flag. "
            )
        num_experts = args.num_experts
        args.num_experts = None
        expert_model_parallel_size = args.expert_model_parallel_size
        args.expert_model_parallel_size = 1
        dense_model_for_upcycling = get_model(model_provider_func, model_type)
        args.num_experts = num_experts
        args.expert_model_parallel_size = expert_model_parallel_size
        _, args.num_floating_point_operations_so_far = upcycling_utils.load_and_upcycle_model(
            load_checkpoint,
            unwrapped_model,
            dense_model_for_upcycling,
            load_kwargs={'model': dense_model_for_upcycling, 'optimizer': None, 'opt_param_scheduler': None}
        )
        args.iteration = 1
        save_checkpoint(args.iteration, model, None, None, args.num_floating_point_operations_so_far)
        torch.distributed.barrier()
        del dense_model_for_upcycling
        if (args.fp16 or args.bf16) and optimizer is not None:
            optimizer.reload_model_params()
        print_rank_0(f'Upcycled checkpoint saved to {args.save}')

    if (args.load is not None or args.pretrained_checkpoint is not None) and not args.moe_use_upcycling:
        one_logger and one_logger.log_metrics({
            'load_checkpoint_start_time': one_logger_utils.get_timestamp_in_ms()
        })
        timers('load-checkpoint', log_level=0).start(barrier=True)

        args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
                model, optimizer, opt_param_scheduler, checkpointing_context=checkpointing_context,
                skip_load_to_model_and_opt=HAVE_FSDP2 and getattr(args, "use_torch_fsdp2", False) and args.ckpt_format == "torch_dist")
        timers('load-checkpoint').stop(barrier=True)
        timers.log(['load-checkpoint'])
        one_logger and one_logger.log_metrics({
            'load_checkpoint_finish_time': one_logger_utils.get_timestamp_in_ms(),
            'load_checkpoint_time': timers('load-checkpoint').active_time()
        })
    else:
        args.iteration = 0
        args.num_floating_point_operations_so_far = 0

    # get model without FP16 and/or DDP wrappers
    if args.iteration == 0 and len(unwrapped_model) == 1 \
        and hasattr(unwrapped_model[0], 'init_state_dict_from_bert'):
        print_rank_0("Initializing ICT from pretrained BERT model")
        unwrapped_model[0].init_state_dict_from_bert()
        if args.fp16:
            optimizer.reload_model_params()

    # Convert checkpoint format.
    if args.ckpt_convert_format is not None:
        load_ckpt_format = args.ckpt_format
        args.ckpt_format = args.ckpt_convert_format
        args.save = os.path.join(args.ckpt_convert_save, args.ckpt_convert_format)
        update_use_dist_ckpt(args)

        save_checkpoint(args.iteration, model, optimizer, opt_param_scheduler,
                        args.num_floating_point_operations_so_far,
                        preprocess_common_state_dict_fn=preprocess_common_state_dict)

        print_rank_0("> converted checkpoint: %s -> %s." % (load_ckpt_format, args.ckpt_format))
        torch.distributed.barrier()

    return model, optimizer, opt_param_scheduler


def load_checkpoint(ddp_model, optimizer, opt_param_scheduler, load_arg='load', strict=True,
                    checkpointing_context=None, skip_load_to_model_and_opt=False):
    """Load a model checkpoint and return the iteration.
    strict (bool): whether to strictly enforce that the keys in
        :attr:`state_dict` of the checkpoint match the names of
        parameters and buffers in model.
    skip_load_to_model_and_opt (bool): whether to call `load_state_dict`
        for :attr:`model` and :attr:`optimizer`. In case of running FSDP2 with mcore distributed
        checkpointing, the tensors are already loaded in-place by `_load_base_checkpoint`.
    """
    args = get_args()
    if is_enable_lora() and args.load_base_model is None:
        strict = False
    load_dir = getattr(args, load_arg)

    # Finetuning directories
    pretrained_dir = getattr(args, 'pretrained_checkpoint', None)
    if pretrained_dir is not None and not checkpoint_exists(load_dir):
        print_rank_0(
            f'Checkpoint file not found in load directory {load_dir} attempting to finetune with checkpoint in {pretrained_dir}'
        )
        load_dir = pretrained_dir
        if not checkpoint_exists(load_dir):
            raise FileNotFoundError("No checkpoint found in load directory or pretrained directory")
        args.finetune = True

    model = unwrap_model(ddp_model)

    ckpt_format = args.ckpt_format
    if args.auto_detect_ckpt_format or ckpt_format == "torch_dist":
        state_dict, checkpoint_name, release, ckpt_type = _load_base_checkpoint(
            load_dir,
            args,
            rank0=True,
            checkpointing_context=checkpointing_context,
        )

        ckpt_format = None
        if ckpt_type == CheckpointType.TORCH_DCP:
            ckpt_format = "torch_dcp"
        elif ckpt_type == CheckpointType.LEGACY:
            ckpt_format = "torch"
        elif ckpt_type in [CheckpointType.LOCAL, CheckpointType.GLOBAL]:
            ckpt_format = "torch_dist"
        elif ckpt_type is None:
            pass    # Not loaded.
        else:
            raise NotImplementedError(f"checkpoint format {ckpt_format} not supported")

    load_kwargs = {}
    if ckpt_format == "torch_dist":
        ckpt_tp_pp = (
            state_dict['args'].tensor_model_parallel_size,
            state_dict['args'].pipeline_model_parallel_size,
            getattr(state_dict['args'], 'encoder_tensor_model_parallel_size', 0),
            getattr(state_dict['args'], 'encoder_pipeline_model_parallel_size', 0),
        )
        run_tp_pp = (
            args.tensor_model_parallel_size,
            args.pipeline_model_parallel_size,
            getattr(args, 'encoder_tensor_model_parallel_size', 0),
            getattr(args, 'encoder_pipeline_model_parallel_size', 0),
        )
        mismatch_msg = "(TP, PP, encoder TP, encoder PP) mismatch after resume ({} vs {} from checkpoint)".format(
            run_tp_pp, ckpt_tp_pp
        )

        # Determine if RNG state will be loaded
        allow_load_rng = (
            not release
            and not args.finetune
            and not args.no_load_rng
            and not getattr(state_dict['args'], 'no_save_rng', False)
        )

        if ckpt_tp_pp == run_tp_pp and allow_load_rng:
            gen_sd_rng_state = get_rng_state(args.ckpt_format)  # we can load the rng state
        else:
            gen_sd_rng_state = None
            if ckpt_tp_pp != run_tp_pp:
                print_rank_0("{}: RNG state will be ignored".format(mismatch_msg))

        optim_sd_kwargs = dict(is_loading=True)
        # Determine if optimizer state will be loaded
        can_load_optimizer = (
            not release
            and not args.finetune
            and not args.no_load_optim
            and not getattr(state_dict['args'], 'no_save_optim', False)
        )
        if can_load_optimizer:
            gen_sd_optim = optimizer
            gen_sd_opt_param_scheduler = opt_param_scheduler

            if args.use_distributed_optimizer:
                optim_sd_kwargs['sharding_type'] = ('fully_sharded_model_space'
                                                    if getattr(state_dict['args'], 'ckpt_fully_parallel_save', False)
                                                    else 'dp_zero_gather_scatter')
                # This is for backwards-compatibility. Can be removed once 'fully_sharded_bucket_space' loading is removed
                for maybe_dist_opt_optim_state in (state_dict['optimizer'], *state_dict['optimizer'].values()):
                    if 'param_state_sharding_type' in maybe_dist_opt_optim_state:
                        if maybe_dist_opt_optim_state['param_state_sharding_type'] == 'fully_sharded_bucket_space':
                            print_rank_0('Detected deprecated `fully_sharded_bucket_space` DistributedOptimizer checkpoint format')
                            optim_sd_kwargs['sharding_type'] = maybe_dist_opt_optim_state['param_state_sharding_type']
                        break

                if ckpt_tp_pp != run_tp_pp and optim_sd_kwargs['sharding_type'] != 'fully_sharded_model_space':
                    raise RuntimeError(f"{mismatch_msg}: not supported for DistributedOptimizer with sharding type {optim_sd_kwargs['sharding_type']}."
                                        f" Please use `--ckpt-fully-parallel-save` flag during checkpoint saving.")
        else:
            gen_sd_optim = None
            gen_sd_opt_param_scheduler = None

        # Determine if rerun state will be loaded
        can_load_rerun_state = (
            ckpt_tp_pp == run_tp_pp
            and not release
            and not args.finetune
            and 'rerun_state_machine' in state_dict
        )
        if can_load_rerun_state:
            rerun_state_machine = get_rerun_state_machine()
            gen_sd_rerun_state = rerun_state_machine.state_dict(
                data_iterator=None, ckpt_format=ckpt_format,
            )
        else:
            gen_sd_rerun_state = None
            if ckpt_tp_pp != run_tp_pp:
                print_rank_0("{}: Rerun state will be ignored".format(mismatch_msg))

        # [ModelOpt]: IMPORTANT! Restoring modelopt_state (sharded or not) must be performed
        # after the model instance has been created and before _load_base_checkpoint is called.
        if has_nvidia_modelopt:
            if ckpt_type == CheckpointType.LOCAL:
                print_rank_0('WARNING: Local checkpointing does not support nvidia_modelopt.')
            elif ckpt_type == CheckpointType.GLOBAL:
                restore_modelopt_state(model, state_dict)
            else:
                restore_sharded_modelopt_state(model, checkpoint_name)

        # [ModelOpt]: Initial loading from non-resume sharded checkpoint to a Distillation Model
        # will result in key mismatch with loss modules potentially containing parameters, since
        # it requires generating a state_dict before loading. Here we hide those modules if present.
        with contextlib.ExitStack() as stack:  # Allows multiple context managers for each model shard
            if args.finetune and hasattr(model[0], "hide_loss_modules"):
                for m in model:
                    stack.enter_context(m.hide_loss_modules())
            load_kwargs['sharded_state_dict'] = generate_state_dict(
                args, model, gen_sd_optim, gen_sd_opt_param_scheduler, gen_sd_rng_state,
                optim_sd_kwargs=optim_sd_kwargs, rerun_state=gen_sd_rerun_state
            )

        # When "--fp8-param-gather" is disabled, this function doesn't modify anything.
        fix_fp8_params_lose_precision_when_loading_dist_ckpt(load_kwargs['sharded_state_dict'])
    elif args.ckpt_format == "torch_dcp":
        model_sd = model[0].state_dict()
        optimizer_sd = optimizer.state_dict(is_loading=True)
        sharded_state_dict = {
            "model": model_sd,
            "optimizer": optimizer_sd,
            "args": None,
            "iteration": 1,
            "rng_state": get_rng_state(args.ckpt_format),
            "checkpoint_version": None,
            "opt_param_scheduler": opt_param_scheduler.state_dict(),
            "num_floating_point_operations_so_far": 0,
        }
        load_kwargs["sharded_state_dict"] = sharded_state_dict


    state_dict, checkpoint_name, release, ckpt_type = _load_base_checkpoint(
        load_dir, args, rank0=False, checkpointing_context=checkpointing_context,
        **load_kwargs
    )

    # Checkpoint not loaded.
    if state_dict is None:
        # Iteration and num_floating_point_operations_so_far default to 0.
        return 0, 0

    # Set checkpoint version.
    set_checkpoint_version(state_dict.get('checkpoint_version', 0))

    # Convert to regular torch tensor to DTensor.
    if ckpt_type == CheckpointType.LEGACY and args.ckpt_format == "torch_dcp":
        dtensor_state_dict = _to_dtensor(ddp_model, state_dict["model"])
        state_dict["model"] = dtensor_state_dict

    # Set iteration.
    if args.finetune or release:
        iteration = 0
    else:
        try:
            iteration = state_dict['iteration']
        except KeyError:
            try:  # Backward compatible with older checkpoints
                iteration = state_dict['total_iters']
            except KeyError as e:
                print_rank_0('A metadata file exists but unable to load '
                             'iteration from checkpoint {}, exiting'.format(checkpoint_name))
                raise RuntimeError(f"Failed to load iteration from checkpoint {checkpoint_name}") from e
    num_floating_point_operations_so_far = state_dict.get('num_floating_point_operations_so_far', 0)

    # Check arguments.
    if not args.consumed_train_samples == 0:
        raise ValueError()
    if not args.skipped_train_samples == 0:
        raise ValueError()
    if not args.consumed_valid_samples == 0:
        raise ValueError()
    if 'args' in state_dict and not args.finetune:
        checkpoint_args = state_dict['args']
        check_checkpoint_args(checkpoint_args)
        args.consumed_train_samples = getattr(checkpoint_args,
                                              'consumed_train_samples', 0)
        args.skipped_train_samples = getattr(checkpoint_args,
                                             'skipped_train_samples', 0)
        update_num_microbatches(consumed_samples=args.consumed_train_samples, verbose=True)
        args.consumed_valid_samples = getattr(checkpoint_args,
                                              'consumed_valid_samples', 0)
    else:
        print_rank_0('could not find arguments in the checkpoint ...')

    # Model.
    strict = False if args.retro_add_retriever else strict
    if not skip_load_to_model_and_opt:
        if len(ddp_model) == 1:
            ddp_model[0].load_state_dict(state_dict['model'], strict=strict)
        else:
            for i, _ in enumerate(ddp_model):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                ddp_model[i].load_state_dict(state_dict['model%d' % i], strict=strict)

    # Fix up query/key/value matrix ordering if needed.
    checkpoint_version = get_checkpoint_version()
    print_rank_0(f' checkpoint version {checkpoint_version}')
    fix_query_key_value_ordering(model, checkpoint_version)

    # Optimizer.
    if not release and not args.finetune and not args.no_load_optim:
        try:
            # Load state dict.
            if not skip_load_to_model_and_opt and optimizer is not None and not optimizer.is_stub_optimizer:
                optimizer.load_state_dict(state_dict['optimizer'])

            # Load distributed optimizer's custom parameter state.
            # For distributed checkpoint it's already loaded in load_state_dict above
            is_torch_dist = ckpt_format == "torch_dist"
            if args.use_distributed_optimizer and not is_torch_dist:
                # NOTE: this is a manual read of the tracker file.
                # This code should not be reached when reading from a non_persistent checkpoint
                if is_torch_dist:
                    raise AssertionError()
                tracker_filename = get_checkpoint_tracker_filename(load_dir)
                iteration, release = read_metadata(tracker_filename)
                model_checkpoint_name = \
                    get_checkpoint_name(load_dir, iteration, release)
                optim_checkpoint_name = \
                    get_distributed_optimizer_checkpoint_name(
                        model_checkpoint_name)
                optimizer.load_parameter_state(optim_checkpoint_name,
                                               update_legacy_format=args.ckpt_convert_update_legacy_dist_opt_format)

            # Load scheduler.
            if opt_param_scheduler is not None:
                if 'lr_scheduler' in state_dict: # backward compatbility
                    opt_param_scheduler.load_state_dict(state_dict['lr_scheduler'])
                else:
                    opt_param_scheduler.load_state_dict(state_dict['opt_param_scheduler'])
        except KeyError as e:
            print_rank_0('Unable to load optimizer from checkpoint {}. '
                         'Specify --no-load-optim or --finetune to prevent '
                         'attempting to load the optimizer state, '
                         'exiting ...'.format(checkpoint_name))
            raise e
    else:
        if (args.fp16 or args.bf16) and optimizer is not None:
            optimizer.reload_model_params()

    # rerun state
    try:
        if 'rerun_state_machine' in state_dict:
            get_rerun_state_machine().load_state_dict(state_dict['rerun_state_machine'])
    except Exception as e:
        print(f"Unable to restore RerunMachine from checkpoint: {e}")
        raise RuntimeError("Unable to restore RerunMachine from checkpoint") from e

    # rng states.
    if not release and not args.finetune and not args.no_load_rng:
        try:
            if 'rng_state' in state_dict:
                # access rng_state for data parallel rank
                if args.data_parallel_random_init:
                    rng_state = state_dict['rng_state'][mpu.get_data_parallel_rank()]
                else:
                    rng_state = state_dict['rng_state'][0]
                random.setstate(rng_state['random_rng_state'])
                np.random.set_state(rng_state['np_rng_state'])
                torch.set_rng_state(rng_state['torch_rng_state'])
                torch.cuda.set_rng_state(rng_state['cuda_rng_state'])
                # Check for empty states array
                if not rng_state['rng_tracker_states']:
                    raise KeyError
                tensor_parallel.get_cuda_rng_tracker().set_states(
                    rng_state['rng_tracker_states'])
            else:  # backward compatability
                random.setstate(state_dict['random_rng_state'])
                np.random.set_state(state_dict['np_rng_state'])
                torch.set_rng_state(state_dict['torch_rng_state'])
                torch.cuda.set_rng_state(state_dict['cuda_rng_state'])
                # Check for empty states array
                if not state_dict['rng_tracker_states']:
                    raise KeyError
                tensor_parallel.get_cuda_rng_tracker().set_states(
                    state_dict['rng_tracker_states'])
        except KeyError as e:
            print_rank_0('Unable to load rng state from checkpoint {}. '
                         'Specify --no-load-rng or --finetune to prevent '
                         'attempting to load the rng state, '
                         'exiting ...'.format(checkpoint_name))
            raise RuntimeError("Failed to load rng state from checkpoint") from e

    # Some utilities want to load a checkpoint without distributed being initialized
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_0(f'  successfully loaded checkpoint from {load_dir} '
                 f'[ t {mpu.get_tensor_model_parallel_rank() + 1}/{mpu.get_tensor_model_parallel_world_size()}, '
                 f'p {mpu.get_pipeline_model_parallel_rank() + 1}/{mpu.get_pipeline_model_parallel_world_size()} ] '
                 f'at iteration {iteration}')

    # Additional callback for wandb (last rank)
    if not torch.distributed.is_initialized() \
       or is_last_rank():
        wandb_utils.on_load_checkpoint_success(checkpoint_name, load_dir)

    if iteration > 0:
        # Notify FT that a checkpoint was loaded.
        is_local_chkpt = (ckpt_type == CheckpointType.LOCAL)
        ft_integration.on_checkpoint_loaded(is_local_chkpt=is_local_chkpt)

    return iteration, num_floating_point_operations_so_far


def model_provider(pre_process=True, post_process=True, modules=None):
    """Builds the model."""
    if modules is None:
        modules = ['image_encoder', 'audio_encoder', 'text_decoder']

    args = get_args()
    print_rank_0("building VLMModel ...")
    vlm_config = deepcopy(args.mm.model)

    # distinguish model construct stage when pipeline parallel
    vlm_config.pre_process = pre_process
    vlm_config.post_process = post_process

    _configure_modules(vlm_config, modules)

    from mindspeed_mm.models.vlm_model import VLMModel

    class LDTVLMModel(VLMModel):
        def __init__(self, config):
            super().__init__(config)

        def _build_text_decoder_model(self, config):
            if self.pp_size <= 1:
                return MMGPTModel(
                    config=config,
                    transformer_layer_spec=get_llm_layer_spec(config),
                    vocab_size=config.vocab_size,
                    max_sequence_length=config.max_position_embeddings,
                    parallel_output=config.parallel_output,
                    position_embedding_type=config.position_embedding_type,
                    share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                    rotary_base=config.rope_theta if getattr(config, 'rope_theta', None) else config.rotary_base,
                    pre_process=self.pre_process,
                    post_process=self.post_process,
                    reward_process=self.reward_process
                )
            if self.enable_vp:
                if self.pp_size * self.vp_size != len(config.pipeline_num_layers) * len(config.pipeline_num_layers[0]):
                    raise ValueError(
                        f"The product of pipeline-model-parallel-size and vpp-size must equal to the total number of stage in pipeline_num_layers, "
                        f"but got pipeline-model-parallel-size: {self.pp_size}, vpp-size: {self.vp_size}, "
                        f"and total number of stage in pipeline_num_layers: {len(config.pipeline_num_layers) * len(config.pipeline_num_layers[0])}.")
            elif self.pp_size != len(config.pipeline_num_layers):
                raise ValueError(f"length of pipeline_num_layers must equal to pipeline-model-parallel-size, "
                                f"but got pipeline_num_layers length:{len(config.pipeline_num_layers)} "
                                f"and pipeline-model-parallel-size:{self.pp_size}.")

            if self.enable_vp:
                local_num_layers = config.pipeline_num_layers[self.vp_rank][self.pp_rank]
            else:
                local_num_layers = config.pipeline_num_layers[self.pp_rank]

            if local_num_layers == 0:
                if mpu.get_virtual_pipeline_model_parallel_rank() == 0 or not mpu.is_pipeline_first_stage(ignore_virtual=True):
                    self.add_text_decoder = False
                    return None

            if self.enable_vp:
                pipeline_start_index = sum(
                    sum(vp_layer) for vp_layer in config.pipeline_num_layers[:self.vp_rank]) + sum(
                    config.pipeline_num_layers[self.vp_rank][:self.pp_rank])
                pipeline_end_index = sum(sum(vp_layer) for vp_layer in config.pipeline_num_layers[:self.vp_rank]) + sum(
                    config.pipeline_num_layers[self.vp_rank][:self.pp_rank + 1])
            else:
                pipeline_start_index = sum(config.pipeline_num_layers[:self.pp_rank])
                pipeline_end_index = sum(config.pipeline_num_layers[:self.pp_rank + 1])

            pre_process = pipeline_start_index == 0
            post_process = pipeline_end_index == config.num_layers

            # PP=0, VPP=1: text_decoder embedding
            # PP=0, VPP=2: text_decoder unembedding
            if mpu.is_pipeline_first_stage(ignore_virtual=True):
                if mpu.get_virtual_pipeline_model_parallel_rank() == 1:
                    pre_process = True
                    post_process = False
                elif mpu.get_virtual_pipeline_model_parallel_rank() == 2:
                    pre_process = False
                    post_process = True
                else:
                    pre_process = post_process = False
            else:
                pre_process = post_process = False

            print(
                f"text decoder pipeline config:\
                pp_rank:{self.pp_rank},\
                pre_process:{pre_process},\
                post_process:{post_process},\
                local_num_layers:{local_num_layers}"
            )
            # num_layers will be divided by pp_size in TransformerBlock from megatron.core
            config.num_layers = self.pp_size * local_num_layers
            if self.enable_vp:
                config.num_layers *= self.vp_size
            return MMGPTModel(
                config=config,
                transformer_layer_spec=get_llm_layer_spec(config),
                vocab_size=config.vocab_size,
                max_sequence_length=config.max_position_embeddings,
                parallel_output=config.parallel_output,
                position_embedding_type=config.position_embedding_type,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                rotary_base=config.rope_theta if getattr(config, 'rope_theta', None) else config.rotary_base,
                pre_process=pre_process,
                post_process=post_process,
                reward_process=self.reward_process
            )

    model = LDTVLMModel(vlm_config)

    _apply_freezing(model, vlm_config)

    return model


# copy from: pretrain_vlm.py
def _configure_modules(vlm_config, modules):
    """Configure each module based on the modules list."""
    module_configs = {
        'image_encoder': _configure_image_encoder,
        'audio_encoder': _configure_audio_encoder,
        'text_decoder': _configure_text_decoder
    }

    for module_name, config_func in module_configs.items():
        if module_name in modules and hasattr(vlm_config, module_name):
            config_func(vlm_config)
        else:
            setattr(vlm_config, module_name, None)


# copy from: pretrain_vlm.py
def _configure_image_encoder(vlm_config):
    """Configure image encoder module."""

    # MindSpeed needs to validate the CP configuration; the attention head must be divisible by the CP sizes.
    # However, since the vision projector does not have an attention head, special handling is required.
    vlm_config.image_encoder.vision_projector.context_parallel_size = 1
    vlm_config.image_encoder.vision_encoder.expert_model_parallel_size = 1
    vlm_config.image_encoder.vision_projector.expert_model_parallel_size = 1
    vlm_config.image_encoder.vision_encoder = get_model_config(vlm_config.image_encoder.vision_encoder)
    vlm_config.image_encoder.vision_projector = get_model_config(vlm_config.image_encoder.vision_projector)


# copy from: pretrain_vlm.py
def _configure_audio_encoder(vlm_config):
    """Configure audio encoder module."""

    vlm_config.audio_encoder.audio_encoder = get_model_config(vlm_config.audio_encoder.audio_encoder)


# copy from: pretrain_vlm.py
def _configure_text_decoder(vlm_config):
    """Configure text decoder module."""

    vlm_config.text_decoder = get_model_config(vlm_config.text_decoder)


# copy from: pretrain_vlm.py
def _apply_freezing(model, vlm_config):
    """Apply freezing settings to the model."""
    has_image = hasattr(vlm_config, 'image_encoder') and vlm_config.image_encoder is not None
    freeze_image_encoder = has_image and getattr(vlm_config.image_encoder.vision_encoder, 'freeze', True)
    freeze_image_projection = has_image and getattr(vlm_config.image_encoder.vision_projector, 'freeze', False)

    has_audio = hasattr(vlm_config, 'audio_encoder') and vlm_config.audio_encoder is not None
    freeze_audio_encoder = has_audio and getattr(vlm_config.audio_encoder.audio_encoder, 'freeze', True)

    model.freeze(
        freeze_image_encoder=freeze_image_encoder,
        freeze_image_projection=freeze_image_projection,
        freeze_audio_encoder=freeze_audio_encoder
    )
