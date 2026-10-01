# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from datetime import datetime
import os
import gc
import sys
import time
from importlib.metadata import version
import torch
import torch_npu

from megatron.core import mpu
from megatron.core.utils import get_model_config
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads
from megatron.training.checkpointing import save_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.training.initialize import set_jit_fusion_options
from megatron.training.initialize import write_args_to_tensorboard
from megatron.training.global_vars import (
    get_args,
    get_signal_handler,
    get_timers,
    get_tensorboard_writer,
    get_wandb_writer,
    get_one_logger,
)
from megatron.core.num_microbatches_calculator import (
    get_current_global_batch_size,
    get_num_microbatches,
    update_num_microbatches,
)
from megatron.training.training import (
    report_memory,
    report_theoretical_memory,
    track_moe_metrics,
    evaluate_and_print_results,
    save_checkpoint_and_time,
    print_datetime,
    num_floating_point_operations,
    get_one_logger,
    append_to_progress_log,
    build_train_valid_test_data_iterators,
    setup_model_and_optimizer,
    disable_forward_pre_hook,
    get_model,
    load_checkpoint,
)
from megatron.training.utils import (
    calc_params_l2_norm,
    check_adlr_autoresume_termination,
    print_rank_0,
    print_rank_last,
    unwrap_model,
)
from mindspeed.core.multi_modal.dist_train.dist_train_config import is_forward_only_model
from megatron.training.arguments import parse_args
from megatron.training.global_vars import set_args

from mindspeed.arguments import parse_args_wrapper
from mindspeed_mm.configs.config import merge_mm_args
from mindspeed_mm.tools.profiler import Profiler
from mindspeed_mm.tools.mem_profiler import memory_profiler
from mindspeed_mm.arguments import extra_args_provider_decorator
from mindspeed_mm.patchs.patch_manager import PatchesManager
from mindspeed_mm.patchs.ep_patch import finalize_model_grads_wrapper
from mindspeed_mm.utils.data_balance.data_balance import GBSImageDataBalance
from mindspeed_mm.utils.random import seed_all
from mindspeed_mm.utils.dpcp_utils import (
    data_aware_parallel_optimize,
    is_use_dynamic_dpcp,
    initialize_parall_switch_list
)
from mindspeed_mm.utils.auto_setting import (
    auto_settings_fun,
    auto_settings_parse_args,
    auto_settings_parse_model,
    auto_settings_profile,
    train_decorator,
    train_step_decorator
)


_TRAIN_START_TIME = time.time()


def pretrain(
    train_valid_test_dataset_provider,
    model_provider,
    model_type,
    forward_step_func,
    process_non_loss_data_func=None,
    extra_args_provider=None,
    args_defaults=None,
):
    """
    Main training program.

    This function will run the following in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the model using the forward_step_func.

    Args:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        model_type: an enum that specifies the type of model being trained.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        process_non_loss_data_func: a function to post process outputs of the
            network. It can be used for dumping output tensors (e.g images) to
            tensorboard. It takes `collected data`(list of tensors),
            `current iteration index` and `tensorboard writer` as arguments.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
    """
    args_defaults = {} if args_defaults is None else args_defaults
    extra_args_provider = extra_args_provider_decorator(extra_args_provider)

    new_parse_args = parse_args_wrapper(parse_args)
    argument = new_parse_args(extra_args_provider, False)
    if getattr(argument, "auto_parallel_mm", False):
        set_args(argument)
        from mindspeed.core.auto_parallel.mm_search.optimizer import auto_parallel_mm_search_optimal_config
        auto_parallel_mm_search_optimal_config(argument)
        return

    # Initialize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(
        extra_args_provider=extra_args_provider, args_defaults=args_defaults
    )

    argument = get_args()
    if argument.auto_settings:
        auto_settings_fun(argument)
        return

    if (os.getenv("OOTB_OPTIMIZER_PARSE_ARGS", "FALSE") == "TRUE"):
        auto_settings_parse_args()
        return

    init_func = args_defaults.get("init_func", None)
    if init_func:
        init_func()

    args = get_args()
    if is_use_dynamic_dpcp():
        from datetime import timedelta
        timeout = timedelta(minutes=10)
        initialize_parall_switch_list(timeout)
        print_rank_0("dynamic dpcp is enabled")
    merge_mm_args(args)

    if args.log_throughput:
        print("[WARNING] Currently, the calculation of TFLOPS is incorrect for multimodal models. "
                             "Please do not use this as a reference for performance.")

    if hasattr(args, "mm") and getattr(args, "profile_subgraph_seg", False):
        from mindspeed.core.auto_parallel.mm_search.profiling import set_profile_model_config
        set_profile_model_config(args)

    if not hasattr(args, "dist_train"):
        args.dist_train = False

    # add deterministic computing function
    if args.use_deter_comp:
        seed_all(args.seed)
        print_rank_0("deterministic computing is applied for npu.")

    if args.jit_compile:
        torch_npu.npu.set_compile_mode(jit_compile=True)

    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.npu.config.allow_internal_format = args.allow_internal_format

    timers = get_timers()

    # apply patches
    PatchesManager.apply_patches_from_config()

    if args.log_progress:
        append_to_progress_log("Starting job")

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options()

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = torch.tensor(
        [_TRAIN_START_TIME], dtype=torch.float, device="cuda"
    )
    torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()
    print_rank_0(
        "time to initialize megatron (seconds): {:.3f}".format(
            time.time() - _TRAIN_START_TIME
        )
    )
    print_datetime("after megatron is initialized")

    args = get_args()
    if args.save_interval == 0 or args.log_interval == 0 or args.eval_interval == 0:
        raise ValueError("save_interval, log_interval, and eval_interval cannot be 0")
    timers = get_timers()

    one_logger = get_one_logger()
    if one_logger:
        one_logger.log_metrics({"train_iterations_warmup": 5})

    memory_profiler.reset(args.mm.tool.memory_profile)

    # Model, optimizer, and learning rate.
    timers("model-and-optimizer-setup", log_level=0).start(barrier=True)
    lr_mult = args.lr_mult
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider, model_type, no_wd_decay_cond=no_wd_decay_cond, scale_lr_cond=scale_lr_cond, lr_mult=lr_mult)

    if hasattr(optimizer, "chained_optimizers") and args.use_torch_fsdp2:
        for sub_optimizer in optimizer.chained_optimizers:
            if getattr(sub_optimizer, "is_moe_param") == "moe":
                from mindspeed_mm.models.transformers.global_vars import get_ep_group
                setattr(sub_optimizer, "grad_stats_parallel_group", get_ep_group())

    if getattr(args, "auto_parallel_profile", False):
        from mindspeed.core.auto_parallel.mm_search.memory_modeling import count_module_param
        from mindspeed.core.auto_parallel.mm_search.help import PROFILE_CONTENT
        module_param_dict = count_module_param(model)
        PROFILE_CONTENT['module_param'] = module_param_dict

    timers("model-and-optimizer-setup").stop()
    print_datetime("after model, optimizer, and learning rate scheduler are built")
    config = get_model_config(model[0])

    if (os.getenv("OOTB_OPTIMIZER_PARSE_MODEL", "FALSE") == "TRUE"):
        auto_settings_parse_model(model, mpu, args)
        return

    # Data stuff.
    timers("train/valid/test-data-iterators-setup", log_level=0).start(barrier=True)
    if args.virtual_pipeline_model_parallel_size is not None:
        train_data_iterator = []
        valid_data_iterator = []
        test_data_iterator = []
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            iterators = build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider
            )
            train_data_iterator.append(iterators[0])
            valid_data_iterator.append(iterators[1])
            test_data_iterator.append(iterators[2])
    else:
        train_data_iterator, valid_data_iterator, test_data_iterator = (
            build_train_valid_test_data_iterators(train_valid_test_dataset_provider)
        )
    timers("train/valid/test-data-iterators-setup").stop()
    print_datetime("after dataloaders are built")

    # Print setup timing.
    print_rank_0("done with setup ...")
    timers.log(
        ["model-and-optimizer-setup", "train/valid/test-data-iterators-setup"],
        barrier=True,
    )

    if not args.skip_train:
        print_rank_0("training ...")

        if args.dataloader_type == "cyclic" and args.retro_project_dir:
            if args.retro_cyclic_train_iters is None:
                raise AssertionError
            args.train_iters = args.retro_cyclic_train_iters
            print_rank_0("retro cyclic train iters : %d" % args.train_iters)

        iteration = 0
        if args.do_train and args.train_iters > 0:
            iteration, num_floating_point_operations_so_far = train(
                forward_step_func,
                model,
                optimizer,
                opt_param_scheduler,
                train_data_iterator,
                valid_data_iterator,
                process_non_loss_data_func,
                config,
            )

        print_datetime("after training is done")

        if judge_save_checkpoint(args, iteration):
            save_checkpoint(
                iteration,
                model,
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
            )
    else:
        print_rank_0("skipping training (--skip-train is on) ...")

        iteration = args.iteration

    if args.do_valid:
        prefix = f"iteration {iteration} on validation set"
        evaluate_and_print_results(
            prefix,
            forward_step_func,
            valid_data_iterator,
            model,
            iteration,
            process_non_loss_data_func,
            config,
            verbose=True,
            write_to_tensorboard=not args.skip_train,
        )

    if args.do_test:
        prefix = f"iteration {iteration} on test set"
        evaluate_and_print_results(
            prefix,
            forward_step_func,
            test_data_iterator,
            model,
            iteration,
            process_non_loss_data_func,
            config,
            verbose=True,
            write_to_tensorboard=not args.skip_train,
        )

    # profiling parser
    if os.getenv('OOTB_OPTIMIZER_PROFILING', 'FALSE') == 'TRUE':
        auto_settings_profile(args)


@train_decorator
def train(
    forward_step_func,
    model,
    optimizer,
    opt_param_scheduler,
    train_data_iterator,
    valid_data_iterator,
    process_non_loss_data_func,
    config,
    call_backs=None,
):
    """Train the model function."""
    args = get_args()
    timers = get_timers()

    # Write args to tensorboard
    write_args_to_tensorboard()

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Data balance initialize
    if args.use_data_balance:
        print_rank_0("[INFO] initializing data_balance ...")
        data_balance_algo = GBSImageDataBalance(
            args.virtual_pipeline_model_parallel_size,
            args.mm_model,
            args.data_balance_sorting_algo,
            len(model),
            train_data_iterator
        )
        print_rank_0("[INFO] initialize GBS image data balance successfully")
        print_rank_0(f"[INFO] image encoder DP (in DataBalance): {data_balance_algo.image_encoder_dp}")
    else:
        data_balance_algo = None

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = args.iteration
    one_logger = get_one_logger()
    if one_logger:
        iteration_start = iteration
        train_samples_start = args.consumed_train_samples
        train_samples_target = args.train_samples
        one_logger.log_metrics(
            {
                "train_samples_start": args.consumed_train_samples,
                "train_iterations_start": iteration,
                "train_samples_target": train_samples_target,
                "train_iterations_target": args.train_iters,
            }
        )

    num_floating_point_operations_so_far = args.num_floating_point_operations_so_far

    # Setup some training config params
    config.grad_scale_func = optimizer.scale_loss if optimizer is not None else None
    config.timers = timers
    if isinstance(model[0], DDP) and args.overlap_grad_reduce:
        if config.no_sync_func is not None:
            raise AssertionError(
                "When overlap_grad_reduce is True, config.no_sync_func must be None; "
                "a custom no_sync_func is not supported when overlapping grad-reduce"
            )
        config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
        if len(model) == 1:
            config.no_sync_func = config.no_sync_func[0]
        if args.align_grad_reduce:
            config.grad_sync_func = [
                model_chunk.start_grad_sync
                for model_chunk in model
            ]
            if len(model) == 1:
                config.grad_sync_func = config.grad_sync_func[0]
    if args.overlap_param_gather and args.align_param_gather:
        config.param_sync_func = [
            lambda x, model_index=model_index: optimizer.finish_param_sync(
                model_index, x
            )
            for model_index in range(len(model))
        ] if optimizer is not None else []
        if len(model) == 1:
            config.param_sync_func = config.param_sync_func[0]
    config.finalize_model_grads_func = finalize_model_grads_wrapper(finalize_model_grads)

    timers("interval-time", log_level=0).start(barrier=True)
    print_datetime("before the start of training step")
    report_memory_flag = True
    exit_flag = False

    if args.manual_gc:
        # Disable the default garbage collector and perform the collection manually.
        # This is to align the timing of garbage collection across ranks.
        if args.manual_gc_interval < 0:
            raise AssertionError(
                "Manual garbage collection interval should be larger than or equal to 0."
            )
        gc.disable()
        gc.collect()

    num_microbatches = get_num_microbatches()
    eval_duration = 0.0
    eval_iterations = 0

    def track_e2e_metrics():
        # Nested function to track a bunch of E2E APP metrics
        if one_logger:
            # overall_elapsed
            train_duration = timers("interval-time").active_time()
            train_samples = args.consumed_train_samples - train_samples_start
            train_iterations = iteration - iteration_start
            train_iterations_time_msecs_avg = (
                (train_duration * 1000.0) / train_iterations
                if train_iterations > 0
                else None
            )
            if eval_iterations > 0:
                validation_iterations_time_msecs_avg = (
                    eval_duration * 1000.0
                ) / eval_iterations
            else:
                validation_iterations_time_msecs_avg = None

            one_logger.log_metrics(
                {
                    "train_iterations_end": iteration,
                    "train_samples_end": args.consumed_train_samples,
                    "train_iterations": train_iterations,
                    "train_samples": train_samples,
                    "train_iterations_time_msecs_avg": train_iterations_time_msecs_avg,
                    "validation_iterations_time_msecs_avg": validation_iterations_time_msecs_avg,
                }
            )

    if os.getenv('OOTB_OPTIMIZER_PROFILING', 'FALSE') != 'TRUE':
        prof = Profiler(args.mm.tool.profile)
        prof.start()

    curr_step_lr = None
    curr_step_dlr = None
    for param_group in optimizer.param_groups:
        if param_group["is_decoupled_lr"]:
            curr_step_dlr = param_group["lr"]
        else:
            curr_step_lr = param_group["lr"]

    while iteration < args.train_iters:
        memory_profiler.step()

        # dynamic dp/cp
        data_aware_parallel_optimize(train_data_iterator)

        # Update number of microbatches first without consistency check to decide if a
        # checkpoint should be saved. If the number of microbatches is different
        # from the previous iteration, save a checkpoint. Then run consistency check
        # to make sure training configuration is still valid.
        update_num_microbatches(args.consumed_train_samples, consistency_check=False)
        if get_num_microbatches() != num_microbatches and iteration != 0 and not is_use_dynamic_dpcp():
            if get_num_microbatches() <= num_microbatches:
                raise AssertionError(
                    "number of microbatches should be increasing due to batch size rampup"
                )
            save_checkpoint_and_time(
                iteration,
                model,
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
                None,
            )
        num_microbatches = get_num_microbatches()
        update_num_microbatches(args.consumed_train_samples, consistency_check=True)

        if args.use_data_balance:
            micro_batch_size = args.micro_batch_size
            encoder_num_microbatches = num_microbatches
            if args.hetero_parallel and args.hetero_encoder_mbs_scale > 1:
                micro_batch_size = args.micro_batch_size * args.hetero_encoder_mbs_scale
                encoder_num_microbatches = num_microbatches // args.hetero_encoder_mbs_scale
            is_vit_last_stage = False
            if model[0].module.module.add_image_encoder:
                is_vit_last_stage = model[0].module.module.image_encoder.post_process
            train_data_iterator = data_balance_algo.build_balanced_train_data_iterator(
                is_vit_last_stage=is_vit_last_stage,
                max_batch_capacity=micro_batch_size,
                micro_batch_size=micro_batch_size,
                num_microbatches=encoder_num_microbatches,
                data_type='image',
            )

        args.curr_iteration = iteration
        loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = train_step(
            forward_step_func,
            train_data_iterator,
            model,
            optimizer,
            opt_param_scheduler,
            config,
            call_backs
        )
        iteration += 1
        if args.use_txt_dynamic_batching:
            dp_process_group = mpu.get_data_parallel_group()
            num_replicas = dp_process_group.size()
            batch_size_per_rank = train_data_iterator.iterable.gi_frame.f_locals['dl'].consumed_train_samples
            batch_size_per_rank = torch.tensor(batch_size_per_rank).npu()
            batch_size_all_rank = [torch.empty_like(batch_size_per_rank) for _ in range(num_replicas)]
            torch.distributed.all_gather(batch_size_all_rank, batch_size_per_rank, group=dp_process_group)
            batch_size = sum(batch_size_all_rank) - args.consumed_train_samples
        else:
            batch_size = (
                mpu.get_data_parallel_world_size()
                * args.micro_batch_size
                * get_num_microbatches()
            )
        args.consumed_train_samples += batch_size
        num_floating_point_operations_so_far += num_floating_point_operations(
            args, batch_size
        )

        # Logging.
        loss_scale = optimizer.get_loss_scale().item()
        params_norm = None
        if args.log_params_norm:
            params_norm = calc_params_l2_norm(model)

        if iteration % args.log_interval == 0:
            track_e2e_metrics()

        report_memory_flag = training_log(
            loss_dict,
            total_loss_dict,
            curr_step_lr,
            curr_step_dlr,
            iteration,
            loss_scale,
            report_memory_flag,
            skipped_iter,
            grad_norm,
            params_norm,
            num_zeros_in_grad,
        )
        for param_group in optimizer.param_groups:
            if param_group["is_decoupled_lr"]:
                curr_step_dlr = param_group["lr"]
            else:
                curr_step_lr = param_group["lr"]

        # Autoresume
        if args.adlr_autoresume and (iteration % args.adlr_autoresume_interval == 0):
            check_adlr_autoresume_termination(
                iteration, model, optimizer, opt_param_scheduler
            )

        # Evaluation
        if args.eval_interval and iteration % args.eval_interval == 0 and args.do_valid:
            timers("interval-time").stop()
            if judge_forward_pre_hook(args, model, optimizer):
                disable_forward_pre_hook(model)
            if args.manual_gc and args.manual_gc_eval:
                # Collect all objects.
                gc.collect()
            prefix = "iteration {}".format(iteration)
            timers("eval-time", log_level=0).start(barrier=True)
            evaluate_and_print_results(
                prefix,
                forward_step_func,
                valid_data_iterator,
                model,
                iteration,
                process_non_loss_data_func,
                config,
                False,
            )
            eval_duration += timers("eval-time").elapsed()
            eval_iterations += args.eval_iters
            timers("eval-time").stop()
            if args.manual_gc and args.manual_gc_eval:
                # Collect only the objects created and used in evaluation.
                gc.collect(generation=0)
            if args.use_distributed_optimizer and args.overlap_param_gather and optimizer is not None:
                optimizer.enable_pre_hook()
            timers("interval-time", log_level=0).start(barrier=True)

        # Checkpointing
        saved_checkpoint = False
        if args.exit_signal_handler:
            signal_handler = get_signal_handler()
            if any(signal_handler.signals_received()):
                save_checkpoint_and_time(
                    iteration,
                    model,
                    optimizer,
                    opt_param_scheduler,
                    num_floating_point_operations_so_far,
                    None,
                )
                print_datetime("exiting program after receiving SIGTERM.")
                exit_flag = True
                break

        if args.save and args.save_interval and iteration % args.save_interval == 0:
            save_checkpoint_and_time(
                iteration,
                model,
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
                None,
            )
            saved_checkpoint = True

        # Exiting based on duration
        if args.exit_duration_in_mins:
            train_time = (time.time() - _TRAIN_START_TIME) / 60.0
            done_cuda = torch.tensor(
                [train_time > args.exit_duration_in_mins],
                dtype=torch.int,
                device="cuda",
            )
            torch.distributed.all_reduce(done_cuda, op=torch.distributed.ReduceOp.MAX)
            done = done_cuda.item()
            if done:
                if not saved_checkpoint:
                    save_checkpoint_and_time(
                        iteration,
                        model,
                        optimizer,
                        opt_param_scheduler,
                        num_floating_point_operations_so_far,
                        None,
                    )
                print_datetime("exiting program after {} minutes".format(train_time))
                exit_flag = True
                break

        # Exiting based on iterations
        if args.exit_interval and iteration % args.exit_interval == 0:
            if args.save and not saved_checkpoint:
                save_checkpoint_and_time(
                    iteration,
                    model,
                    optimizer,
                    opt_param_scheduler,
                    num_floating_point_operations_so_far,
                    None,
                )
            torch.distributed.barrier()
            print_datetime("exiting program at iteration {}".format(iteration))
            exit_flag = True
            break

        if args.manual_gc:
            if args.manual_gc_interval != 0 and iteration % args.manual_gc_interval == 0:
                gc.collect()

        if os.getenv('OOTB_OPTIMIZER_PROFILING', 'FALSE') != 'TRUE':
            prof.step()
    if os.getenv('OOTB_OPTIMIZER_PROFILING', 'FALSE') != 'TRUE':
        prof.stop()

    track_e2e_metrics()

    # Flush TensorBoard and WandB writers.
    writer = get_tensorboard_writer()
    if writer:
        writer.flush()
    wandb_writer = get_wandb_writer()
    if wandb_writer:
        wandb_writer.finish()

    # Close out pre-hooks if using distributed optimizer and overlapped param gather.
    if judge_forward_pre_hook(args, model, optimizer):
        disable_forward_pre_hook(model)

    # If any exit conditions (signal handler, duration, iterations) have been reached, exit.
    if exit_flag:
        sys.exit()

    if getattr(args, "auto_parallel_profile", False):
        from mindspeed.core.auto_parallel.mm_search.profiling import save_profile_data
        save_profile_data(args)

    return iteration, num_floating_point_operations_so_far


@train_step_decorator
def train_step(
        forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config, call_backs
):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    # Set grad to zero.
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    if optimizer is not None:
        optimizer.zero_grad()

    # Forward pass.
    from megatron.core.pipeline_parallel import get_forward_backward_func
    if args.hetero_parallel and args.pipeline_model_parallel_size > 1:
        import mindspeed_mm.patchs.hetero_pipeline_patches as hetero_pp
        get_forward_backward_func = hetero_pp.hp_get_forward_backward_func

    forward_backward_func = get_forward_backward_func()
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=get_num_microbatches(),
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        decoder_seq_length=args.decoder_seq_length,
        forward_only=False,
    )

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    # Vision gradients.
    if (
        getattr(args, "vision_pretraining", False)
        and args.vision_pretraining_type == "dino"
    ):
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

    # Update parameters.
    timers("optimizer", log_level=1).start(barrier=args.barrier_with_L1_time)
    if optimizer is not None:
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    else:
        torch.distributed.barrier()
        update_successful = True
        grad_norm = 0
        num_zeros_in_grad = 0
    if call_backs:
        if isinstance(call_backs, list):
            for call_back in call_backs:
                call_back(unwrap_model(model[0]))
    timers("optimizer").stop()

    # Vision momentum.
    if (
        getattr(args, "vision_pretraining", False)
        and args.vision_pretraining_type == "dino"
    ):
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.update_momentum(args.curr_iteration)

    # Update learning rate.
    if update_successful:
        increment = (
            get_num_microbatches() * args.micro_batch_size * args.data_parallel_size
        )
        if opt_param_scheduler is not None:
            opt_param_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1

    # Empty unused memory.
    if args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    loss_is_needed = mpu.is_pipeline_last_stage(ignore_virtual=True)
    # Loss will be output from the pipeline first stage if patch
    # `layerwise_disaggregated_training` is enabled.
    cfg = args.mm.model
    if hasattr(cfg, "patch"):
        cfg = cfg.patch.to_dict()
        if cfg.get("layerwise_disaggregated_training", False):
            loss_is_needed = mpu.is_pipeline_first_stage(ignore_virtual=True)

    if loss_is_needed:
        # Average loss across microbatches.
        loss_reduced = {}
        if not config.calculate_per_token_loss:
            for key in losses_reduced[0]:
                losses_reduced_for_key = [x[key] for x in losses_reduced]
                loss_reduced[key] = sum(losses_reduced_for_key) / len(
                    losses_reduced_for_key
                )
        else:
            for key in losses_reduced[0].keys():
                numerator = 0
                denominator = 0
                for x in losses_reduced:
                    val = x[key]
                    # there is one dict per microbatch. in new reporting, we average
                    # over the total number of tokens across the global batch.
                    if isinstance(val, tuple) or isinstance(val, list):
                        numerator += val[0]
                        denominator += val[1]
                    else:
                        # legacy behavior. we average over the number of microbatches,
                        # and so the denominator is 1.
                        numerator += val
                        denominator += 1
                loss_reduced[key] = numerator / denominator
        return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
    return {}, skipped_iter, grad_norm, num_zeros_in_grad


def training_log(loss_dict, total_loss_dict, learning_rate, decoupled_learning_rate, iteration,
                 loss_scale, report_memory_flag, skipped_iter,
                 grad_norm, params_norm, num_zeros_in_grad):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()
    wandb_writer = get_wandb_writer()
    one_logger = get_one_logger()

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = 'advanced iterations'
    skipped_iters_key = 'skipped iterations'
    nan_iters_key = 'nan iterations'
    # Advanced iterations.
    if not skipped_iter:
        total_loss_dict[advanced_iters_key] = total_loss_dict.get(
            advanced_iters_key, 0) + 1
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(
        skipped_iters_key, 0) + skipped_iter
    # Update losses and set nan iterations
    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = total_loss_dict.get(
                key, torch.tensor([0.0], dtype=torch.float, device='cuda')) + loss_dict[key]
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float('inf') or \
                     value == -float('inf') or \
                     value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(
        nan_iters_key, 0) + int(got_nan)

    # Logging.
    timers_to_log = [
        'forward-backward',
        'forward-compute',
        'backward-compute',
        'batch-generator',
        'forward-recv',
        'forward-send',
        'backward-recv',
        'backward-send',
        'forward-send-forward-recv',
        'forward-send-backward-recv',
        'backward-send-forward-recv',
        'backward-send-backward-recv',
        'forward-backward-send-forward-backward-recv',
        'layernorm-grads-all-reduce',
        'embedding-grads-all-reduce',
        'all-grads-sync',
        'params-all-gather',
        'optimizer-copy-to-main-grad',
        'optimizer-unscale-and-check-inf',
        'optimizer-clip-main-grad',
        'optimizer-count-zeros',
        'optimizer-inner-step',
        'optimizer-copy-main-to-model-params',
        'optimizer']

    # Calculate batch size.
    batch_size = get_current_global_batch_size()

    # Track app tag & app tag ID
    if one_logger:
        job_name = os.environ.get('SLURM_JOB_NAME', None)
        current_app_tag = f'{job_name}_{batch_size}_{args.world_size}'
        one_logger.log_app_tag(current_app_tag)

    total_iterations = total_loss_dict[advanced_iters_key] + \
                       total_loss_dict[skipped_iters_key]

    # Tensorboard values.
    # Timer requires all the ranks to call.
    if args.log_timers_to_tensorboard and \
       (iteration % args.tensorboard_log_interval == 0):
        timers.write(timers_to_log, writer, iteration,
                     normalizer=total_iterations)
    if writer and (iteration % args.tensorboard_log_interval == 0):
        if wandb_writer:
            wandb_writer.log({'samples vs steps': args.consumed_train_samples},
                             iteration)
        for key in loss_dict:
            writer.add_scalar(key, loss_dict[key], iteration)
            writer.add_scalar(key + ' vs samples', loss_dict[key],
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({key: loss_dict[key]}, iteration)
        if args.log_loss_scale_to_tensorboard:
            writer.add_scalar('loss-scale', loss_scale, iteration)
            writer.add_scalar('loss-scale vs samples', loss_scale,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'loss-scale': loss_scale}, iteration)
        if args.log_world_size_to_tensorboard:
            writer.add_scalar('world-size', args.world_size, iteration)
            writer.add_scalar('world-size vs samples', args.world_size,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'world-size': args.world_size}, iteration)
        if grad_norm is not None:
            writer.add_scalar('grad-norm', grad_norm, iteration)
            writer.add_scalar('grad-norm vs samples', grad_norm,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'grad-norm': grad_norm}, iteration)
        if num_zeros_in_grad is not None:
            writer.add_scalar('num-zeros', num_zeros_in_grad, iteration)
            writer.add_scalar('num-zeros vs samples', num_zeros_in_grad,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'num-zeros': num_zeros_in_grad}, iteration)
        if params_norm is not None:
            writer.add_scalar('params-norm', params_norm, iteration)
            writer.add_scalar('params-norm vs samples', params_norm,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'params-norm': params_norm}, iteration)
        if args.log_memory_to_tensorboard:
            mem_stats = torch.cuda.memory_stats()
            writer.add_scalar(
                "mem-reserved-bytes",
                mem_stats["reserved_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-bytes",
                mem_stats["allocated_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-count",
                mem_stats["allocation.all.current"],
                iteration,
            )
    if args.num_experts is not None:
        moe_loss_scale = 1 / get_num_microbatches()
        track_moe_metrics(moe_loss_scale, iteration, writer, wandb_writer, total_loss_dict, args.moe_per_layer_logging)

    if hasattr(args, "mtp_num_layers") and args.mtp_num_layers:
        from mindspeed_mm.models.common.transformer.multi_token_prediction import MTPLossLoggingHelper

        mtp_loss_scale = 1 / get_num_microbatches()
        MTPLossLoggingHelper.track_mtp_metrics(
            mtp_loss_scale, iteration, writer, wandb_writer, total_loss_dict
        )

    if iteration % args.log_interval == 0:
        elapsed_time = timers('interval-time').elapsed(barrier=True)
        elapsed_time_per_iteration = elapsed_time / total_iterations

        throughput = num_floating_point_operations(args, batch_size) / (
            elapsed_time_per_iteration * 10**12 * args.world_size)
        if args.log_timers_to_tensorboard:
            if writer:
                writer.add_scalar('iteration-time',
                                  elapsed_time_per_iteration, iteration)
            if wandb_writer:
                wandb_writer.log({'iteration-time': elapsed_time_per_iteration},
                                 iteration)
        log_string = f" [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
        log_string += ' iteration {:8d}/{:8d} |'.format(
            iteration, args.train_iters)
        log_string += ' consumed samples: {:12d} |'.format(
            args.consumed_train_samples)
        log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
            elapsed_time_per_iteration * 1000.0)
        if args.log_throughput:
            log_string += f' throughput per GPU (TFLOP/s/GPU): {throughput:.1f} |'
            if args.log_timers_to_tensorboard:
                if writer:
                    writer.add_scalar('throughput', throughput, iteration)
                if wandb_writer:
                    wandb_writer.log({'throughput': throughput}, iteration)
        if learning_rate is None:
            raise AssertionError
        # Decoupled_learning_rate should be not None only on first and last pipeline stage.
        log_string += ' learning rate: {:.6E} |'.format(learning_rate)
        if args.decoupled_lr is not None and (mpu.is_pipeline_first_stage(ignore_virtual=True) or
                                              mpu.is_pipeline_last_stage(ignore_virtual=True)):
            if decoupled_learning_rate is None:
                raise AssertionError
            log_string += ' decoupled learning rate: {:.6E} |'.format(decoupled_learning_rate)
        else:
            if decoupled_learning_rate is not None:
                raise AssertionError
        log_string += ' global batch size: {:5d} |'.format(batch_size)
        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key,
                           nan_iters_key]:
                avg = total_loss_dict[key].item() / \
                      float(max(1, total_loss_dict[advanced_iters_key]))
                if avg >= 0.0:
                    log_string += ' {}: {:.6E} |'.format(key, avg)
                total_loss_dict[key] = torch.tensor([0.0], dtype=torch.float, device='cuda')
        log_string += ' loss scale: {:.1f} |'.format(loss_scale)
        if grad_norm is not None:
            log_string += ' grad norm: {:.3f} |'.format(grad_norm)
        if num_zeros_in_grad is not None:
            log_string += ' num zeros: {:.1f} |'.format(num_zeros_in_grad)
        if params_norm is not None:
            log_string += ' params norm: {:.3f} |'.format(params_norm)
        log_string += ' number of skipped iterations: {:3d} |'.format(
            total_loss_dict[skipped_iters_key])
        log_string += ' number of nan iterations: {:3d} |'.format(
            total_loss_dict[nan_iters_key])
        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        print_rank_last(log_string)
        if report_memory_flag and learning_rate > 0.:
            # Report memory after optimizer state has been initialized.
            if torch.distributed.get_rank() == 0:
                num_microbatches = get_num_microbatches()
                report_theoretical_memory(args, num_microbatches=num_microbatches, verbose=True)
            report_memory('(after {} iterations)'.format(iteration))
            report_memory_flag = False
        timers.log(timers_to_log, normalizer=args.log_interval)

    return report_memory_flag


def judge_save_checkpoint(args, iteration):
    if not args.save or iteration == 0:
        return False
    if iteration % args.save_interval == 0:
        return False
    if os.getenv('OOTB_OPTIMIZER_PROFILING', 'FALSE') == 'TRUE':
        return False
    return True


# Close out pre-hooks if using distributed optimizer and overlapped param gather.
def judge_forward_pre_hook(args, model, optimizer):
    if not args.use_distributed_optimizer and not args.overlap_param_gather:
        return False
    if not isinstance(model, DDP):
        return False
    if optimizer:
        return False
    return True


def no_wd_decay_cond(name: str, param) -> bool:
    """
    Condition function to determine if a parameter should be excluded from weight decay.

    Args:
        name: str - Parameter name string (e.g., model layer parameter name)
        param - Model parameter object

    Returns:
        bool - True means the parameter should be excluded from weight decay; False means weight decay is applied normally
    """
    args = get_args()
    no_wd_module_keywords = args.weight_decay_exclude_modules

    if not no_wd_module_keywords:
        return False

    # Case-insensitive matching: check if parameter name contains any exclusion keyword
    return any(keyword.lower() in name.lower() for keyword in no_wd_module_keywords)


def scale_lr_cond(name: str, param) -> bool:
    """
    Condition function to determine if a parameter should apply learning rate scaling (with --lr-mult).

    Args:
        name: str - Parameter name string (e.g., model layer parameter name)
        param - Model parameter object (not directly used here but retained for filter function interface compatibility)

    Returns:
        bool - True means the parameter should apply learning rate scaling; False means use the default learning rate
    """
    args = get_args()
    scale_lr_module_keywords = args.lr_scale_modules

    if not scale_lr_module_keywords:
        return False

    # Case-insensitive matching: check if parameter name contains any scaling keyword
    return any(keyword.lower() in name.lower() for keyword in scale_lr_module_keywords)
