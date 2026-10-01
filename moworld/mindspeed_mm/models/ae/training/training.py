# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import os
import time
from datetime import datetime, timezone
import pytz

import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm import tqdm

from megatron.training.utils import print_rank_0
from megatron.training.training import print_datetime
from megatron.core.timers import Timers

from mindspeed_mm.configs.config import MMConfig
from mindspeed_mm.tools.profiler import Profiler
from mindspeed_mm.models.ae.training.arguments import parse_ae_args
from mindspeed_mm.models.ae.training.global_vars import (
    set_ae_global_variables,
    get_ae_args
)
from mindspeed_mm.utils.ema import EMA
from mindspeed_mm.utils.utils import (
    set_modules_requires_grad,
    save_ae_checkpoint
)


def pretrain_ae(
    train_valid_test_dataset_provider,
    model_provider,
    forward_step_func,
):
    """
    Main AE training program.

    This function will run the following in the order provided:
        1) initialize DDP.
        2) setup model, optimizer and AMP scaler.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the model using the forward_step_func.

    Args:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns vanilla version of the AE
            generator and discriminator models. By vanilla we mean a simple
            model on cpu with no fp16 or ddp.
        forward_step_func: a function that takes a `data batch`, `AE generator`
            and `discriminator` models, and returns the loss of the corresponding
            model.
    """
    # setup ddp
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    global_rank = dist.get_rank()

    # parse AEModel Train args
    args = parse_ae_args()

    # parse model, data and tool config file
    args.model = MMConfig({"model": args.model_config}).model
    args.data = MMConfig({"data": args.data_config}).data
    args.tool = MMConfig({"tool": args.tool_config}).tool

    # set global args
    set_ae_global_variables(args)

    torch.backends.cuda.matmul.allow_tf32 = getattr(args.model, "allow_tf32", False)
    torch.npu.config.allow_internal_format = getattr(args.model, "allow_internal_format", False)

    # Model
    rank = int(os.environ["LOCAL_RANK"])
    args = get_ae_args()
    ae_model, discrim_model = model_provider()
    ae_model, discrim_model = ae_model.to(rank), discrim_model.to(rank)
    ae_model = nn.parallel.DistributedDataParallel(
        ae_model, device_ids=[rank], find_unused_parameters=args.find_unused_parameters
    )
    discrim_model = nn.parallel.DistributedDataParallel(
        discrim_model, device_ids=[rank], find_unused_parameters=args.find_unused_parameters
    )

    # Optimizer
    modules_to_train = [module for module in ae_model.module.get_decoder()]
    if not args.freeze_encoder:
        modules_to_train += [module for module in ae_model.module.get_encoder()]
    else:
        for module in ae_model.module.get_encoder():
            module.eval()
            module.requires_grad_(False)
        print_rank_0("Encoder is freezed!")

    parameters_to_train = []
    for module in modules_to_train:
        parameters_to_train += list(filter(lambda p: p.requires_grad, module.parameters()))

    ae_optim = torch.optim.AdamW(parameters_to_train, lr=args.ae_lr, weight_decay=args.ae_wd)
    discrim_optim = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, discrim_model.module.discriminator.parameters()),
        lr=args.discriminator_lr,
        weight_decay=args.discriminator_wd
    )

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler()
    mix_precision = torch.bfloat16
    if args.mix_precision == "fp16":
        mix_precision = torch.float16
    elif args.mix_precision == "fp32":
        mix_precision = torch.float32
    args.mix_precision = mix_precision

    print_datetime("after model, optimizer, and scaler are built")

    # Data stuff.
    train_dataloader, valid_dataloader, test_data_loader = train_valid_test_dataset_provider()

    print_datetime("after dataloaders are built")

    # Load from checkpoint
    start_epoch = 0
    current_step = 0
    if args.load:
        if not os.path.isfile(args.load):
            raise Exception(
                f"Make sure `{args.load}` is a ckpt file."
            )
        checkpoint = torch.load(args.load, map_location="cpu")
        ae_model.module.load_state_dict(checkpoint["state_dict"]["ae_model"], strict=False)
        discrim_model.module.load_state_dict(checkpoint["state_dict"]["discriminator_model"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        ae_optim.load_state_dict(checkpoint["optimizer_state"]["ae_optimizer"])
        discrim_optim.load_state_dict(checkpoint["optimizer_state"]["discriminator_optimizer"])
        train_dataloader.sampler.load_state_dict(checkpoint["sampler_state"])
        start_epoch = checkpoint["sampler_state"]["epoch"]
        current_step = checkpoint["current_step"]
        print(
            f"Checkpoint loaded from {args.load}, starting from epoch {start_epoch} step {current_step}"
        )

    if args.ema:
        print_rank_0(f"Start with EMA. EMA decay = {args.ema_decay}.")
        ema = EMA(ae_model, args.ema_decay)
        ema.register()

    # Print setup timing.
    print_rank_0("done with setup ...")

    # Training Loop
    args.train_iters = (
        args.epochs * len(train_dataloader) if args.train_iters is None else args.train_iters
    )
    print_rank_0("Training Details: ")
    print_rank_0(f" Max steps: {args.train_iters}")
    print_rank_0(f" Dataset Samples: {len(train_dataloader)}")
    print_rank_0(
        f" Total Batch Size: {train_dataloader.batch_size} * {args.world_size}"
    )
    dist.barrier()

    print_rank_0("training ...")
    prof = Profiler(args.tool.profile)
    prof.start()

    args.current_step = current_step
    args.current_epoch = start_epoch
    timers = Timers(log_level=0, log_option="minmax")
    timers("discriminator-interval-time", log_level=0).start(barrier=True)
    timers("generator-interval-time", log_level=0).start(barrier=True)
    for epoch in range(args.epochs):
        if current_step >= args.train_iters:
            break
        for module in modules_to_train:
            module.train()
        train_dataloader.sampler.set_epoch(epoch)  # Shuffle data at every epoch
        for _, batch in enumerate(train_dataloader):
            if current_step >= args.train_iters:
                break

            if (
                current_step % 2 == 1
                and current_step >= discrim_model.module.discriminator_iter_start
            ):
                set_modules_requires_grad(modules_to_train, False)
                args.step_gen = False
                args.step_disc = True
                timers("discriminator-interval-time", log_level=0).elapsed(barrier=True)
            else:
                set_modules_requires_grad(modules_to_train, True)
                args.step_gen = True
                args.step_disc = False
                timers("generator-interval-time", log_level=0).elapsed(barrier=True)

            # Forward
            gen_loss, discrim_loss = forward_step_func(batch, ae_model, discrim_model)

            # Backward
            # Generator Step
            if args.step_gen:
                ae_optim.zero_grad()
                scaler.scale(gen_loss).backward()
                scaler.step(ae_optim)
                scaler.update()

                if args.ema:
                    ema.update()

                elapsed_time_per_iteration = timers("generator-interval-time").elapsed(barrier=True)
                training_log(current_step, gen_loss.item(), ae_optim.param_groups[0]["lr"], scaler.get_scale(), elapsed_time_per_iteration)
            # Discriminator Step
            if args.step_disc:
                discrim_optim.zero_grad()
                scaler.scale(discrim_loss).backward()
                scaler.unscale_(discrim_optim)
                nn.utils.clip_grad_norm_(discrim_model.module.discriminator.parameters(), 1.0)
                scaler.step(discrim_optim)
                scaler.update()

                elapsed_time_per_iteration = timers("discriminator-interval-time").elapsed(barrier=True)
                training_log(current_step, discrim_loss.item(), discrim_optim.param_groups[0]["lr"], scaler.get_scale(), elapsed_time_per_iteration)

            current_step += 1
            args.current_step = current_step

            # checkpoint
            if current_step % args.save_interval == 0 and global_rank == 0:
                file_path = save_ae_checkpoint(
                    epoch,
                    current_step,
                    {
                        "ae_optimizer": ae_optim.state_dict(),
                        "discriminator_optimizer": discrim_optim.state_dict(),
                    },
                    {
                        "ae_model": ae_model.module.state_dict(),
                        "discriminator_model": discrim_model.module.state_dict(),
                    },
                    scaler.state_dict(),
                    train_dataloader.sampler.state_dict(),
                    args.save,
                    f"checkpoint-{current_step}.ckpt",
                    ema_state_dict=ema.shadow if args.ema else {},
                )
                print_rank_0(f"Checkpoint has been saved to `{file_path}`.")

            prof.step()
    prof.stop()

    print_datetime("after training is done")


def training_log(
    iteration,
    loss,
    learning_rate,
    grad_scale,
    elapsed_time_per_iteration
):
    args = get_ae_args()
    loss_name = ""
    if args.step_gen:
        loss_name = "generator     loss"
    else:
        loss_name = "discriminator loss"

    log_string = f" [{datetime.now(timezone.utc).astimezone(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')}]"
    log_string += ' iteration {:8d}/{:8d} |'.format(iteration, args.train_iters)
    log_string += ' elapsed time per iteration (s): {:.6f} |'.format(
            elapsed_time_per_iteration)
    log_string += ' learning rate: {:.6E} |'.format(learning_rate)
    log_string += ' {}: {:.6E} |'.format(loss_name, loss)
    log_string += ' loss scale: {:.1f} |'.format(grad_scale)
    print_rank_0(log_string)
