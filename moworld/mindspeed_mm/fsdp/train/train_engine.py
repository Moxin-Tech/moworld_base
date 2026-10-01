# pylint: skip-file
import logging
from datetime import datetime

import torch

from mindspeed.fsdp.utils.log import print_rank

from mindspeed_mm.fsdp.utils.dtype import get_dtype
from mindspeed_mm.fsdp.distributed.fully_shard_parallel import pregather_fsdp_params
from mindspeed_mm.fsdp.distributed.parallel_state import get_parallel_state
from mindspeed_mm.fsdp.utils.utils import move_to_device, get_time, configure_hsdp_gradient_sync, tensor_to_dtensor, \
    report_memory
from mindspeed_mm.fsdp.data.data_utils.utils import build_iterations
from mindspeed_mm.fsdp.optimizer.clip_grad_norm import clip_grad_norm
from mindspeed_mm.fsdp.tools.profiler import Profiler
from mindspeed_mm.fsdp.tools.memory_profiler import memory_profiler
from mindspeed_mm.fsdp.loss.loss_func import build_loss_func
from mindspeed_mm.fsdp.params.argument import Arguments
from mindspeed_mm.fsdp.utils.lora_utils import load_state_dict
from mindspeed_mm.fsdp.data.dataloader.dataloader import Preloader
from mindspeed_mm.fsdp.checkpoint.hf_load_utils import load_hf_checkpoint, looks_like_hf_weight_dir
from mindspeed_mm.fsdp.utils.constants import MEMORY_REPORT_ITERATION
from mindspeed_mm.fsdp.train.training_context import TrainingStage, TrainingContext

logger = logging.getLogger(__name__)


class TrainEngine:
    """Training engine that manages the main training loop and operations."""

    def __init__(
        self,
        args: Arguments,
        train_dataloader,
        model,
        optimizer,
        scheduler,
        checkpointer,
        lora_weight_manager=None,
        **kwargs,
    ):
        self.args = args

        self.model = model
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = scheduler
        self.checkpointer = checkpointer
        self.lora_weight_manager = lora_weight_manager

        # Training state tracking
        self.iteration, self.consumed_train_samples = 0, 0

        # Load checkpoint if specified
        if args.training.load:
            self.iteration, self.consumed_train_samples = self.load()

        if (
            args.training.init_model_with_meta_device
            and args.training.lora.enable
            and args.training.lora.pretrained_lora_path
        ):
            lora_state_dict = load_state_dict(args.training.lora.pretrained_lora_path)
            model_state_dict = model.state_dict()
            for key, value in lora_state_dict.items():
                if key in model_state_dict:
                    target_tensor = model_state_dict[key]
                    device_mesh = getattr(target_tensor, "device_mesh", None)
                    placements = getattr(target_tensor, "placements", None)
                    if device_mesh is not None and placements is not None:
                        target_tensor.copy_(tensor_to_dtensor(value, device_mesh, placements))
                    else:
                        target_tensor.copy_(value)
            print_rank(
                logger.info,
                f"Reloaded {len(lora_state_dict)} LoRA parameters from {args.training.lora.pretrained_lora_path}",
            )

        self.profiler = Profiler(args.tools.profile)
        self.profiler.start()

    def average_losses_across_data_parallel_group(self, losses):
        """Reduce a tensor of losses across all GPUs."""
        ps = get_parallel_state()
        averaged_losses = torch.cat([loss.clone().detach().view(1) for loss in losses])
        torch.distributed.all_reduce(averaged_losses, group=ps.get_dp_group())
        averaged_losses = averaged_losses / torch.distributed.get_world_size(group=ps.get_dp_group())

        return averaged_losses

    def get_batch(self, data_iterator):
        """Generate a batch."""
        if data_iterator is not None:
            batch = next(data_iterator)
        else:
            raise ValueError("Data iterator is None. Unable to retrieve batch.")

        # Move input to device and cast precision
        if not self.args.data or not self.args.data.dataloader_param.enable_preload:
            param_dtype = self.args.parallel.fsdp_plan.param_dtype
            batch = move_to_device(batch, get_dtype(param_dtype) if param_dtype else None)
        return batch

    def set_loss_func(self, batch_data):
        args = self.args
        if args.features.loss_cfg.loss_type == "raw":
            return
        chunk_size = args.features.chunkloss_plan.chunk_size if args.features.enable_chunk_loss else None
        if args.features.enable_dynamic_chunk_loss:
            batch_data['total_chunk_size'] = args.features.chunkloss_plan.total_chunk_size
        loss_func = build_loss_func(args.features.loss_cfg.loss_type, chunk_size=chunk_size, **batch_data)

        if hasattr(self.model, "loss_function"):
            self.model.loss_function = loss_func
        else:
            setattr(self.model, "loss_function", loss_func)

        output_router_logits = args.features.loss_cfg.router_aux_loss_coef > 0.0
        if output_router_logits:
            batch_data.update(output_router_logits=True)

    def train_step(self, train_dataloader_iter):
        """Perform a single training step with gradient accumulation."""
        args = self.args
        total_loss = 0
        total_aux_loss = None
        all_mtp_loss = None
        # Gradient accumulation
        for step in range(args.training.gradient_accumulation_steps):
            # Wait for the preloaded batch to be ready
            batch_data = self.get_batch(train_dataloader_iter)

            # setup loss ctx
            self.set_loss_func(batch_data)

            # Determine if this is the last step of gradient accumulation
            is_last_step = step == args.training.gradient_accumulation_steps - 1
            configure_hsdp_gradient_sync(self.model, is_last_step)

            # forward step
            TrainingContext().set_training_stage(TrainingStage.FORWARD)
            output = self.model(**batch_data, use_cache=False)
            loss = output.loss / args.training.gradient_accumulation_steps
            total_loss += loss

            # mtp loss
            mtp_loss = getattr(output, 'mtp_loss', None)
            if mtp_loss is not None:
                final_mtp_loss = torch.mean(torch.stack(mtp_loss)) / args.training.gradient_accumulation_steps
                loss += final_mtp_loss * getattr(args.model, 'mtp_loss_scaling_factor', 0.1)
                if all_mtp_loss is None:
                    all_mtp_loss = [torch.zeros_like(loss) for loss in mtp_loss]
                for i in range(len(mtp_loss)):
                    all_mtp_loss[i] += mtp_loss[i] / args.training.gradient_accumulation_steps

            # Backward
            TrainingContext().set_training_stage(TrainingStage.BACKWARD)
            loss.backward()

            if getattr(output, 'aux_loss', None) is not None:
                aux_loss = output.aux_loss / args.training.gradient_accumulation_steps
                total_aux_loss = aux_loss if total_aux_loss is None else total_aux_loss + aux_loss

        # log dict for loss
        loss_dict = {}

        # Average loss across data parallel group
        total_loss = self.average_losses_across_data_parallel_group([total_loss])
        loss_dict["loss"] = total_loss.item()

        if all_mtp_loss:
            for i in range(len(all_mtp_loss)):
                all_mtp_loss[i] = self.average_losses_across_data_parallel_group([all_mtp_loss[i]])
                loss_dict[f"mtp_{i+1} loss"] = all_mtp_loss[i].item()

        if total_aux_loss:
            loss_dict["aux loss"] = total_aux_loss.item()

        return loss_dict

    def train(self):
        """Main training loop."""
        args = self.args

        # Get data iterator
        train_dataloader_iter, _, _ = build_iterations(self.train_dataloader)
        param_dtype = get_dtype(args.parallel.fsdp_plan.param_dtype) if args.parallel.fsdp_plan.param_dtype else None

        # Preload data
        if args.data and args.data.dataloader_param.enable_preload:
            train_dataloader_iter = Preloader(train_dataloader_iter, param_dtype=param_dtype)

        self.model.train()

        # --- Train Loop ---
        curr_step_lr = self.lr_scheduler.get_last_lr()[0]
        while self.iteration < args.training.train_iters:
            # Record memory usage if enabled
            memory_profiler.step()
            start_time = get_time(barrier=True)

            if self.args.parallel.fsdp_plan.pregather:
                pregather_fsdp_params(self.model)

            loss_dict = self.train_step(train_dataloader_iter)

            # Clip gradients when clip_grad>0 and get total grad_norm
            grad_norm = clip_grad_norm(
                self.model, max_norm=args.training.clip_grad, foreach=args.training.clip_grad_foreach
            )

            # Update parameters
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            # Update training state
            self.consumed_train_samples += args.training.global_batch_size
            self.iteration += 1

            # Calculate iteration time
            elapsed_time_per_iteration = get_time(barrier=True) - start_time

            # Stop profiling if enabled
            self.profiler.step()

            # Logging
            if self.iteration % args.training.log_interval == 0:
                self.training_log(
                    self.iteration,
                    elapsed_time_per_iteration,
                    curr_step_lr,
                    self.consumed_train_samples,
                    loss_dict,
                    grad_norm,
                )

            # Report memory after optimizer state has been initialized.
            if self.iteration == MEMORY_REPORT_ITERATION:
                report_memory('(after {} iterations)'.format(self.iteration))

            curr_step_lr = self.lr_scheduler.get_last_lr()[0]

            # Save checkpoint at specified intervals
            if (
                args.training.save
                and args.training.save_interval > 0
                and self.iteration % args.training.save_interval == 0
            ):
                self.save(self.iteration, self.consumed_train_samples)

        # Stop profiling if enabled
        self.profiler.stop()
        memory_profiler.stop()
        # Final save after training completes
        if args.training.save:
            self.save(self.iteration, self.consumed_train_samples)

    def training_log(
        self, iteration, elapsed_time_per_iteration, curr_step_lr, consumed_train_samples, loss_dict, grad_norm
    ):
        args = self.args
        log_string = f" [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
        log_string += ' iteration {:8d}/{:8d} |'.format(iteration, args.training.train_iters)
        log_string += ' consumed samples: {:12d} |'.format(consumed_train_samples)
        log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(elapsed_time_per_iteration * 1000.0)
        log_string += ' learning rate: {:.6E} |'.format(curr_step_lr)
        log_string += ' global batch size: {:5d} |'.format(args.training.global_batch_size)
        for name, value in loss_dict.items():
            log_string += f" {name}: {value:.6E} |"

        if grad_norm is not None:
            log_string += ' grad norm: {:.3f} |'.format(grad_norm)

        print_rank(logger.info, log_string)

    def load(self):
        """Load checkpoint and restore training state."""
        args = self.args
        iteration, consumed_train_samples = 0, 0

        state = {"model": self.model, "extra_state": {}}  # cannot be None
        if not args.training.no_load_optim:
            state["optimizer"] = self.optimizer

        fmt = args.training.load_format
        if fmt == "auto":
            fmt = "hf" if looks_like_hf_weight_dir(args.training.load) else "dcp"
        if fmt == "hf":
            release = load_hf_checkpoint(
                self.model, args.training.load,
                load_rank0_and_broadcast=args.training.load_rank0_and_broadcast,
                enable_lora=args.training.lora.enable,
                load_strict=args.training.load_strict,
            )
        else:
            release = self.checkpointer.load(
                path=args.training.load,
                state=state,
                load_rank0_and_broadcast=args.training.load_rank0_and_broadcast,
                load_strict=args.training.load_strict,
                enable_lora=args.training.lora.enable,
            )

        if not release:
            iteration = state["extra_state"]["iteration"]
            consumed_train_samples = state["extra_state"]["consumed_train_samples"]

            self.lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
            if self.train_dataloader is not None:
                self.train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
            if not args.training.no_load_rng:
                if "torch_rng_state" not in state["extra_state"]:
                    print_rank(logger.warning, "No RNG state found in checkpoint, skipping RNG loading")
                else:
                    torch.set_rng_state(state["extra_state"]["torch_rng_state"])

        # Synchronize all processes after loading
        torch.distributed.barrier()

        return iteration, consumed_train_samples

    def save(self, iteration, consumed_train_samples):
        """Save checkpoint with model, optimizer, and training state."""
        args = self.args

        # Handle LoRA save modes
        if args.training.lora.enable:
            # Save only LoRA adapter weights
            if self.lora_weight_manager is not None:
                self.lora_weight_manager.save_lora_only(
                    save_path=args.training.save,
                    iteration=iteration,
                )

        # Default save behavior (full model)
        state = {
            "model": self.model,
            "extra_state": {
                "iteration": iteration,
                "consumed_train_samples": consumed_train_samples,
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "train_dataloader": self.train_dataloader.state_dict(),
            },
        }
        if not args.training.no_save_optim:
            state["optimizer"] = self.optimizer
        if not args.training.no_save_rng:
            state["extra_state"]["torch_rng_state"] = torch.get_rng_state()
        self.checkpointer.save(args.training.save, state=state, iteration=iteration)

        # Synchronize all processes after saving
        torch.distributed.barrier()
