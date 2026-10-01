# mindspeed_mm/fsdp/train/funasr_train_engine.py
import logging

# Import FunASR specific components
from contextlib import nullcontext

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from mindspeed.fsdp.utils.log import print_rank

from mindspeed_mm.fsdp.distributed.fully_shard_parallel import pregather_fsdp_params
from mindspeed_mm.fsdp.optimizer.clip_grad_norm import clip_grad_norm
from mindspeed_mm.fsdp.tools.memory_profiler import memory_profiler
from mindspeed_mm.fsdp.utils.dtype import get_dtype
from mindspeed_mm.fsdp.utils.utils import get_time, move_to_device
from mindspeed_mm.fsdp.train.train_engine import TrainEngine

logger = logging.getLogger(__name__)


class FunasrTrainEngine(TrainEngine):
    """FunASR-specific training engine with custom step logic and epoch/split loop."""

    def train_step(self, train_dataloader_iter):
        """
        FunASR-specific train step:
        - Handles (loss, stats, weight) return signature
        - Applies FSDP no_sync for gradient accumulation
        - Accumulates stats with weight weighting
        """
        total_loss = 0.0
        accumulated_stats = {}
        total_weight = 0.0
        accum_steps = self.args.training.gradient_accumulation_steps

        for accum_step in range(accum_steps):
            # Reuse parent's get_batch
            batch = self.get_batch(train_dataloader_iter)
            batch = move_to_device(
                batch,
                get_dtype(self.args.parallel.fsdp_plan.param_dtype)
                if self.args.parallel.fsdp_plan.param_dtype else None
            )

            # FSDP no_sync for all but last accumulation step
            sync_context = nullcontext
            if hasattr(self.model, 'no_sync') and accum_step < accum_steps - 1:
                sync_context = self.model.no_sync

            with sync_context():
                # FunASR model returns (loss, stats, weight)
                loss, stats, weight = self.model(**batch)
                scaled_loss = loss / accum_steps
                scaled_loss.backward()

            # Accumulate loss and stats
            total_loss += scaled_loss
            if accum_step == 0:
                accumulated_stats = {k: v * weight for k, v in stats.items()}
            else:
                for k, v in stats.items():
                    accumulated_stats[k] = accumulated_stats.get(k, 0) + v * weight
            total_weight += weight

        # Reuse parent's loss averaging
        total_loss = self.average_losses_across_data_parallel_group([total_loss])
        return total_loss

    def train(self):
        """
        FunASR-specific training loop with epoch/split dataloader logic.
        Reuses parent's utility methods: training_log, save, load, profiler, etc.
        """
        # Validate dataloader interface
        if not hasattr(self.train_dataloader, 'build_iter'):
            raise RuntimeError("FunASR dataloader factory must have 'build_iter(epoch, data_split_i, start_step)' method")

        self.model.train()

        # Restore FunASR-specific state if present (set via FunasrTrainer)
        current_epoch = getattr(self, '_current_epoch', 0)
        start_data_split_i = getattr(self, 'start_data_split_i', 0)
        start_step = getattr(self, 'start_step', 0)

        for epoch in range(current_epoch, self.args.training.max_epochs):
            for data_split_i in range(start_data_split_i, self.train_dataloader.data_split_num):
                # Build dataloader for this epoch/split
                dataloader_tr, dataloader_val = self.train_dataloader.build_iter(
                    epoch=epoch,
                    data_split_i=data_split_i,
                    start_step=start_step
                )

                if hasattr(dataloader_tr, 'batch_sampler'):
                    dataloader_tr.batch_sampler.set_epoch(epoch)

                dataloader_iter = iter(dataloader_tr)

                # Train on this split using iteration-based loop
                while self.iteration < self.args.training.train_iters:
                    memory_profiler.step()
                    start_time = get_time(barrier=True)

                    # FSDP pregather if enabled
                    if self.args.parallel.fsdp_plan.pregather and not isinstance(self.model, DDP):
                        pregather_fsdp_params(self.model)

                    try:
                        loss = self.train_step(dataloader_iter)
                    except StopIteration:
                        break  # End of this data split

                    # Gradient clipping
                    grad_norm = None
                    if self.args.training.clip_grad > 0:
                        grad_norm = clip_grad_norm(
                            self.model,
                            max_norm=self.args.training.clip_grad,
                            norm_type=self.args.training.clip_norm_type,
                            foreach=self.args.training.clip_grad_foreach
                        )
                        if not torch.isfinite(grad_norm):
                            logger.warning(f"Non-finite grad_norm ({grad_norm}). Skipping update.")
                            self.optimizer.zero_grad()
                            continue

                    # Optimizer step
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    self.profiler.step()
                    self.consumed_train_samples += self.args.training.global_batch_size
                    self.iteration += 1

                    elapsed_time = get_time(barrier=True) - start_time
                    curr_lr = self.lr_scheduler.get_last_lr()[0]

                    # Logging (reuse parent method)
                    if self.iteration % self.args.training.log_interval == 0:
                        self.training_log(
                            self.iteration, elapsed_time, curr_lr,
                            self.consumed_train_samples, loss, loss.new_tensor(0.0), grad_norm
                        )

                    # Checkpointing (reuse parent method)
                    if (self.args.training.save and
                        self.args.training.save_interval > 0 and
                        self.iteration % self.args.training.save_interval == 0):
                        self.save(self.iteration, self.consumed_train_samples)

                    if self.iteration >= self.args.training.train_iters:
                        break

                # Reset split-level state for next epoch
                start_step = 0
            # Reset epoch-level state
            start_data_split_i = 0

            if self.iteration >= self.args.training.train_iters:
                break

        # Final cleanup
        self.profiler.stop()
        memory_profiler.stop()
        if self.args.training.save:
            self.save(self.iteration, self.consumed_train_samples)

    def save(self, iteration, consumed_train_samples):
        args = self.args
        extra_state = {
            "iteration": iteration,
            "consumed_train_samples": consumed_train_samples,
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }
        if hasattr(self, '_funasr_dataloader') and hasattr(self._funasr_dataloader, 'state_dict'):
            extra_state["funasr_dataloader"] = self._funasr_dataloader.state_dict()
        if not args.training.no_save_rng:
            extra_state["torch_rng_state"] = torch.get_rng_state()

        state = {
            "model": self.model,
            "extra_state": extra_state,
        }
        if not args.training.no_save_optim:
            state["optimizer"] = self.optimizer

        self.checkpointer.save(args.training.save, state=state, iteration=iteration)
        torch.distributed.barrier()

    def load(self):
        args = self.args

        attempt_optim_load = not args.training.no_load_optim

        state = {"model": self.model, "extra_state": {}}
        if attempt_optim_load:
            state["optimizer"] = self.optimizer

        state_model_only = {"model": self.model, "extra_state": {}}
        release = self.checkpointer.load(
            path=args.training.load, state=state_model_only
        )
        state["extra_state"] = state_model_only["extra_state"]


        if not release:
            iteration = state["extra_state"]["iteration"]
            consumed_train_samples = state["extra_state"]["consumed_train_samples"]
            self.lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
            if not args.training.no_load_rng:
                if "torch_rng_state" not in state["extra_state"]:
                    print_rank(logger.warning, f"No RNG state found in checkpoint, skipping RNG loading")
                else:
                    torch.set_rng_state(state["extra_state"]["torch_rng_state"])
        else:
            iteration, consumed_train_samples = 0, 0

        torch.distributed.barrier()
        return iteration, consumed_train_samples
