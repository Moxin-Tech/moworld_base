# Copyright (c) [2025] [FastVideo Team]
# Copyright (c) [2025] [ByteDance Ltd. and/or its affiliates.]
# SPDX-License-Identifier: [Apache License 2.0]
#
# This file has been modified by [ByteDance Ltd. and/or its affiliates.] in 2025.
#
# Original file was released under [Apache License 2.0], with the full license text
# available at [https://github.com/hao-ai-lab/FastVideo/blob/main/LICENSE].
#
# This modified file is released under the same license.

import json
import os
import time
import argparse
from collections import deque
import math
from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.utils.data import DistributedSampler, DataLoader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
from diffusers import get_scheduler
from tqdm.auto import tqdm
from accelerate.utils import set_seed

from mindspeed_mm.configs.config import mm_extra_args_provider, merge_mm_args
from mindspeed_mm.tasks.rl.soragrpo.dataset.latent_flux_rl_datasets import latent_collate_function
from mindspeed_mm.tasks.rl.soragrpo.utils.communications_flux import sp_parallel_dataloader_wrapper
from mindspeed_mm.tasks.rl.soragrpo.utils.fsdp_util import get_dit_fsdp_kwargs, apply_fsdp_checkpointing
from mindspeed_mm.tasks.rl.soragrpo.utils.parallel_states import initialize_sequence_parallel_state, \
    get_sequence_parallel_state, destroy_sequence_parallel_group


class SoraGRPOTrainer(ABC):
    def __init__(self, train_valid_test_dataset_provider):
        self.local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("hccl")
        torch.cuda.set_device(self.local_rank)
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        if self.local_world_size != int(torch.cuda.device_count()):
            raise AssertionError(f"ASCEND_RT_VISIBLE_DEVICES which is {int(torch.cuda.device_count())} must specify the exact number of devices used per node which is {self.local_world_size}. "
                                 "Please verify its value and whether the current devices are available.")
        self.train_valid_test_dataset_provider = train_valid_test_dataset_provider
        self.optimizer = None
        self.lr_scheduler = None
        self.args = self.get_args()
        merge_mm_args(self.args)
        self.device = torch.cuda.current_device()
        self.hyper_model = None
        self.gc_iteration = 5
        initialize_sequence_parallel_state(self.args.sp_size)

    def train(self):
        import gc
        gc.disable()
        args = self.args
        rank = self.rank
        world_size = self.world_size
        local_rank = self.local_rank
        device = self.device
        local_world_size = self.local_world_size

        # enable communication deterministic can improve performance when world_size < 8
        if world_size <= 8:
            os.environ['HCCL_DETERMINISTIC'] = 'true'
        elif world_size >= 16:
            os.environ['HCCL_OP_EXPANSION_MODE'] = 'AIV'

        # We use different seeds for the noise generation in each process to ensure that the noise is different in a batch.
        if args.seed is not None:
            set_seed(args.seed + rank)

        # Handle the repository creation
        if rank <= 0 and args.save is not None:
            os.makedirs(args.save, exist_ok=True)

        transformer = None
        load_rank_batchsize = args.load_rank
        for start_rank in range(0, local_world_size, load_rank_batchsize):
            end_rank = min(start_rank + load_rank_batchsize, world_size)
            load_ranks = list(range(start_rank, end_rank))
            if local_rank in load_ranks:
                if local_rank % load_rank_batchsize == 0:
                    print(f"rank {load_ranks} load start")
                self.hyper_model = self.model_provider(args)
                transformer = self.hyper_model.diffuser
                fsdp_kwargs, split_modules = get_dit_fsdp_kwargs(
                    self.hyper_model,
                    args.fsdp_sharding_strategy,
                    False,
                    args.use_cpu_offload,
                    args.master_weight_type,
                )
                self.hyper_model.diffuser = FSDP(transformer, **fsdp_kwargs, )
                if args.gradient_checkpointing:
                    apply_fsdp_checkpointing(
                        transformer, split_modules, args.selective_checkpointing
                    )
                if local_rank % load_rank_batchsize == 0:
                    print(f"rank {load_ranks} load success")
            dist.barrier()

        self.main_print(
            f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_strategy}"
        )

        # Set model as trainable.
        transformer.train()

        params_to_optimize = transformer.parameters()
        params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
            eps=1e-8,
        )

        init_steps = 0
        self.main_print(f"optimizer: {optimizer}")

        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=1000000,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
            last_epoch=init_steps - 1,
        )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        train_dataset = self.train_valid_test_dataset_provider(args)
        sampler = DistributedSampler(
            train_dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.sampler_seed
        )

        train_dataloader = DataLoader(
            train_dataset,
            sampler=sampler,
            collate_fn=latent_collate_function,
            pin_memory=True,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
            drop_last=True,
        )

        # Train!
        total_batch_size = (
                args.train_batch_size
                * world_size
                * args.gradient_accumulation_steps
                / args.sp_size
                * args.train_sp_batch_size
        )
        self.main_print("***** Running training *****")
        self.main_print(f"  Num examples = {len(train_dataset)}")
        self.main_print(f"  Dataloader size = {len(train_dataloader)}")
        self.main_print(f"  Resume training from step {init_steps}")
        self.main_print(f"  Instantaneous batch size per device = {args.train_batch_size}")
        self.main_print(
            f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}"
        )
        self.main_print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        self.main_print(f"  Total optimization steps per epoch = {args.train_iters}")
        self.main_print(
            f"  Total training parameters per FSDP shard = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B"
        )
        self.main_print(f"  Master weight dtype: {transformer.parameters().__next__().dtype}")

        progress_bar = tqdm(
            range(0, 100000),
            initial=init_steps,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=local_rank > 0,
        )

        loader = sp_parallel_dataloader_wrapper(
            train_dataloader,
            device,
            args.train_batch_size,
            args.sp_size,
            args.train_sp_batch_size,
        )

        step_times = deque(maxlen=100)

        # The number of epochs 1 is a random value; you can also set the number of epochs to be two.
        for epoch in range(1):
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)  # Crucial for distributed shuffling per epoch

            for step in range(init_steps + 1, args.train_iters + 1):
                start_time = time.time()
                if args.save is not None and step % args.save_interval == 0:
                    self.save_checkpoint(transformer, rank, args.save, step, epoch)
                    dist.barrier()
                loss, grad_norm = self.train_one_step(loader)

                step_time = time.time() - start_time
                step_times.append(step_time)

                progress_bar.set_postfix(
                    {
                        "loss": f"{loss}",
                        "step_time": f"{step_time:.2f}s",
                        "grad_norm": grad_norm,
                    }
                )
                progress_bar.update(1)
                if step % self.gc_iteration == 0:
                    gc.collect()

        if get_sequence_parallel_state():
            destroy_sequence_parallel_group()

    def train_one_step(self, dataloader):
        device = self.device
        hyper_model = self.hyper_model
        args = self.args
        max_grad_norm = args.max_grad_norm
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler

        total_loss = 0.0
        optimizer.zero_grad()

        samples_batched_list, train_timesteps, sigma_schedule, perms = self.sample_reference(dataloader)

        for i, sample in list(enumerate(samples_batched_list)):
            for j in range(train_timesteps):
                clip_range = args.clip_range
                adv_clip_max = args.adv_clip_max
                new_log_probs = self.grpo_one_step(
                    sample,
                    perms[i][j],
                    sigma_schedule,
                    j
                )
                ratio = torch.exp(new_log_probs - sample["log_probs"][:, j])

                advantages = torch.clamp(
                    sample["advantages"],
                    -adv_clip_max,
                    adv_clip_max,
                )
                unclipped_loss = -advantages * ratio
                clipped_loss = -advantages * torch.clamp(
                    ratio,
                    1.0 - clip_range,
                    1.0 + clip_range,
                )
                loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss)) / (
                        args.gradient_accumulation_steps * train_timesteps)

                loss.backward()
                avg_loss = loss.detach().clone()
                dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
                total_loss += avg_loss.item()

            if dist.get_rank() % self.world_size == 0:
                print("hps reward", sample["rewards"].item())
                print("ratio", ratio)
                print("advantage", sample["advantages"].item())
                print("final loss", loss.item())

            if (i + 1) % args.gradient_accumulation_steps == 0:
                grad_norm = hyper_model.diffuser.clip_grad_norm_(max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            dist.barrier()
        return total_loss, grad_norm.item()

    @abstractmethod
    def sample_reference(self, dataloader):
        raise NotImplementedError("Subclasses must implement this method")

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--dataloader_num_workers",
            type=int,
            default=10,
            help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
        )
        parser.add_argument(
            "--train_batch_size",
            type=int,
            default=1,
            help="Batch size (per device) for the training dataloader.",
        )
        parser.add_argument(
            "--num_latent_t",
            type=int,
            default=1,
            help="number of latent frames",
        )
        # text encoder & vae & diffusion model
        parser.add_argument("--cache_dir", type=str, default="./cache_dir")
        # diffusion setting
        parser.add_argument("--ema_decay", type=float, default=0.995)
        parser.add_argument("--ema_start_step", type=int, default=0)
        parser.add_argument("--cfg", type=float, default=0.0)
        parser.add_argument(
            "--precondition_outputs",
            action="store_true",
            help="Whether to precondition the outputs of the model.",
        )
        # validation & logs
        parser.add_argument(
            "--seed", type=int, default=None, help="A seed for reproducible training."
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default=None,
            help="The output directory where the model predictions and checkpoints will be written.",
        )
        parser.add_argument(
            "--checkpointing_steps",
            type=int,
            default=500,
            help=(
                "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
                " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
                " training using `--resume_from_checkpoint`."
            ),
        )
        # optimizer & scheduler & Training
        parser.add_argument(
            "--train-iters",
            type=int,
            default=None,
            help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
        )
        parser.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass.",
        )
        parser.add_argument(
            "--lr",
            type=float,
            default=1e-4,
            help="Initial learning rate (after the potential warmup period) to use.",
        )
        parser.add_argument(
            "--lr_warmup_steps",
            type=int,
            default=10,
            help="Number of steps for the warmup in the lr scheduler.",
        )
        parser.add_argument(
            "--max_grad_norm", default=2.0, type=float, help="Max gradient norm."
        )
        parser.add_argument(
            "--gradient_checkpointing",
            action="store_true",
            help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
        )
        parser.add_argument("--selective_checkpointing", type=float, default=1.0)
        parser.add_argument(
            "--mixed_precision",
            type=str,
            default=None,
            choices=["no", "fp16", "bf16"],
            help=(
                "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
                " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
                " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
            ),
        )
        parser.add_argument(
            "--use_cpu_offload",
            action="store_true",
            help="Whether to use CPU offload for param & gradient & optimizer states.",
        )
        parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
        parser.add_argument(
            "--train_sp_batch_size",
            type=int,
            default=1,
            help="Batch size for sequence parallel training",
        )
        parser.add_argument("--fsdp_sharding_strategy", default="hybrid_full")
        parser.add_argument(
            "--lr_scheduler",
            type=str,
            default="constant_with_warmup",
            help=(
                'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
                ' "constant", "constant_with_warmup"]'
            ),
        )
        parser.add_argument(
            "--lr_num_cycles",
            type=int,
            default=1,
            help="Number of cycles in the learning rate scheduler.",
        )
        parser.add_argument(
            "--lr_power",
            type=float,
            default=1.0,
            help="Power factor of the polynomial scheduler.",
        )
        parser.add_argument(
            "--weight-decay", type=float, default=0.01, help="Weight decay to apply."
        )
        parser.add_argument(
            "--master_weight_type",
            type=str,
            default="fp32",
            help="Weight type to use - fp32 or bf16.",
        )
        # GRPO training
        parser.add_argument(
            "--h",
            type=int,
            default=None,
            help="video height",
        )
        parser.add_argument(
            "--w",
            type=int,
            default=None,
            help="video width",
        )
        parser.add_argument(
            "--t",
            type=int,
            default=None,
            help="video length",
        )
        parser.add_argument(
            "--sampling_steps",
            type=int,
            default=None,
            help="sampling steps",
        )
        parser.add_argument(
            "--eta",
            type=float,
            default=None,
            help="noise eta",
        )
        parser.add_argument(
            "--sampler_seed",
            type=int,
            default=None,
            help="seed of sampler",
        )
        parser.add_argument(
            "--loss_coef",
            type=float,
            default=1.0,
            help="the global loss should be divided by",
        )
        parser.add_argument(
            "--use_group",
            action="store_true",
            default=False,
            help="whether compute advantages for each prompt",
        )
        parser.add_argument(
            "--num_generations",
            type=int,
            default=16,
            help="num_generations per prompt",
        )
        parser.add_argument(
            "--use_hpsv2",
            action="store_true",
            default=False,
            help="whether use hpsv2 as reward model",
        )
        parser.add_argument(
            "--ignore_last",
            action="store_true",
            default=False,
            help="whether ignore last step of mdp",
        )
        parser.add_argument(
            "--init_same_noise",
            action="store_true",
            default=False,
            help="whether use the same noise within each prompt",
        )
        parser.add_argument(
            "--shift",
            type=float,
            default=1.0,
            help="shift for timestep scheduler",
        )
        parser.add_argument(
            "--timestep_fraction",
            type=float,
            default=1.0,
            help="timestep downsample ratio",
        )
        parser.add_argument(
            "--clip_range",
            type=float,
            default=1e-4,
            help="clip range for grpo",
        )
        parser.add_argument(
            "--adv_clip_max",
            type=float,
            default=5.0,
        )
        parser.add_argument(
            "--log-interval",
            type=int,
            help="print log train step interval",
        )
        parser.add_argument(
            "--save-interval",
            type=int,
            help="save checkpoint train step interval",
        )
        parser.add_argument(
            "--eval-interval",
            type=int,
            help="evaluation train step interval",
        )
        parser.add_argument(
            "--eval-iters",
            type=int,
            help="evaluation iterations",
        )
        parser.add_argument(
            "--save",
            type=str,
            default=None,
            help="save checkpoint path",
        )
        parser.add_argument(
            "--ckpt-format",
            type=str,
            default="torch",
            help="save checkpoint format",
        )
        parser.add_argument(
            "--distributed-backend",
            type=str,
            default="nccl",
            help="distributed backend",
        )
        parser.add_argument(
            "--load",
            type=str,
            required=True,
            help="pretrained checkpoint path",
        )
        parser.add_argument(
            "--hps_reward_save",
            type=str,
            help="hps reward save path",
        )
        parser.add_argument(
            "--sample_batch_size",
            type=int,
            help="sample reference batch size",
        )
        parser.add_argument(
            "--load_rank",
            type=int,
            default=8,
            help="load rank batch size",
        )
        parser = mm_extra_args_provider(parser)
        return parser.parse_args()

    @abstractmethod
    def model_provider(self, args):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def grpo_one_step(self, sample, perm, sigma_schedule, index):
        raise NotImplementedError("Subclasses must implement this method")

    def sd3_time_shift(self, shift, t):
        return (shift * t) / (1 + (shift - 1) * t)

    def gather_tensor(self, tensor):
        if not dist.is_initialized():
            return tensor
        world_size = dist.get_world_size()
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, tensor)
        return torch.cat(gathered_tensors, dim=0)

    @staticmethod
    def assert_eq(x, y, msg=None):
        if not x == y:
            raise AssertionError(f"{msg} not equal")

    def grpo_step(
            self,
            model_output: torch.Tensor,
            latents: torch.Tensor,
            sigmas: torch.Tensor,
            prev_sample: torch.Tensor,
            config: dict
    ):
        grpo = config["grpo"]
        sde_solver = config["sde_solver"]
        eta = config["eta"]
        index = config["index"]
        sigma = sigmas[index]
        dsigma = sigmas[index + 1] - sigma
        prev_sample_mean = latents + dsigma * model_output

        pred_original_sample = latents - sigma * model_output

        delta_t = sigma - sigmas[index + 1]
        std_dev_t = eta * math.sqrt(delta_t)

        if sde_solver:
            score_estimate = -(latents - pred_original_sample * (1 - sigma)) / sigma ** 2
            log_term = -0.5 * eta ** 2 * score_estimate
            prev_sample_mean = prev_sample_mean + log_term * dsigma

        if grpo and prev_sample is None:
            prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t

        if grpo:
            # log prob of prev_sample given prev_sample_mean and std_dev_t
            log_prob = (
                    -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2) / (
                    2 * (std_dev_t ** 2))
            )
            - math.log(std_dev_t) - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))

            # mean along all but batch dimension
            log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
            return prev_sample, pred_original_sample, log_prob
        else:
            return prev_sample_mean, pred_original_sample

    def save_checkpoint(self, transformer, rank, output_dir, step, epoch):
        self.main_print(f"--> saving checkpoint at step {step}")
        with FSDP.state_dict_type(
                transformer,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            cpu_state = transformer.state_dict()
        if rank <= 0:
            save_dir = os.path.join(output_dir, f"checkpoint-{step}-{epoch}")
            os.makedirs(save_dir, exist_ok=True)
            # save using safetensors
            weight_path = os.path.join(save_dir, "diffusion_pytorch_model.safetensors")
            save_file(cpu_state, weight_path)
            config_dict = dict(transformer.config)
            if "dtype" in config_dict:
                del config_dict["dtype"]
            config_path = os.path.join(save_dir, "config.json")
            # save dict as json
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=4)
        self.main_print(f"--> checkpoint saved at step {step}")

    def main_print(self, content):
        if int(os.environ["LOCAL_RANK"]) <= 0:
            print(content)
