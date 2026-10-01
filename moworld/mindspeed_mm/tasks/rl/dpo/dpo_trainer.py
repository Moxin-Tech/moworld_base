# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import torch_npu

from megatron.core import mpu
from megatron.core.utils import get_model_config
from megatron.training.checkpointing import save_checkpoint
from megatron.training.global_vars import (
    get_args,
    get_timers,
)
from megatron.training.initialize import initialize_megatron
from megatron.training.initialize import set_jit_fusion_options
from megatron.training.training import (
    evaluate_and_print_results,
    print_datetime,
    get_one_logger,
    append_to_progress_log,
    build_train_valid_test_data_iterators,
    setup_model_and_optimizer,
)
from megatron.training.utils import average_losses_across_data_parallel_group, unwrap_model
from megatron.training.utils import print_rank_0
from mindspeed_mm.arguments import extra_args_provider_decorator
from mindspeed_mm.configs.config import merge_mm_args
from mindspeed_mm.patchs.patch_manager import PatchesManager
from mindspeed_mm.tasks.rl.utils import compute_log_probs
from mindspeed_mm.training import train
from mindspeed_mm.utils.random import seed_all

_TRAIN_START_TIME = time.time()


class DPOTrainer(ABC):
    """
    A trainer class for Direct Preference Optimization (DPO).
    This class provides methods for model initialize, computing losses and metrics, and training.
    """

    def __init__(
            self,
            train_valid_test_dataset_provider,
            model_type,
            process_non_loss_data_func=None,
            extra_args_provider=None,
            args_defaults=None,
            call_backs=None
    ):
        """
        Initializes the DPOTrainer instance.
        """
        self.train_valid_test_dataset_provider = train_valid_test_dataset_provider
        self.model_type = model_type
        self.process_non_loss_data_func = process_non_loss_data_func
        self.extra_args_provider = extra_args_provider
        self.args_defaults = args_defaults
        self.train_model = None
        self.refer_model = None
        self.optimizer = None
        self.opt_param_scheduler = None
        self.args = None
        self.call_backs = call_backs
        self.initialize()

    def initialize(self):
        """Set up the necessary configuration, logging, initializing the model, optimizer, extc."""
        extra_args_provider = self.extra_args_provider
        args_defaults = self.args_defaults
        model_provider = self.model_provider
        model_type = self.model_type

        extra_args_provider = extra_args_provider_decorator(extra_args_provider)
        # Initialize and get arguments, timers, and Tensorboard writer.
        initialize_megatron(
            extra_args_provider=extra_args_provider, args_defaults=args_defaults
        )

        args = get_args()
        merge_mm_args(args)
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
            "Initialization time for Megatron (seconds): {:.3f}".format(
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

        # Model, optimizer, and learning rate.
        timers("model-and-optimizer-setup", log_level=0).start(barrier=True)
        model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
            model_provider, model_type
        )

        if self.call_backs:
            if isinstance(self.call_backs, list):
                for call_back in self.call_backs:
                    call_back.setup(unwrap_model(model)[0])

        self.args = get_args()
        self.train_model = model
        self.optimizer = optimizer
        self.opt_param_scheduler = opt_param_scheduler

        timers("model-and-optimizer-setup").stop()
        print_datetime("after model, optimizer, and learning rate scheduler are built")

    def train(self):
        model = self.train_model
        forward_step_func = self.forward_step
        config = get_model_config(model[0])
        timers = get_timers()
        args = get_args()
        optimizer = self.optimizer
        opt_param_scheduler = self.opt_param_scheduler
        process_non_loss_data_func = self.process_non_loss_data_func
        train_valid_test_dataset_provider = self.train_valid_test_dataset_provider

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
                    self.call_backs,
                )

            print_datetime("after training is done")

            if args.save and iteration != 0 and iteration % args.save_interval != 0:
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

    @abstractmethod
    def disable_dropout(self):
        """
        disable dropout
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def model_provider(self, pre_process, post_process):
        """
        Builds the model

        Args:
            pre_process (bool, optional): Include the embedding leayer in the gpt decoder (used with pipeline parallelism).
            post_process (bool, optional): Include an output layer and a layernorm in the gpt decoder (used with pipeline parallelism).

        Returns:
            Vision-Language multi-modal model
        """
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    @abstractmethod
    def get_batch(data_iterator):
        """
        Generate a batch.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def loss_func(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor):
        """DPO Loss function.

        Args:
            input_tensor (torch.Tensor): The tensor with the labels (repeated in double)
            output_tensor (torch.Tensor): The tensor with the Policy and Reference Model's Logits
        """
        args = get_args()
        labels = input_tensor[:args.actual_micro_batch_size // 2, ...]

        all_policy_logits, all_reference_logits = torch.chunk(output_tensor, 2, dim=0)
        all_reference_logits = all_reference_logits.detach()

        loss, metrics = self.get_batch_loss_metrics(
            all_policy_logits,
            all_reference_logits,
            labels
        )

        # Check individual rank losses are not NaN prior to DP all-reduce.
        if args.check_for_nan_in_loss_and_grad:
            global_rank = torch.distributed.get_rank()
            if loss.isnan():
                raise ValueError(f'Rank {global_rank}: found NaN in local forward loss calculation. '
                                 f'Device: {torch.cuda.current_device()}')

        # Reduce loss for logging.
        metrics['lm loss'] = average_losses_across_data_parallel_group([loss])
        for key in metrics.keys():
            metrics[key] = average_losses_across_data_parallel_group([metrics[key]])

        return loss, metrics

    @abstractmethod
    def forward_step(self, data_iterator, model):
        """
        DPO Forward training step.

        Perform a forward pass and compute the loss.
        This method is called during each training iteration.

        Args:
            data_iterator : Input data iterator
            model : The VLM Model
        """
        raise NotImplementedError("Subclasses must implement this method")

    def dpo_loss(
            self,
            policy_chosen_loss: torch.Tensor,
            policy_rejected_loss: torch.Tensor,
            reference_chosen_loss: torch.Tensor,
            reference_rejected_loss: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_loss:
            Log probabilities or mean squared error of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_loss:
            Log probabilities or mean squared error of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_loss:
            Log probabilities or mean squared error of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_loss:
            Log probabilities or mean squared error of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the
            chosen and rejected responses, respectively.
        """
        policy_ratios = policy_chosen_loss - policy_rejected_loss
        ref_ratios = reference_chosen_loss - reference_rejected_loss
        loss_diff = policy_ratios - ref_ratios

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0.
        # The label_smoothing parameter encodes our uncertainty about the labels and calculates a conservative DPO loss.
        if self.args.dpo_loss_type == "sigmoid":
            losses = (
                    -F.logsigmoid(self.args.dpo_beta * loss_diff) * (1 - self.args.dpo_label_smoothing)
                    - F.logsigmoid(-self.args.dpo_beta * loss_diff) * self.args.dpo_label_smoothing
            )
        else:
            raise ValueError(
                f"Unknown loss type: {self.args.dpo_loss_type}."
                f" Should be one of ['sigmoid']"
            )

        chosen_rewards = (
                self.args.dpo_beta
                * (
                        policy_chosen_loss - reference_chosen_loss
                ).detach()
        )
        rejected_rewards = (
                self.args.dpo_beta
                * (
                        policy_rejected_loss - reference_rejected_loss
                ).detach()
        )

        return losses, chosen_rewards, rejected_rewards

    def compute_preference_loss(
            self,
            policy_chosen_log_probs: torch.Tensor,
            policy_rejected_log_probs: torch.Tensor,
            reference_chosen_log_probs: torch.Tensor,
            reference_rejected_log_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Computes the preference loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_log_probs: Log probabilities of the policy model for the chosen responses.
            policy_rejected_log_probs: Log probabilities of the policy model for the rejected responses.
            reference_chosen_log_probs: Log probabilities of the reference model for the chosen responses.
            reference_rejected_log_probs: Log probabilities of the reference model for the rejected responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the preference loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the
            chosen and rejected responses, respectively.
        """
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_log_probs,
            policy_rejected_log_probs,
            reference_chosen_log_probs,
            reference_rejected_log_probs
        )
        return losses, chosen_rewards, rejected_rewards

    def get_batch_loss_metrics(
            self,
            all_policy_logits,
            all_reference_logits,
            label
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Computes the sum log probabilities of the labels under the given logits.

        Otherwise, the average log probabilities.

        Args:
            all_policy_logits: Logits of the policy model.
            all_reference_logits: Logits of the reference model.
            label: The label tensor.

        Returns:
            A tuple containing the loss tensor and a dictionary of metrics.
        """
        metrics = {}

        (
            policy_chosen_log_probs,
            policy_rejected_log_probs,
            policy_chosen_log_probs_avg,
        ) = self._compute_log_probs(all_policy_logits, label)

        reference_chosen_log_probs, reference_rejected_log_probs, *_ = self._compute_log_probs(
            all_reference_logits,
            label
        )
        # compute DPO loss
        losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
            policy_chosen_log_probs,
            policy_rejected_log_probs,
            reference_chosen_log_probs,
            reference_rejected_log_probs,
        )

        sft_loss = -policy_chosen_log_probs_avg
        if self.args.pref_ftx > 1e-6:
            losses += self.args.pref_ftx * sft_loss

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = ""
        metrics["{}rewards/accuracies".format(prefix)] = reward_accuracies.detach().mean()

        return losses.mean(), metrics

    def _compute_log_probs(self, all_logits, label=None) -> Tuple[torch.Tensor, ...]:
        """
        Computes the sum log probabilities of the labels under given logits if loss_type.
        Otherwise, the average log probabilities.
        Assuming IGNORE_INDEX is all negative numbers, the default is -100.

        Args:
            all_logits: The logits tensor.
            label: The label tensor.

        Returns:
            A tuple containing the log probabilities and other tensors.
        """
        label = label[:, 1:].clone()
        all_logits = all_logits[:, :-1, :]
        batch_size = all_logits.size(0) // 2

        all_log_probs, valid_length, _ = compute_log_probs(
            all_logits,
            label
        )

        chosen_log_probs, rejected_log_probs = all_log_probs.split(batch_size, dim=0)
        chosen_length, _ = valid_length.split(batch_size, dim=0)
        all_results = (chosen_log_probs, rejected_log_probs, chosen_log_probs / chosen_length)

        return all_results
