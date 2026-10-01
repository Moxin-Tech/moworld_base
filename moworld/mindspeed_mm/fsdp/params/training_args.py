# Copyright 2025 Bytedance Ltd. and/or its affiliates
from dataclasses import field
from typing import List, Literal, Optional
import logging
import os

from mindspeed_mm.fsdp.params.lora_args import LoraArguments
from mindspeed_mm.config.arguments.base_args import BaseArguments
from mindspeed.fsdp.parallel_engine_config import QuantizeConfig

logger = logging.getLogger(__name__)


class Profiler(BaseArguments):
    enable: bool = field(
        default=False,
        metadata={"help": "Enable profiling."},
    )
    start_step: int = field(
        default=1,
        metadata={"help": "Start step for profiling."},
    )
    end_step: int = field(
        default=2,
        metadata={"help": "End step for profiling."},
    )
    save_path: str = field(
        default="./profiling",
        metadata={"help": "Direction to export the profiling result."},
    )
    record_shapes: bool = field(
        default=True,
        metadata={"help": "Whether or not to record the shapes of the input tensors."},
    )
    with_memory: bool = field(
        default=True,
        metadata={"help": "Whether or not to profile the memory usage."},
    )
    with_stack: bool = field(
        default=True,
        metadata={"help": "Whether or not to record the stack traces."},
    )
    ranks: List[int] = field(
        default_factory=lambda: [0],
        metadata={
            "help": "List of ranks to profile (default is rank 0 only)"
        },
    )

    def model_post_init(self, __context):
        self._train_steps = -1
        self.local_rank = int(os.getenv("LOCAL_RANK"))
        self.global_rank = int(os.getenv("RANK"))
        self.world_size = int(os.getenv("WORLD_SIZE"))

        # determine whether to profile this rank
        if self.enable:
            if self.global_rank in self.ranks:
                self.profile_this_rank = True
            else:
                self.profile_this_rank = False
        else:
            self.profile_this_rank = False


class LearningRateScalingPlanArguments(BaseArguments):
    match: str = field(
        metadata={"help": "Regex pattern to match the layerwise learning rate."},
    )
    scale: float = field(
        default=1.0,
        metadata={"help": "Learning rate ratio for the matched layers."},
    )


class TrainingArguments(BaseArguments):
    profile: Profiler = field(default_factory=Profiler)
    lora: LoraArguments = field(default_factory=LoraArguments)
    lr: float = field(
        default=5e-5,
        metadata={"help": "Maximum learning rate or defult learning rate, or init learning rate for warmup."},
    )
    lr_min: float = field(
        default=0.0,
        metadata={"help": "Minimum learning rate."},
    )
    lr_start: float = field(
        default=0.0,
        metadata={"help": "Learning rate for warmup start. Default to 0.0."},
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "L2 regularization strength."},
    )
    no_decay_modules: List[str] = field(
        default_factory=list,
        metadata={"help": "Modules without weight decay, for example, RMSNorm."},
    )
    no_decay_params: List[str] = field(
        default_factory=list,
        metadata={"help": "Parameters without weight decay, for example, bias."},
    )
    optimizer: Literal["adamw", "muon"] = field(
        default="adamw",
        metadata={"help": "Optimizer. Supported: adamw, muon. Default to adamw."},
    )
    matched_adamw_rms: float = field(
        default=0.2,
        metadata={
            "help": (
                "Matched AdamW RMS value for Muon optimizer. "
                "Controls how closely Muon matches AdamW update magnitude."
            )
        },
    )
    muon_momentum: float = field(
        default=0.95,
        metadata={"help": "Momentum coefficient for Muon internal SGD."},
    )
    ns_steps: int = field(
        default=5,
        metadata={"help": "Number of Newton-Schulz iterations for Muon orthogonalization."},
    )
    adam_fused: bool = field(
        default=True,
        metadata={"help": "Whether to use fused AdamW optimizer for better performance."},
    )
    adam_beta1: float = field(
        default=0.9,
        metadata={"help": "The beta1 parameter for Adam optimizer."},
    )
    adam_beta2: float = field(
        default=0.999,
        metadata={"help": "The beta2 parameter for Adam optimizer."},
    )
    adam_eps: float = field(
        default=1e-8,
        metadata={"help": "The epsilon parameter for Adam optimizer for numerical stability."},
    )
    clip_grad: float = field(
        default=1.0,
        metadata={"help": "Clip value for gradient norm. Gradients with norm larger than this will be scaled down."},
    )
    clip_grad_foreach: bool = field(
        default=True,
        metadata={"help": "Whether to use foreach implementation for gradient clipping for better performance."},
    )
    micro_batch_size: int = field(
        default=1,
        metadata={"help": "Micro batch size. The number of samples per iteration on each device."},
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Gradient accumulation steps. If None, use `global_batch_size` // (`micro_batch_size` * data_parallel_size)."},
    )
    lr_warmup_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of learning rate warmup steps."},
    )
    lr_decay_style: str = field(
        default="constant",
        metadata={"help": "Name of the learning rate scheduler."},
    )
    lr_decay_ratio: float = field(
        default=1.0,
        metadata={"help": "Ratio of learning rate decay steps."},
    )
    enable_full_determinism: bool = field(
        default=False,
        metadata={"help": "Enable full determinism."},
    )
    ckpt_manager: str = field(
        default="dcp",
        metadata={"help": "Checkpoint manager."},
    )
    save_async: bool = field(
        default=False,
        metadata={"help": "Whether to save checkpoint asynchronously."},
    )
    load_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to checkpoint to resume from."},
    )
    save_steps: int = field(
        default=0,
        metadata={"help": "Number of steps between two checkpoint saves."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed."},
    )
    max_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Max training steps per epoch. (for debug)"},
    )
    init_model_with_meta_device: bool = field(
        default=False,
        metadata={"help": "Whether to initialize model weights on meta device for memory efficiency."},
    )
    train_iters: int = field(
        default=10000,
        metadata={"help": "Total number of training iterations."},
    )
    train_epochs: Optional[int] = field(
        default=None,
        metadata={"help": "Number of training epochs."},
    )
    load: str = field(
        default=None,
        metadata={"help": "Path to load checkpoint from. Used for resuming training."},
    )
    load_format: Literal["auto", "hf", "dcp"] = field(
        default="auto",
        metadata={
            "help": (
                "Format of the load path. 'auto' detects HF safetensors vs DCP "
                "automatically (recommended); set explicitly to override."
            )
        },
    )
    load_strict: bool = field(
        default=False,
        metadata={"help": "Whether to load checkpoint strictly."},
    )
    load_rank0_and_broadcast: bool = field(
        default=False,
        metadata={"help": "Whether to load checkpoint on rank 0 and broadcast to other ranks."},
    )
    no_load_optim: bool = field(
        default=False,
        metadata={"help": "Do not load optimizer when loading checkpoint."},
    )
    no_load_rng: bool = field(
        default=False,
        metadata={"help": "Do not load rng state when loading checkpoint."},
    )
    save: str = field(
        default=None,
        metadata={"help": "Directory path to save checkpoints to."},
    )
    no_save_optim: bool = field(
        default=False,
        metadata={"help": "Do not save current optimizer."},
    )
    no_save_rng: bool = field(
        default=False,
        metadata={"help": "Do not save current rng state."},
    )
    log_interval: int = field(
        default=1,
        metadata={"help": "Number of steps between logging training metrics."},
    )
    save_interval: int = field(
        default=1,
        metadata={"help": "Number of steps between checkpoint saves."},
    )
    use_deter_comp: bool = field(
        default=False,
        metadata={"help": "Whether to use deterministic computation for reproducibility."},
    )
    allow_hf32: bool = field(
        default=None,
        metadata={"help": "This switch controls the value of `allow_hf32`."},
    )
    plugin: List[str] = field(
        default_factory=list,
        metadata={"help": "Path to load custom dataset/model plugin."},
    )
    lr_scaling_plan: Optional[List[LearningRateScalingPlanArguments]] = field(
        default=None,
        metadata={"help": "Learning rate ratios for layers in the model. Each element is a regex pattern to match the layer name and a ratio to apply to the learning rate."},
    )

    quantization_plan: QuantizeConfig = field(
        default_factory=QuantizeConfig,
        metadata={"help": "Quantization plan."},
    )

    def model_post_init(self, __context):
        self._train_steps = -1
        self.local_rank = int(os.getenv("LOCAL_RANK"))
        self.global_rank = int(os.getenv("RANK"))
        self.world_size = int(os.getenv("WORLD_SIZE"))

        if self.lr < self.lr_start:
            raise ValueError(f"Learning rate {self.lr} < starting lr {self.lr_start}. Check scheduler configuration.")

        if self.lr < self.lr_min:
            raise ValueError(f"Learning rate {self.lr} < minimum lr {self.lr_min}. Check scheduler configuration.")

    def compute_distributed_training(
        self, parallel_args
    ) -> None:
        """
        Computes the training steps per epoch according to the data length.
        """
        data_parallel_size = getattr(parallel_args, "data_parallel_size", 1)

        if self.gradient_accumulation_steps is None:
            self.global_batch_size = self.micro_batch_size * data_parallel_size
            self.gradient_accumulation_steps = 1
            logger.info("`gradient_accumulation_steps` is None, disable gradient accumulation.")
        else:
            self.global_batch_size = self.micro_batch_size * data_parallel_size * self.gradient_accumulation_steps
