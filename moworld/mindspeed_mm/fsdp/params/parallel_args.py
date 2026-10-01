from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union
import logging
import os
import torch

from mindspeed_mm.fsdp.utils.device import IS_NPU_AVAILABLE
from mindspeed_mm.config.arguments.base_args import BaseArguments

logger = logging.getLogger(__name__)


class FSDPPlanConfig(BaseArguments):
    """Configuration for Fully Sharded Data Parallelism (FSDP) plan."""
    ignored_modules: List[str] = field(default_factory=list)
    apply_modules: List[str] = field(default_factory=list)

    # mp_policy settings
    param_dtype: Optional[str] = None
    reduce_dtype: Optional[str] = None
    output_dtype: Optional[str] = None
    cast_forward_inputs: bool = True
    reshard_after_forward: bool = True

    # prefetch settings
    num_to_forward_prefetch: Optional[int] = 0
    num_to_backward_prefetch: Optional[int] = 0

    # pregather settings
    pregather: bool = False

    # fsdp2 hook manager
    hook_modules: Optional[List[str]] = None

    cpu_offload: bool = False


class TPPlanConfig(BaseArguments):
    """Configuration for Tensor Parallelism (TP) plan."""
    colwise_parallel: List[str] = field(default_factory=list)
    rowwise_parallel: List[str] = field(default_factory=list)
    sequence_parallel: List[str] = field(default_factory=list)


class EPPlanConfig(BaseArguments):
    """Configuration for Expert Parallelism (EP) plan for MoE models."""
    apply_modules: List[str] = field(default_factory=list)
    use_npu_fused_ops: bool = True
    dispatcher: Literal["alltoall", "allgather", "mc2"] = "alltoall"
    apply_efsdp_modules: List[str] = field(default_factory=list)
    _gradient_divide_factor: float = None


class RecomputePlanConfig(BaseArguments):
    """Configuration for recompute plan."""
    apply_modules: List[str] = field(default_factory=list)
    use_reentrant: bool = False


class ParallelArguments(BaseArguments):
    data_parallel_size: Optional[int] = field(
        default=None,
        metadata={"help": "Size of data parallelism. If None, calculated automatically."}
    )

    fully_shard_parallel_size: Union[str, int] = field(
        default="auto",
        metadata={"help": "Fully Sharded Data Parallel size. (Sharding parameters)"}
    )

    fsdp_plan: FSDPPlanConfig = field(default_factory=FSDPPlanConfig)

    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Tensor Parallel size. (Cols/Rows splitting)"}
    )
    tp_plan: TPPlanConfig = field(default_factory=TPPlanConfig)

    ring_attention_size: int = 1 # Size for Ring Attention
    ulysses_parallel_size: int = 1 # Size for Ulysses parallelism

    expert_parallel_size: int = field(
        default=1,
        metadata={"help": "Expert Parallel size for MoE models."}
    )
    expert_fully_shard_parallel_size: int = field(
        default=None,
        metadata={"help": "FSDP size inside Expert Parallel groups."}
    )
    ep_plan: EPPlanConfig = field(default_factory=EPPlanConfig)

    recompute: bool = field(
        default=False,
        metadata={"help": "Whether to enable Gradient Checkpointing (Activation Recomputation)."}
    )
    recompute_plan: RecomputePlanConfig = field(default_factory=RecomputePlanConfig)

    def model_post_init(self, __context):
        self.local_rank = int(os.getenv("LOCAL_RANK"))
        self.global_rank = int(os.getenv("RANK"))
        self.world_size = int(os.getenv("WORLD_SIZE"))

        if self.fully_shard_parallel_size == "auto":
            # If -1, use all remaining processes after tensor parallelism for FSDP
            self.fully_shard_parallel_size = self.world_size // self.tensor_parallel_size
        else:
            self.fully_shard_parallel_size = int(self.fully_shard_parallel_size)

        if self.expert_fully_shard_parallel_size is None:
            self.expert_fully_shard_parallel_size = self.world_size // self.expert_parallel_size

        if (
            self.world_size
            % (
                self.tensor_parallel_size
                * self.ring_attention_size
                * self.ulysses_parallel_size
            )
            != 0
        ):
            raise ValueError(
                f"World size should be a multiple of tensor_parallel_size: {self.tensor_parallel_size}, ulysses_parallel_size: {self.ulysses_parallel_size}, ring_attention_size: {self.ring_attention_size}."
            )
        if (
            self.world_size
            % (
                self.tensor_parallel_size
                * self.fully_shard_parallel_size
            )
            != 0
        ):
            raise ValueError(
                f"World size should be a multiple of tensor_parallel_size: {self.tensor_parallel_size}, fully_shard_parallel_size: {self.fully_shard_parallel_size}."
            )

        dp_size = self.world_size // (
            self.tensor_parallel_size
            * self.ring_attention_size
            * self.ulysses_parallel_size
        )
        if self.data_parallel_size is None:
            self.data_parallel_size = dp_size

        if self.data_parallel_size != dp_size:
            raise ValueError(f"data_parallel_size should be equal to tensor_parallel_size: {self.tensor_parallel_size}, ulysses_parallel_size: {self.ulysses_parallel_size}, ring_attention_size: {self.ring_attention_size}.")

        if self.fully_shard_parallel_size < self.ring_attention_size * self.ulysses_parallel_size:
            raise ValueError("fully shard parallel size should be greater the ring_attention_size * ulysses_parallel_size.")
        if self.tensor_parallel_size != 1:
            raise ValueError("Tensor parallel size not supported yet.")
        if self.ring_attention_size != 1 and not IS_NPU_AVAILABLE:
            raise ValueError("Ring Attention only support on NPU.")
