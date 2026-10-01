# pylint: skip-file
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from mindspeed_mm.config.arguments.base_args import BaseArguments


class RecomputePlanConfig(BaseArguments):
    """Configuration for recompute plan"""
    apply_modules: List[str] = field(default_factory=list)
    use_reentrant: bool = False


class ChunkLossPlanConfig(BaseArguments):
    apply_module: str = field(
        default="lm_head",
        metadata={"help": "module that applied chunk loss"}
    )
    chunk_size: int = field(
        default=1024,
        metadata={"help": "Size of each chunk loss"},
    )
    total_chunk_size: int = field(
        default=4096,
        metadata={"help": "Size of total chunk loss"},
    )


class LossArguments(BaseArguments):
    loss_type: Optional[str] = field(
        default="raw",
        metadata={"help": "Type of loss function type, If ot provided, will be computed based on raw model loss function"},
    )
    router_aux_loss_coef: float = field(
        default=0.0,
        metadata={"help": "Router Auxiliary Loss Coefficient"},
    )
    router_aux_loss_offload: bool = field(
        default=False,
        metadata={"help": "Whether apply router auxiliary loss offload"},
    )


class ActivationOffloadPlanConfig(BaseArguments):
    apply_modules: List[str] = field(
        default=None,
        metadata={"help": "module that applied activation offload"}
    )


class ChunkMbsPlanConfig(BaseArguments):
    apply_modules: List[str] = field(
        default=None,
        metadata={"help": "module that applied chunkmbs"}
    )

    chunk_mbs: int = field(
        default=1,
        metadata={"help": "chunk_mbs, chunked micro batch size"}
    )

    batch_dim: int = field(
        default=0,
        metadata={"help": "chunk_mbs, batchsize dim"}
    )

    chunk_arg_indexs: List[int] = field(
        default=[0],
        metadata={"help": "chunk_mbs, chunk args indexs"}
    )

    chunk_kwarg_names: List[str] = field(
        default=[],
        metadata={"help": "chunk_mbs, chunk kwarg names"}
    )


class FeatureArguments(BaseArguments):
    recompute_plan: RecomputePlanConfig = field(default_factory=RecomputePlanConfig)

    loss_cfg: LossArguments = field(default_factory=LossArguments)

    enable_chunk_loss: bool = field(
        default=False,
        metadata={"help": "Whether apply chunkloss for loss compute"},
    )
    enable_dynamic_chunk_loss: bool = field(
        default=False,
        metadata={"help": "Whether apply dynamic chunkloss for loss compute"},
    )
    chunkloss_plan: ChunkLossPlanConfig = field(default_factory=ChunkLossPlanConfig)

    enable_activation_offload: bool = field(
        default=False,
        metadata={"help": "Whether apply activation offload"}
    )
    activation_offload_plan: ActivationOffloadPlanConfig = field(default_factory=ActivationOffloadPlanConfig)

    enable_chunk_mbs: bool = field(
        default=False,
        metadata={"help": "Whether apply chunk_mbs"}
    )
    chunkmbs_plan: ChunkMbsPlanConfig = field(default_factory=ChunkMbsPlanConfig)
