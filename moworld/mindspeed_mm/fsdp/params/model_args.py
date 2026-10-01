# pylint: skip-file
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from mindspeed_mm.config.arguments.base_args import BaseArguments


class ModelArguments(BaseArguments):
    model_id: Optional[str] = field(
        default=None,
        metadata={
            "help": "Model identifier.If not provided, will be generated automatically based on model_name_or_path."
        },
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust remote code (e.g., custom modeling files) when loading model"},
    )
    attn_implementation: Optional[
        Literal[
            "eager",
            "sdpa",
            "flash_attention_2",
            "flash_attention_3",
            "native-sparse",
        ]
    ] = field(
        default="flash_attention_2",
        metadata={"help": "Attention implementation to use."},
    )
    freeze: List[str] = field(
        default_factory=list,
        metadata={"help": "List of module names to freeze during training."},
    )
    mtp_num_layers: int = field(
        default=0,
        metadata={"help": "Number of mtp layers."},
    )
    mtp_loss_scaling_factor: float = field(
        default=0.1,
        metadata={"help": "Mtp loss scaling factor."},
    )
