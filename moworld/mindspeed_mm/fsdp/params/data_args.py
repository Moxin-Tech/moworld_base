from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional
import logging

from mindspeed_mm.fsdp.data.data_utils.func_utils.convert import DatasetAttr
from mindspeed_mm.fsdp.data.data_utils.func_utils.convert import DataArguments as BasicDataAruments
from mindspeed_mm.fsdp.data.data_utils.func_utils.model_args import ProcessorArguments
from mindspeed_mm.config.arguments.base_args import BaseArguments

logger = logging.getLogger(__name__)


class DataSetArguments(BaseArguments):
    dataset_type: str = field(
        metadata={"help": "Type of dataset to use."}
    )
    basic_parameters: BasicDataAruments = field(default_factory=BasicDataAruments)
    preprocess_parameters: Optional[ProcessorArguments] = field(default_factory=ProcessorArguments)
    attr: DatasetAttr = field(default_factory=DatasetAttr)


class CollateArguments(BaseArguments):
    model_name: str = field(metadata={"help": "Name of the model for which collation is configured."})
    ignore_pad_token_for_loss: bool = field(
        default=False,
        metadata={"help": ""}
    )
    pad_to_multiple_of: int = field(
        default=8,
        metadata={"help": "Pad sequences to a multiple of this value for efficient processing."}
    )


class DataloaderArguments(BaseArguments):
    dataloader_mode: str = field(metadata={"help": "Mode of dataloader."})
    sampler_type: str = field(metadata={"help": "Type of sampler to use."})
    shuffle: Optional[bool] = field(metadata={"help": "Whether to shuffle the data during training."})
    drop_last: bool = field(metadata={"help": "Whether to drop the last incomplete batch if dataset size is not divisible by batch size."})
    pin_memory: bool = field(metadata={"help": "Whether to pin memory for faster data transfer to GPU."})
    collate_param: CollateArguments = field(default_factory=CollateArguments)
    num_workers: int = field(default=2, metadata={"help": "Number of worker processes for data loading."})
    enable_preload: bool = field(
        default=False,
        metadata={"help": "Whether to enable async data preloading to overlap CPU-H2D transfer with training."},
    )


class DataArguments(BaseArguments):
    dataset_param: DataSetArguments = field(default_factory=DataSetArguments)
    dataloader_param: DataloaderArguments = field(default_factory=DataloaderArguments)
