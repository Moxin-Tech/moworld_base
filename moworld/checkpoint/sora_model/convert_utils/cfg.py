from typing import Union, List
from pydantic import BaseModel, PositiveInt, NonNegativeInt


class ParallelConfig(BaseModel):
    pp_layers: Union[List[NonNegativeInt], List[List[NonNegativeInt]]] = []
    tp_size: PositiveInt = 1
    ep_size: PositiveInt = 1


class ConvertConfig(BaseModel):
    source_path: str
    lora_path: str = ""
    hf_dir: str = ""
    target_path: str
    target_parallel_config: ParallelConfig
