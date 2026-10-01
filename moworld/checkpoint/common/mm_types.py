from typing import Dict, Tuple, List

import torch

STATE_DICT_T = Dict[str, torch.Tensor]
QV_NAME_T = Tuple[str, str]
QKV_NAME_T = Tuple[str, str, str]
PP_LAYER_NUM_T = List[int]
VPP_LAYER_NUM_T = List[PP_LAYER_NUM_T]
