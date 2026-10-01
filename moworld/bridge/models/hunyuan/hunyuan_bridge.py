# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from megatron.core.transformer.module import MegatronModule

from bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from bridge.models.conversion.model_bridge import MegatronModelBridge
from bridge.models.conversion.param_mapping import (
    ReplicatedMapping,
)
from mindspeed_mm.models.sora_model import SoRAModel


class HunyuanVideo_1_5_DiffusionTransformer():
    pass


@MegatronModelBridge.register_bridge(source=HunyuanVideo_1_5_DiffusionTransformer, target=SoRAModel)
class HunyuanBridge(MegatronModelBridge):
    """
    Megatron Bridge for Hunyuan Video 1.5 model.

    """

    def mapping_registry(self) -> MegatronMappingRegistry:
        mapping_list = []

        mapping_list.extend(
            [
                ReplicatedMapping(
                    megatron_param="**",
                    hf_param="**",
                ),
            ]
        )
        return MegatronMappingRegistry(*mapping_list)
