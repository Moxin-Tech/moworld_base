# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from megatron.training import get_args
from megatron.core import mpu
from megatron.core.transformer.module import fp32_to_float16, float16_to_fp32, Float16Module


def _transform_model_dtype(module, transform, ignore_child_modules):
    for child_module in module.children():
        if child_module in ignore_child_modules:
            continue
        child_module = transform(child_module)
    return module


def float16Module_init(self, config, module):
    super(Float16Module, self).__init__(config)

    # if AEModel use fp32
    ae_config = getattr(config.mm.model, "ae", None)
    ae_float32 = False
    if ae_config is not None and getattr(ae_config, "dtype", None) == torch.float32:
        module.ae = module.ae.float()
        ae_float32 = True

    if config.fp16:
        if ae_float32:
            self.add_module("module", _transform_model_dtype(module, lambda module: module.half(), [module.ae]))
        else:
            self.add_module("module", module.half())

        def float16_convertor(val):
            return val.half()

    elif config.bf16:
        if ae_float32:
            self.add_module("module", _transform_model_dtype(module, lambda module: module.bfloat16(), [module.ae]))
        else:
            self.add_module("module", module.bfloat16())

        def float16_convertor(val):
            return val.bfloat16()
    else:
        raise Exception("should not be here")

    self.float16_convertor = float16_convertor


def float16Module_forward(self, *inputs, **kwargs):
    args = get_args()
    if mpu.is_pipeline_first_stage():
        # if AEModel use fp16 or bf16
        ae_config = getattr(args.mm.model, "ae", None)
        if ae_config is not None and getattr(ae_config, "dtype", None) != torch.float32:
            inputs = fp32_to_float16(inputs, self.float16_convertor)
    outputs = self.module(*inputs, **kwargs)
    if mpu.is_pipeline_last_stage():
        outputs = float16_to_fp32(outputs)
    return outputs
