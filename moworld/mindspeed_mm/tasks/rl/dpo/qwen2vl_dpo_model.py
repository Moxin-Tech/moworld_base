# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
import torch

from megatron.core import mpu
from megatron.core.pipeline_parallel.schedules import get_attr_wrapped_model
from megatron.training import get_args
from mindspeed_mm.tasks.rl.hyper_model import HyperModelABC
from mindspeed_mm.tasks.rl.utils import get_attr_from_wrapped_model


class Qwen2VLDPOModel(HyperModelABC):
    """
    The hyper model wraps multiple models required in reinforcement learning into a single model,
    maintaining the original distributed perspective unchanged.
    """

    def __init__(self, train_model, refer_model):
        super().__init__()
        self.args = get_args()
        self.train_model = train_model
        self.refer_model = refer_model

        self.ori_micro_batch_size = self.args.micro_batch_size
        self.new_micro_batch_size = self.args.actual_micro_batch_size // 2

        self.input_tensor = None

    def __call__(self, input_ids, attention_mask, pixel_values, image_grid_thw, labels):
        self.set_input_tensor()
        self.args.micro_batch_size = self.new_micro_batch_size

        refer_input_ids = input_ids.detach()
        refer_attention_mask = attention_mask.detach()
        refer_pixel_values = pixel_values.detach()
        refer_image_grid_thw = image_grid_thw.detach()
        refer_labels = labels.detach()

        with torch.no_grad():
            refer_output = self.refer_model[0](input_ids=refer_input_ids, pixel_values=refer_pixel_values, image_grid_thw=refer_image_grid_thw,
                          attention_mask=refer_attention_mask, labels=refer_labels)

        policy_output = self.train_model[0](input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                          attention_mask=attention_mask, labels=labels)

        output_tensor = None
        if mpu.is_pipeline_last_stage():
            refer_output = refer_output["logits"]
            policy_output = policy_output["logits"]
            output_tensor = torch.cat((policy_output, refer_output), dim=0)
        else:
            output_tensor = torch.cat((policy_output, refer_output), dim=1)

        self.args.micro_batch_size = self.ori_micro_batch_size

        return output_tensor

    def set_input_tensor(self) -> None:
        """Sets input tensor to the hyper model.

        See megatron.model.transformer.set_input_tensor()
        """
        input_tensor = get_attr_from_wrapped_model(self.train_model[0], "input_tensor_dpo")

        if input_tensor[0] is not None:
            self.input_tensor = torch.chunk(input_tensor[0], 2, dim=1)

            set_train_input_tensor = get_attr_wrapped_model(self.train_model[0], "set_input_tensor")
            set_refer_input_tensor = get_attr_wrapped_model(self.refer_model[0], "set_input_tensor")
            set_train_input_tensor(self.input_tensor[0])
            set_refer_input_tensor(self.input_tensor[1])
