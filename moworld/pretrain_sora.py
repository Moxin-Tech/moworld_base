# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain SoRA."""
from collections import defaultdict
import os
os.environ["USE_TF"] = "FALSE"
import torch
import torch_npu

import mindspeed.megatron_adaptor
from mindspeed.megatron_adaptor import get_mindspeed_args
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.training import get_args, print_rank_0
from megatron.training.utils import (
    average_losses_across_data_parallel_group,
    unwrap_model,
)

from mindspeed_mm.configs.config import mm_extra_args_provider
from mindspeed_mm.training import pretrain
from mindspeed_mm.data import build_mm_dataloader, build_mm_dataset
from mindspeed_mm.data.data_utils.constants import (
    VIDEO,
    PROMPT_IDS,
    PROMPT_MASK,
    VIDEO_MASK
)
from mindspeed_mm.data.data_utils.utils import build_iterations
from mindspeed_mm.models.sora_model import SoRAModel
import mindspeed_mm.utils.dpcp_utils as dpcp
mindspeed_args = get_mindspeed_args()

if hasattr(mindspeed_args, "ai_framework") and mindspeed_args.ai_framework == "mindspore" and mindspeed_args.optimization_level >= 0:
    import mindspeed_mm.mindspore.mindspore_adaptor


def model_provider(pre_process=True, post_process=True):
    """Builds the model."""
    args = get_args()
    print_rank_0("building SoRA model ...")
    model = SoRAModel(args.mm.model)

    if mpu.get_pipeline_model_parallel_world_size() > 1:
        if not hasattr(model.predictor, "initialize_pipeline_tensor_shapes"):
            raise AttributeError("The predictor should provide initialize_pipeline_tensor_shapes for PP_size>1. ")
        args.pipeline_tensor_shapes = model.predictor.initialize_pipeline_tensor_shapes()
        setattr(forward_step, 'pipeline_tensor_shapes', args.pipeline_tensor_shapes)
        # modification for core 0.12.1 when using multi-parameter PP
        model.config.pipeline_tensor_shapes = args.pipeline_tensor_shapes

    return model


def get_batch_on_this_tp_rank(data_iterator):
    """Get a batch from the data iterator on the current TP rank."""
    args = get_args()
    interleaved = args.mm.model.interleaved \
        if hasattr(args.mm.model, "interleaved") else False
    if data_iterator is not None:
        batch = next(data_iterator, None)
    else:
        return None
    # data is loaded in cpu for interleaved.
    if batch is not None and not interleaved:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(torch_npu.npu.current_device())
    return batch


def get_batch(data_iterator):
    """Generate a batch."""
    if mpu.is_pipeline_first_stage():
        batch = get_batch_on_this_tp_rank(data_iterator)
        return batch
    else:
        return None


def loss_func(output_tensor):
    """Loss function."""
    loss = output_tensor[0].mean()
    averaged_loss = average_losses_across_data_parallel_group([loss])
    loss = loss.unsqueeze(0)
    # NOTE: Megatron multiplies the loss by `mpu.get_context_parallel_world_size()`(cp_size).
    # To keep the effective loss unchanged we divide by `cp_size` before returning from this function.
    #
    # Old behaviour: loss was returned unscaled. We scale the grad in backward of all-gather.
    #
    # New behaviour: loss = loss_raw / cp_size  (done right here).
    #                loss = loss * cp_size  (megatron).
    return loss / mpu.get_context_parallel_world_size(), {"loss": averaged_loss[0]}


def get_batch_for_step(data_iterator):
    """Get batches by step interval from data iterator."""
    args = get_args()
    enable_encoder_dp = getattr(args.mm.model, "enable_encoder_dp", False)
    encoder_offload_interval = getattr(args.mm.model, "encoder_offload_interval", 1)
    args.curr_forward_iteration += 1

    tp_cp_group_size = torch.distributed.get_world_size(mpu.get_tensor_and_context_parallel_group())
    encoder_dp_interval = tp_cp_group_size if enable_encoder_dp else 1
    get_batch_interval = encoder_dp_interval * encoder_offload_interval
    batches = []

    if get_batch_interval == 1 or args.curr_forward_iteration % get_batch_interval == 1:
        for _ in range(encoder_offload_interval):
            batch = get_batch(data_iterator)
            if batch is not None:
                batches.append(batch)

    return batches, get_batch_interval


def forward_step(data_iterator, model):
    """Forward step."""
    batch, video, prompt_ids, video_mask, prompt_mask = {}, None, None, None, None
    skip_encode = False
    if mpu.is_pipeline_first_stage():
        if dpcp.is_use_dynamic_dpcp():
            batches = [dpcp.get_dpcp_batch_for_step(data_iterator)]
            get_batch_interval = 0
        else:
            batches, get_batch_interval = get_batch_for_step(data_iterator)
        skip_encode = not batches
        i2v_params = defaultdict(list)
        # while encoder dp or encoder interleave offload is enabled. reconstruct data as list: [step_1, ... step_n]
        if get_batch_interval > 1 and len(batches) >= 1:
            video, prompt_ids, video_mask, prompt_mask = [], [], [], []
            for single_batch in batches:
                _prompt_ids = single_batch.pop(PROMPT_IDS, None)
                _prompt_mask = single_batch.pop(PROMPT_MASK, None)
                _prompt_ids = _prompt_ids if isinstance(_prompt_ids, (list, tuple)) else [_prompt_ids]
                _prompt_mask = _prompt_mask if isinstance(_prompt_mask, (list, tuple)) else [_prompt_mask]
                video.append(single_batch.pop(VIDEO, None))
                video_mask.append(single_batch.pop(VIDEO_MASK, None))
                prompt_ids.append(_prompt_ids)
                prompt_mask.append(_prompt_mask)
                for key, value in single_batch.items():
                    i2v_params[key].append(value)
            batch = i2v_params
        elif len(batches) == 1:
            batch = batches[0]
            video = batch.pop(VIDEO, None)
            prompt_ids = batch.pop(PROMPT_IDS, None)
            video_mask = batch.pop(VIDEO_MASK, None)
            prompt_mask = batch.pop(PROMPT_MASK, None)

    output_tensor_list = model(video, prompt_ids, video_mask, prompt_mask=prompt_mask, skip_encode=skip_encode, **batch)
    return output_tensor_list, loss_func


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    data_config = args.mm.data
    generator = torch.Generator()
    dataloader_seed = getattr(args, 'seed')
    generator.manual_seed(dataloader_seed)
    train_dataset = build_mm_dataset(data_config.dataset_param)

    enable_encoder_dp = args.mm.model.enable_encoder_dp if hasattr(args.mm.model, "enable_encoder_dp") else False
    if enable_encoder_dp:
        process_group = torch.distributed.group.WORLD
    else:
        process_group = mpu.get_data_parallel_group()

    train_dataloader = build_mm_dataloader(
        train_dataset,
        data_config.dataloader_param,
        process_group=process_group,
        consumed_samples=args.consumed_train_samples,
        dataset_param=data_config.dataset_param,
        generator=generator,
    )
    data_iterator, _, _ = build_iterations(train_dl=train_dataloader)
    return data_iterator, None, None


if __name__ == "__main__":
    from mindspeed_mm.patchs import torch_dcp_patch
    train_valid_test_datasets_provider.is_distributed = True
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=mm_extra_args_provider,
        args_defaults={"dataloader_type": "external", "vision_pretraining": False, "curr_forward_iteration": 0},
    )
