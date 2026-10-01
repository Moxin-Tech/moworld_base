# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain VLM (ViT+MLP+LLM) MODEL."""
import os
os.environ["USE_TF"] = "FALSE"
from copy import deepcopy
from functools import partial
from typing import Dict, Any

from datasets import Dataset
import torch

import mindspeed.megatron_adaptor
from mindspeed.megatron_adaptor import get_mindspeed_args
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.training import get_args, print_rank_0
from megatron.training.utils import average_losses_across_data_parallel_group
from mindspeed_mm.configs.config import mm_extra_args_provider
from mindspeed_mm.data import build_mm_dataloader, build_mm_dataset
from mindspeed_mm.data.data_utils.utils import build_iterations
from mindspeed_mm.models.vlm_model import VLMModel
from mindspeed_mm.patchs import dummy_optimizer_patch
from mindspeed_mm.training import pretrain
from mindspeed_mm.utils.transformer_model_config import get_model_config
from mindspeed_mm.utils.hetero_parallel import change_parallel_state, apply_hetero_parallel_hooks
from mindspeed_mm.utils.utils import EncoderBalanceComm
from mindspeed_mm.utils.hetero_parallel import hetero_align_config
from mindspeed_mm.utils.utils import compute_token_level_loss
mindspeed_args = get_mindspeed_args()
if hasattr(mindspeed_args, "ai_framework") and mindspeed_args.ai_framework == "mindspore" and mindspeed_args.optimization_level >= 0:
    import mindspeed_mm.mindspore.mindspore_adaptor


def model_provider(pre_process=True, post_process=True, modules=None):
    """Builds the model."""
    if modules is None:
        modules = ['image_encoder', 'audio_encoder', 'text_decoder']

    args = get_args()
    print_rank_0("building VLMModel ...")
    vlm_config = deepcopy(args.mm.model)

    # distinguish model construct stage when pipeline parallel
    vlm_config.pre_process = pre_process
    vlm_config.post_process = post_process

    _configure_modules(vlm_config, modules)

    model = VLMModel(vlm_config)

    if args.hetero_parallel:
        print_rank_0("apply hetero parallel ...")
        apply_hetero_parallel_hooks(model)

    _apply_freezing(model, vlm_config)

    return model


def _configure_modules(vlm_config, modules):
    """Configure each module based on the modules list."""
    module_configs = {
        'image_encoder': _configure_image_encoder,
        'audio_encoder': _configure_audio_encoder,
        'text_decoder': _configure_text_decoder
    }

    for module_name, config_func in module_configs.items():
        if module_name in modules and hasattr(vlm_config, module_name):
            config_func(vlm_config)
        else:
            setattr(vlm_config, module_name, None)


def _configure_image_encoder(vlm_config):
    """Configure image encoder module."""
    if get_args().hetero_parallel:
        hetero_align_config(vlm_config.image_encoder.vision_encoder, vlm_config.image_encoder)
        hetero_align_config(vlm_config.image_encoder.vision_projector, vlm_config.image_encoder)

    # MindSpeed needs to validate the CP configuration; the attention head must be divisible by the CP sizes.
    # However, since the vision projector does not have an attention head, special handling is required.
    vlm_config.image_encoder.vision_projector.context_parallel_size = 1
    vlm_config.image_encoder.vision_encoder.expert_model_parallel_size = 1
    vlm_config.image_encoder.vision_projector.expert_model_parallel_size = 1
    vlm_config.image_encoder.vision_encoder = get_model_config(vlm_config.image_encoder.vision_encoder)
    vlm_config.image_encoder.vision_projector = get_model_config(vlm_config.image_encoder.vision_projector)


def _configure_audio_encoder(vlm_config):
    """Configure audio encoder module."""
    if get_args().hetero_parallel:
        hetero_align_config(vlm_config.audio_encoder.audio_encoder, vlm_config.audio_encoder)

    vlm_config.audio_encoder.audio_encoder = get_model_config(vlm_config.audio_encoder.audio_encoder)


def _configure_text_decoder(vlm_config):
    """Configure text decoder module."""
    if get_args().hetero_parallel:
        hetero_align_config(vlm_config.text_decoder, vlm_config.text_decoder)

    vlm_config.text_decoder = get_model_config(vlm_config.text_decoder)


def _apply_freezing(model, vlm_config):
    """Apply freezing settings to the model."""
    has_image = hasattr(vlm_config, 'image_encoder') and vlm_config.image_encoder is not None
    freeze_image_encoder = has_image and getattr(vlm_config.image_encoder.vision_encoder, 'freeze', True)
    freeze_image_projection = has_image and getattr(vlm_config.image_encoder.vision_projector, 'freeze', False)

    has_audio = hasattr(vlm_config, 'audio_encoder') and vlm_config.audio_encoder is not None
    freeze_audio_encoder = has_audio and getattr(vlm_config.audio_encoder.audio_encoder, 'freeze', True)

    model.freeze(
        freeze_image_encoder=freeze_image_encoder,
        freeze_image_projection=freeze_image_projection,
        freeze_audio_encoder=freeze_audio_encoder
    )


def move_to_device(batch: Dict[str, Any], float_dtype: str):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            dtype = float_dtype if torch.is_floating_point(v) else None
            batch[k] = v.to(device=torch.cuda.current_device(), dtype=dtype)
        elif isinstance(v, list) and all(isinstance(t, torch.Tensor) for t in v):
            batch[k] = [t.to(device=torch.cuda.current_device(),
                             dtype=float_dtype if torch.is_floating_point(t) else None)
                        for t in v]


def get_batch(data_iterator, is_vit_last_stage=False):
    """Generate a batch."""
    if data_iterator is not None:
        batch = next(data_iterator)
    else:
        raise ValueError("Data iterator is None. Unable to retrieve batch.")
    move_to_device(batch, get_args().params_dtype)
    has_video = 'pixel_values_videos' in batch and 'video_grid_thw' in batch
    if has_video:
        batch['pixel_values'] = batch.pop('pixel_values_videos')
        batch['image_grid_thw'] = batch.pop('video_grid_thw')
    if (mpu.is_pipeline_first_stage() or is_vit_last_stage) and get_args().encoder_dp_balance:
        batch['pixel_values'], batch['tranfer'] = EncoderBalanceComm.apply(
            batch['pixel_values'],
            mpu.get_data_parallel_group())
    else:
        batch['tranfer'] = None
    return batch


def get_tps(output_tensor):
    """Get the tokens per sample"""
    B, S, _ = output_tensor.shape
    dp_size = torch.distributed.get_world_size(group=mpu.get_data_parallel_group())
    cp_size = torch.distributed.get_world_size(group=mpu.get_context_parallel_group())
    tokens_per_sample = torch.tensor(S, device=output_tensor.device) / dp_size * cp_size
    torch.distributed.all_reduce(tokens_per_sample, group=mpu.get_data_parallel_group())
    return tokens_per_sample


def loss_func(output_tensor):
    """Loss function."""
    args = get_args()
    loss_dict = output_tensor['loss_dict']

    loss_dir = {}
    if args.log_tps:
        tokens_per_sample = get_tps(output_tensor['logits'])
        loss_dir["tokens per sample"] = tokens_per_sample

    if args.calculate_per_token_loss:
        loss, local_num_tokens, reporting_loss = compute_token_level_loss(loss_dict)
        loss_dir["loss"] = (reporting_loss[0], reporting_loss[1])
        return (
            loss[0].clone(),
            local_num_tokens,
            loss_dir
        )

    loss = loss_dict['loss']
    averaged_loss = average_losses_across_data_parallel_group([loss])
    loss_dir["loss"] = averaged_loss[0]
    loss = loss.unsqueeze(0).clone()
    return loss / mpu.get_context_parallel_world_size(), loss_dir


def forward_step(data_iterator, model):
    """Forward step."""
    is_vit_last_stage = False
    if model.module.module.add_image_encoder:
        is_vit_last_stage = model.module.module.image_encoder.post_process
    output_tensor = model(**get_batch(data_iterator, is_vit_last_stage))
    return output_tensor, loss_func


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    data_config = args.mm.data
    if args.hetero_parallel:
        print_rank_0("change parallel state for data loader ...")
        change_parallel_state("text_decoder")

        if args.hetero_encoder_mbs_scale > 1:
            pp_mbs = args.micro_batch_size
            args.micro_batch_size = pp_mbs * args.hetero_encoder_mbs_scale

    datasets = build_mm_dataset(data_config.dataset_param)
    build_dataloader = partial(
        build_mm_dataloader,
        dataloader_param=data_config.dataloader_param,
        process_group=mpu.get_data_parallel_group(),
        dataset_param=data_config.dataset_param,
        consumed_samples=args.consumed_train_samples
    )

    micro_batch_size = args.micro_batch_size
    if args.use_data_balance:
        global_batch_size = args.micro_batch_size * get_num_microbatches()
        if args.hetero_encoder_mbs_scale > 1:
            global_batch_size = global_batch_size // args.hetero_encoder_mbs_scale
        args.micro_batch_size = global_batch_size

    if isinstance(datasets, tuple) and len(datasets) == 2:
        train_dataset, valid_dataset = datasets
        train_dataloader = build_dataloader(train_dataset)
        args.micro_batch_size = micro_batch_size
        valid_dataloader = build_dataloader(valid_dataset)
        train_dataloader, valid_dataloader, test_dataloader = build_iterations(train_dataloader, valid_dataloader)
    else:
        train_dataset = datasets
        val_rate = getattr(data_config.dataset_param.basic_parameters, 'val_rate', 0.0)
        if not (0.0 <= val_rate <= 1.0):
            raise ValueError(f'val_rate must be between 0.0 and 1.0, got {val_rate}')
        if isinstance(train_dataset, Dataset) and val_rate > 0:
            dataset = train_dataset.train_test_split(test_size=val_rate, seed=args.seed)
            train_dataset, valid_dataset = dataset['train'], dataset['test']
            train_dataloader = build_dataloader(train_dataset)
            args.micro_batch_size = micro_batch_size
            valid_dataloader = build_dataloader(valid_dataset)
            train_dataloader, valid_dataloader, test_dataloader = build_iterations(train_dataloader, valid_dataloader)
        else:
            train_dataloader = build_dataloader(train_dataset)
            args.micro_batch_size = micro_batch_size
            train_dataloader, valid_dataloader, test_dataloader = build_iterations(train_dataloader)

    if args.hetero_parallel and args.hetero_encoder_mbs_scale > 1:
        args.micro_batch_size = pp_mbs

    return train_dataloader, valid_dataloader, test_dataloader


if __name__ == "__main__":
    from mindspeed_mm.patchs import ring_attn_patch, ulysses_patches, torch_dcp_patch
    import gc
    # set gc threshold to mitigate performance fluctuation
    gc.set_threshold(700, 10, 1000)
    train_valid_test_datasets_provider.is_distributed = True
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=mm_extra_args_provider,
        args_defaults={"dataloader_type": "external"},
    )
