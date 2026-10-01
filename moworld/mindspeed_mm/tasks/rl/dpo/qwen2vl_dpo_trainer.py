# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
from copy import deepcopy
from functools import partial

import torch

from megatron.core.enums import ModelType
from megatron.training import get_args, print_rank_0
from megatron.training.checkpointing import load_checkpoint
from megatron.training.global_vars import set_args
from megatron.training.training import get_model
from mindspeed_mm.models.qwen2vl_model import Qwen2VLModel
from mindspeed_mm.tasks.finetune.lora.utils import is_enable_lora
from mindspeed_mm.tasks.rl.dpo.dpo_trainer import DPOTrainer
from mindspeed_mm.tasks.rl.dpo.qwen2vl_dpo_model import Qwen2VLDPOModel
from mindspeed_mm.utils.transformer_model_config import get_model_config


class Qwen2VLDPOTrainer(DPOTrainer):
    """
    A trainer class for Direct Preference Optimization (DPO).

    This class provides methods for model initialize, computing losses and metrics, and training.
    """

    def __init__(
            self,
            train_valid_test_dataset_provider,
            model_type,
            process_non_loss_data_func=None,
            extra_args_provider=None,
            args_defaults=None,
    ):
        """
        Initializes the DPOTrainer instance.

        Sets up the instance variables for the model provider, actual micro batch size,
        and initializes the DPO model.
        """
        super().__init__(
            train_valid_test_dataset_provider,
            model_type,
            process_non_loss_data_func,
            extra_args_provider,
            args_defaults
        )

        self.args.actual_micro_batch_size = self.args.micro_batch_size * 4
        self.hyper_model = Qwen2VLDPOModel(
            self.train_model,
            self._init_reference_model()
        )
        self.disable_dropout()

    def model_provider(self, pre_process=True, post_process=True):
        """Builds the model."""
        args = get_args()
        print_rank_0("building QWen2VL model ...")
        vlm_config = deepcopy(args.mm.model)

        # distinguish model construct stage when pipeline parallel
        vlm_config.pre_process = pre_process
        vlm_config.post_process = post_process

        if vlm_config.image_encoder:
            vlm_config.image_encoder.vision_encoder = get_model_config(vlm_config.image_encoder.vision_encoder)
            vlm_config.image_encoder.vision_projector = get_model_config(vlm_config.image_encoder.vision_projector)
            vlm_config.text_decoder = get_model_config(vlm_config.text_decoder)

            model = Qwen2VLModel(vlm_config)

            model.freeze(freeze_image_encoder=getattr(vlm_config.image_encoder.vision_encoder, 'freeze', True),
                         freeze_image_projection=getattr(vlm_config.image_encoder.vision_projector, 'freeze', True))
        else:
            vlm_config.text_decoder = get_model_config(vlm_config.text_decoder)
            model = Qwen2VLModel(vlm_config)

        return model

    def disable_dropout(self):
        """
        disable dropout
        """
        args_ = get_args()
        args_.attention_dropout = 0.0
        args_.hidden_dropout = 0.0
        args_.retro_encoder_hidden_dropout = 0.0
        args_.retro_encoder_attention_dropout = 0.0
        set_args(args_)

    @staticmethod
    def get_batch(data_iterator):
        """Generate a batch."""
        if data_iterator is not None:
            batch = next(data_iterator)
        else:
            raise ValueError("Data iterator is None. Unable to retrieve batch.")
        input_ids = batch['input_ids'].to(torch.cuda.current_device())
        labels = batch['labels'].to(torch.cuda.current_device())
        attention_mask = batch['attention_mask'].to(torch.cuda.current_device())
        has_image = 'pixel_values' in batch and 'image_grid_thw' in batch
        has_video = 'pixel_values_videos' in batch and 'video_grid_thw' in batch
        if has_image or has_video:
            if has_image:
                pixel_values = batch['pixel_values'].to(torch.cuda.current_device())
                image_grid_thw = batch['image_grid_thw'].to(torch.cuda.current_device())
            if has_video:
                pixel_values = batch['pixel_values_videos'].to(torch.cuda.current_device())
                image_grid_thw = batch['video_grid_thw'].to(torch.cuda.current_device())
        else:  # Text only
            pixel_values = None
            image_grid_thw = None
        batch = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'image_grid_thw': image_grid_thw
        }
        return batch['input_ids'], batch['labels'], batch['attention_mask'], batch['pixel_values'], batch['image_grid_thw']

    def forward_step(self, data_iterator, model):
        """DPO Forward training step.

        Args:
            data_iterator : Input data iterator
            model : vlm model
        """
        # Get the batch.
        input_ids, labels, attention_mask, pixel_values, image_grid_thw = self.get_batch(data_iterator)

        output_tensor = self.hyper_model(input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                              attention_mask=attention_mask, labels=labels)

        return output_tensor, partial(self.loss_func, labels)

    def _init_reference_model(self):
        """
        Initializes the reference model frozen.

        Returns:
            The initialized reference model.
        """
        model = get_model(self.model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=False)

        self.args.load = self.args.ref_model if self.args.ref_model is not None else self.args.load
        if self.args.load:
            if is_enable_lora():
                strict = False
            else:
                strict = True
            # to avoid assert error
            consumed_train_samples = self.args.consumed_train_samples
            self.args.consumed_train_samples = 0
            args_ = get_args()
            if not args_.finetune:
                args_.is_load_refer = True
                args_.no_load_rng = True
                args_.no_load_optim = True
                set_args(args_)
            load_checkpoint(model, None, None, 'load', strict=strict)
            self.args.consumed_train_samples = consumed_train_samples

        return [model[k].eval() for k in range(len(model))]
