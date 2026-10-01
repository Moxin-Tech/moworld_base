import os
import importlib
import logging

import torch
import torch.distributed as dist
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel
from accelerate import init_empty_weights

from mindspeed.fsdp.utils.str_match import module_name_match
from mindspeed.fsdp.utils.log import print_rank

from mindspeed_mm.fsdp.params.model_args import ModelArguments
from mindspeed_mm.fsdp.params.feature_args import FeatureArguments
from mindspeed_mm.fsdp.params.training_args import TrainingArguments
from mindspeed_mm.fsdp.utils.register import model_register
from mindspeed_mm.fsdp.models.base_model import BaseModel


logger = logging.getLogger(__name__)


class ModelHub:
    """
    Responsible for building HuggingFace native models.
    """
    @staticmethod
    def _build_custom_model(model_args: ModelArguments, training_args: TrainingArguments) -> BaseModel:
        # First try to get model class from custom MODEL_MAPPINGS using model_id
        model_id = getattr(model_args, "model_id", None)
        if model_id:
            model_cls = model_register.get(model_id)
        else:
            raise ValueError("`model_id` must be provided in model_args when using custom models.")

        if model_cls is None:
            raise ValueError(f"model_id '{model_id}' is not registered in MODEL_MAPPINGS. ")

        # Initialize model with meta device for memory efficiency if specified
        if training_args.init_model_with_meta_device:
            with init_empty_weights():
                model = model_cls._from_config(model_args).float()
            for m in model.modules():
                if getattr(m, "_is_hf_initialized", False):
                    m._is_hf_initialized = False
        else:
            # Load model from pretrained weights
            model = model_cls.from_pretrained(model_args).float()

        return model

    @staticmethod
    def _build_transformers_model(transformer_config: PretrainedConfig, model_args: ModelArguments,
        feature_args: FeatureArguments, training_args: TrainingArguments) -> PreTrainedModel:
        # Get model architecture from config
        architectures = getattr(transformer_config, "architectures", [])
        model_cls = None

        # First try to get model class from custom MODEL_MAPPINGS using model_id
        model_id = getattr(model_args, "model_id", None)
        if model_id:
            model_cls = model_register.get(model_id)
        # If not found in mappings, try to get from transformers module using architecture name
        elif architectures:
            transformers_module = importlib.import_module("transformers")
            model_cls = getattr(transformers_module, architectures[0], None)

        if model_cls is None:
            raise ValueError("load model from config failed")

        # overwrite transformer config with model_args and feature_args
        if callable(getattr(model_cls, 'overwrite_transformer_config', None)):
            transformer_config = model_cls.overwrite_transformer_config(transformer_config, model_args, feature_args)


        # Initialize model with meta device for memory efficiency if specified
        if training_args.init_model_with_meta_device:
            with init_empty_weights():
                model = model_cls._from_config(transformer_config).float()
            for m in model.modules():
                if getattr(m, "_is_hf_initialized", False):
                    m._is_hf_initialized = False
        else:
            # Load model from pretrained weights
            model = model_cls.from_pretrained(
                model_args.model_name_or_path,
                config=transformer_config,
                dtype=torch.float32,
                low_cpu_mem_usage=True,
                device_map="cpu",
                trust_remote_code=model_args.trust_remote_code
            )

        return model

    @staticmethod
    def build(model_args: ModelArguments, feature_args: FeatureArguments, training_args: TrainingArguments):
        """
        Build a model instance from HuggingFace based on model arguments and training configuration.

        Args:
            model_args: Contains model_name_or_path, trust_remote_code, etc.
            training_args: Contains training configuration like init_model_with_meta_device, etc.

        Returns:
            Configured model instance ready for training.
        """
        try:
            # Load HuggingFace Config
            print_rank(logger.info, f"> Loading AutoConfig from {model_args.model_name_or_path}...")
            transformer_config = AutoConfig.from_pretrained(
                model_args.model_name_or_path,
                trust_remote_code=model_args.trust_remote_code,
                _attn_implementation=model_args.attn_implementation
            )
        except Exception as e:
            # If config loading fails, treat as custom model
            first_line = next(iter(str(e).splitlines()), "")
            _MAX_ERROR_MSG_LEN = 200
            msg = first_line[:_MAX_ERROR_MSG_LEN] + "..." if len(first_line) > _MAX_ERROR_MSG_LEN else first_line
            logger.warning(
                f"AutoConfig.from_pretrained failed for '{model_args.model_name_or_path}' "
                f"({type(e).__name__}: {msg}); falling back to custom model builder. "
                f"If you intended to load a HuggingFace model, check the error above."
            )
            transformer_config = None

        # Determine which builder to use based on config availability
        if transformer_config:
            print_rank(logger.info, f"Building transformers model from configuration...")
            model: PreTrainedModel = ModelHub._build_transformers_model(transformer_config, model_args, feature_args,
                                                                        training_args)
        else:
            print_rank(logger.info, f"Building custom model...")
            model: BaseModel = ModelHub._build_custom_model(model_args, training_args)

        # Apply parameter freezing if specified
        freezed_named_modules = []
        if len(model_args.freeze) > 0:
            for name, module in model.named_modules():
                for pattern in model_args.freeze:
                    if module_name_match(pattern, name):
                        freezed_named_modules.append((name, module))
            for name, module in freezed_named_modules:
                print_rank(logger.info, f"freezing module {name}...")
                for param in module.parameters():
                    param.requires_grad_(False)

        return model
