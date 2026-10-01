from megatron.core.transformer.transformer_config import TransformerConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.models.omni.mllms.bagel_qwen2_mot import Qwen2ForCausalLM


class ModelConfigManager:
    """
    ModelConfigManager - Central registry for managing model classes and configurations.

    Provides a unified interface to map model identifiers to their corresponding
    implementation classes and configuration objects.
    """
    MODEL_REGISTRY = {
        "qwen2MoT": {
            "decoder_class": Qwen2ForCausalLM,  # Model implementation class
            "config_class": Qwen2Config(),  # Model configuration object
        }
    }

    def __init__(self, model_id):
        self.model_id = model_id

    def get_model_class(self):
        return self.MODEL_REGISTRY[self.model_id]["decoder_class"]

    def get_config_class(self):
        return self.MODEL_REGISTRY[self.model_id]["config_class"]

    def process_config(self, origin_config):
        """Transfer configuration attributes from source to target configuration.

        Copies all attributes from the original configuration to the target
        configuration class, enabling customization while preserving structure.

        Args:
            origin_config: Source configuration object with custom settings

        Returns:
            object: Target configuration with transferred attributes
        """
        config_dict = vars(origin_config)  # Convert to dictionary
        target_config = self.get_config_class()
        for key, value in config_dict.items():
            setattr(target_config, key, value)

        return target_config


class MllmModel(MultiModalModule):
    """
    MllmModel - Main multi-modal model builder.

    Responsible for constructing multi-modal models by integrating various
    components (vision encoder, autoencoder) with a base language model.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config=config)
        self.config = config
        self.model_id = getattr(self.config.mllm, "model_id")  # Extract model identifier
        self.config_manager = ModelConfigManager(self.model_id)
        self.model_config = self._build_model_config()  # Build final configuration
        self.mllm = self._initialize_mllm()  # Initialize the multi-modal LLM

    def _merge_config(self, base_config):
        if self.config.image_encoder:
            base_config.image_encoder = self.config.image_encoder

        if self.config.ae:
            base_config.ae = self.config.ae

        return base_config

    def _build_model_config(self):
        target_config = self.config_manager.process_config(self.config.mllm)
        merged_config = self._merge_config(target_config)
        return merged_config

    def _initialize_mllm(self):
        model_class = self.config_manager.get_model_class()
        return model_class(self.model_config)

    def get_model(self):
        return self.mllm
