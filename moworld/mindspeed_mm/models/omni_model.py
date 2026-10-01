from typing import Dict, Union

import torch
from torch import nn
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args

from mindspeed_mm.models.ae.base import AEModel
from mindspeed_mm.models.omni.mllm_model import MllmModel
from mindspeed_mm.models.vision.vision_model import VisionModel


class OmniModel(nn.Module):
    """
    OmniModel - A unified multi-modal model that can incorporate vision encoder,
    multi-modal large language model (MLLM), and autoencoder components.

    This model supports flexible configurations for different multi-modal tasks,
    allowing selective freezing of components during training.
    """

    def __init__(self, config):
        """Initialize the OmniModel with configuration.

        Args:
            config: Configuration object containing model architecture settings
        """
        super().__init__()

        # Extract transformer configuration from command line arguments
        self.config = core_transformer_config_from_args(get_args())

        # Flags indicating which components to include based on config
        self.add_image_encoder = config.image_encoder is not None
        self.add_mllm = config.mllm is not None
        self.add_ae = config.ae is not None

        # Initialize components based on configuration
        if self.add_image_encoder:
            self.image_encoder = VisionModel(config.image_encoder)
        if self.add_mllm:
            self.mllm = MllmModel(config).get_model()
        if self.add_ae:
            self.ae = AEModel(config.ae).get_model()

    def set_input_tensor(self, input_tensor):
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

    def freeze(
            self,
            freeze_mllm: bool = False,
            freeze_image_encoder: bool = False,
            freeze_image_projection: bool = False,
            freeze_ae: bool = True,
    ):
        if self.add_image_encoder:
            self.image_encoder.freeze(freeze_image_encoder, freeze_image_projection)
        if self.add_mllm and freeze_mllm:
            for param in self.mllm.parameters():
                param.requires_grad = False
        if self.add_ae and freeze_ae:
            for param in self.ae.parameters():
                param.requires_grad = False

    def forward(self, inputs) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """Forward pass through the OmniModel.

        Args:
            inputs: Dictionary containing input tensors for different modalities

        Returns:
            Model outputs, which could be a dictionary of tensors or a single tensor
        """
        # Process image inputs through vision encoder if present and not already processed
        if self.add_image_encoder and "vit_input_embeds" not in inputs:
            vit_input_embeds = self.image_encoder(**inputs)
            inputs["vit_inputs_embeds"] = vit_input_embeds

        # Process inputs through autoencoder if present and not already encoded
        if self.add_ae and "padded_latent" not in inputs:
            padded_latent = self.ae.encode(**inputs)
            inputs["padded_latent"] = padded_latent

        # Pass all processed inputs through the multi-modal language model
        outputs = self.mllm(**inputs)

        return outputs
