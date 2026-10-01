import torch

from megatron.training import get_args
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from mindspeed_mm.models.common.module import MultiModalModule
from .projectors.deepseekvl_mlp import create_deepseekvl_mlp
from .projectors.internvl_mlp import InternVLMLP
from .projectors.multimodal_projector import MultimodalProjector
from .projectors.glm_mlp import GlmMLP
from .vision_encoders.clip_vit_model import CLIPViT
from .vision_encoders.internvit_model import InternViT
from .vision_encoders.qwen2vl_vit_model import Qwen2VLViT
from .vision_encoders.qwen3vl_vit_model import Qwen3VLViT
from .vision_encoders.siglip_vit_model import create_siglip_vit
from .vision_encoders.glm4v_vl_vit_model import GlmViT

VISION_ENCODER_MAPPINGS = {
    "clip": CLIPViT,
    "InternViT": InternViT,
    "qwen2vit": Qwen2VLViT,
    "qwen2_5_vit": Qwen2VLViT,
    "qwen3vit": Qwen3VLViT,
    "videoalign_vit": Qwen2VLViT,
    "SigLip": create_siglip_vit,
    "glm4v_vit": GlmViT,
}

VISION_PROJECTION_MAPPINGS = {
    "mlp": MultimodalProjector,
    "InternVLMLP": InternVLMLP,
    "lnmlp": MultimodalProjector,
    "DeepSeekVL2MLP": create_deepseekvl_mlp,
    "GlmMLP": GlmMLP,
}


class VisionModel(MultiModalModule):
    """
    Instantiate a vision encoder model from config.

    Args:
        config (dict): the general config for Vision Model
        {
            "vision_encoder": {...},  # Config for the image encoder.
            "vision_projector": {...},  # Config for the image projector.
            "drop_vision_class_token": (bool),  # Drop vision class token(s) before input to the text decoder.
        }
    """

    def __init__(
            self,
            config: TransformerConfig,
            encoder_transformer_layer_spec: ModuleSpec = None,
            projector_layer_spec: ModuleSpec = None,
            pre_process: bool = True,
            post_process: bool = True,
            *args,
            **kwargs
    ) -> None:
        super().__init__(config=config)
        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = config.vision_encoder is not None
        self.add_projector = config.vision_projector is not None and self.post_process
        # When enabling PP, projector exists only on the last card, set to None by default (otherwise freeze is affected).
        self.projector = None
        self.encoder = None
        if self.add_encoder:
            self.encoder = VISION_ENCODER_MAPPINGS[config.vision_encoder.model_id](
                config=config.vision_encoder,
                transformer_layer_spec=encoder_transformer_layer_spec,
                pre_process=self.pre_process,
                post_process=self.post_process,
                projector_config=config.vision_projector
            )
        if self.add_projector:
            self.projector = VISION_PROJECTION_MAPPINGS[config.vision_projector.model_id](
                config=config.vision_projector,
                submodules=projector_layer_spec,
            )

    def set_input_tensor(self, input_tensor):
        self.encoder.set_input_tensor(input_tensor)

    def freeze(
            self,
            freeze_encoder: bool = False,
            freeze_projector: bool = False
    ):
        """
        Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False for the module's parameters.

        Args:
            freeze_encoder (bool): Freeze the image encoder module.
            freeze_projection (bool): Freeze the image projector module.
        """

        modules = []
        if freeze_encoder and self.encoder is not None:
            modules.append(self.encoder)
        if freeze_projector and self.projector is not None:
            modules.append(self.projector)

        for module in modules:
            unfreeze_names = getattr(module, "unfreeze_param_names", [])
            for name, param in module.named_parameters():
                is_unfreeze = any(
                    unfreeze_name in name
                    for unfreeze_name in unfreeze_names
                )
                if is_unfreeze:
                    continue
                param.requires_grad = False

    def forward(self, images: torch.Tensor = None, image_grid_thw: torch.Tensor = None, *args,
                **kwargs) -> torch.Tensor:
        if self.add_encoder:
            encoder_out = self.encoder(pixel_values=images, grid_thw=image_grid_thw, **kwargs)

        deepstack_feature_lists = None
        if isinstance(encoder_out, tuple):
            if len(encoder_out) == 3:
                image_embeddings, window_index, deepstack_feature_lists = encoder_out
            elif len(encoder_out) == 2:
                image_embeddings, window_index = encoder_out
        else:
            image_embeddings, window_index = encoder_out, None
        if self.add_projector:
            image_embeddings = self.projector(image_embeddings)
            if window_index is not None:
                reverse_indices = torch.argsort(window_index)
                image_embeddings = image_embeddings[reverse_indices, :]

        if deepstack_feature_lists:
            return image_embeddings, deepstack_feature_lists
        return image_embeddings


class Qwen2vlVisionModel(VisionModel):
    def forward(self, images: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        encoder_out = self.encoder(images, image_grid_thw)
        if isinstance(encoder_out, tuple):
            image_embeddings, window_index = encoder_out
        else:
            image_embeddings, window_index = encoder_out, None
        if self.add_projector:
            image_embeddings = self.projector(image_embeddings)
            if window_index is not None:
                reverse_indices = torch.argsort(window_index)
                image_embeddings = image_embeddings[reverse_indices, :]

        return image_embeddings
