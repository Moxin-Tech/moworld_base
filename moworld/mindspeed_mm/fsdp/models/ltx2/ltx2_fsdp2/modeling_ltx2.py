from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib
import sys
import types
from typing import Any

import torch
import torch_npu
if "ltx_core" not in sys.modules:
    sys.modules["ltx_core"] = importlib.import_module("mindspeed_mm.fsdp.models.ltx2.ltx_core")

from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
from ltx_core.model.transformer.model import LTXModel
from ltx_core.model.transformer.model_configurator import (
    LTXModelConfigurator,
    LTXV_MODEL_COMFY_RENAMING_MAP,
)
from ltx_core.model.transformer.modality import Modality
from ltx_core.text_encoders.gemma.encoders.av_encoder import (
    AV_GEMMA_TEXT_ENCODER_KEY_OPS,
    AVGemmaTextEncoderModelConfigurator,
    GEMMA_MODEL_OPS,
)
from ltx_core.text_encoders.gemma.encoders.base_encoder import module_ops_from_gemma_root
from ltx_core.utils import find_matching_file
from mindspeed_mm.fsdp.models.base_model import BaseModel
from mindspeed_mm.fsdp.params.model_args import ModelArguments
from mindspeed_mm.fsdp.utils.register import model_register
from mindspeed_mm.fsdp.models.ltx2.ltx2_fsdp2.modified import _process_transformer_blocks
from mindspeed_mm.fsdp.models.ltx2.ltx2_fsdp2.modified import block_forward
from mindspeed_mm.fsdp.models.ltx2.ltx2_fsdp2.modified import model_forward

@dataclass
class LTX2ModelOutput:
    loss: torch.Tensor
    video_pred: torch.Tensor | None = None
    audio_pred: torch.Tensor | None = None

@model_register.register("ltx2")
class LTX2ForTraining(torch.nn.Module, BaseModel):
    """FSDP2 training wrapper for LTX2 transformer.

    This wrapper adapts LTX2's native `(video, audio) -> (video_pred, audio_pred)` interface
    to MindSpeed-MM's expectation that `model(**batch)` returns an object with `.loss`.
    """

    def __init__(self, transformer: LTXModel, text_encoder: torch.nn.Module | None = None):
        super().__init__()
        self.transformer = transformer
        self.__dict__["_text_encoder_runtime"] = text_encoder
        if text_encoder is not None:
            text_encoder.requires_grad_(False)
            text_encoder.eval()
        # Monkey patch the transformer to use the modified methods to use recompute while using FSDP2 training
        self.transformer._process_transformer_blocks = types.MethodType(_process_transformer_blocks, self.transformer)
        for block in self.transformer.transformer_blocks:
            block.forward = types.MethodType(block_forward, block)
        self.transformer.forward = types.MethodType(model_forward, self.transformer)


    @property
    def text_encoder(self) -> torch.nn.Module | None:
        return self.__dict__.get("_text_encoder_runtime")

    @staticmethod
    def _to_mapping(obj: Any) -> dict[str, Any]:
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "__dict__"):
            return {
                key: value
                for key, value in vars(obj).items()
                if not key.startswith("_")
            }
        return {}

    @classmethod
    def _build_transformer_from_config(cls, model_args: ModelArguments) -> LTXModel:
        transformer_cfg = cls._to_mapping(getattr(model_args, "transformer", {}))
        return LTXModelConfigurator.from_config({"transformer": transformer_cfg})

    @classmethod
    def _from_config(cls, config: ModelArguments) -> "LTX2ForTraining":
        model = cls._build_transformer_from_config(config)
        if bool(getattr(config, "enable_gradient_checkpointing", False)):
            model.set_gradient_checkpointing(True)
        return cls(transformer=model)

    @classmethod
    def from_pretrained(cls, config: ModelArguments) -> "LTX2ForTraining":
        ckpt_path = getattr(config, "checkpoint_path", None) or config.model_name_or_path
        if ckpt_path is None:
            raise ValueError("`model_name_or_path` or `checkpoint_path` must be provided for LTX2.")

        ckpt_path = str(Path(ckpt_path).expanduser())
        transformer = SingleGPUModelBuilder(
            model_path=ckpt_path,
            model_class_configurator=LTXModelConfigurator,
            model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
        ).build(device=torch.device("cpu"), dtype=torch.float32)

        if bool(getattr(config, "enable_gradient_checkpointing", False)):
            transformer.set_gradient_checkpointing(True)

        text_encoder = cls._build_text_encoder(config, ckpt_path)
        return cls(transformer=transformer, text_encoder=text_encoder)

    @classmethod
    def _build_text_encoder(cls, config: ModelArguments, checkpoint_path: str) -> torch.nn.Module | None:
        text_encoder_path = getattr(config, "text_encoder_path", None)
        if not text_encoder_path:
            return None

        text_encoder_root = Path(text_encoder_path).expanduser()
        if not text_encoder_root.is_dir():
            raise ValueError(f"`text_encoder_path` is not a directory: {text_encoder_path}")

        gemma_model_folder = find_matching_file(str(text_encoder_root), "model*.safetensors").parent
        gemma_weight_paths = [str(p) for p in gemma_model_folder.rglob("*.safetensors")]
        text_encoder = SingleGPUModelBuilder(
            model_path=(str(checkpoint_path), *gemma_weight_paths),
            model_class_configurator=AVGemmaTextEncoderModelConfigurator,
            model_sd_ops=AV_GEMMA_TEXT_ENCODER_KEY_OPS,
            module_ops=(GEMMA_MODEL_OPS, *module_ops_from_gemma_root(str(text_encoder_root))),
        ).build(device=torch.device("cpu"), dtype=torch.bfloat16)
        # Match upstream trainer: keep only lightweight embedding connectors.
        text_encoder.model = None
        text_encoder.tokenizer = None
        text_encoder.feature_extractor_linear = None
        return text_encoder

    def _ensure_text_encoder_device(self, device: torch.device) -> None:
        if self.text_encoder is None:
            return
        first_param = next(self.text_encoder.parameters(), None)
        if first_param is None:
            return
        if first_param.device != device or first_param.dtype != torch.bfloat16:
            self.text_encoder.to(device=device, dtype=torch.bfloat16)

    def forward(  # noqa: PLR0913
        self,
        video_latent: torch.Tensor,
        video_timesteps: torch.Tensor,
        video_positions: torch.Tensor,
        context: torch.Tensor,
        video_target_velocity: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        audio_latent: torch.Tensor | None = None,
        audio_timesteps: torch.Tensor | None = None,
        audio_positions: torch.Tensor | None = None,
        audio_target_velocity: torch.Tensor | None = None,
        **_: Any,
    ) -> LTX2ModelOutput:
        audio_context = None
        audio_context_mask = None

        if self.text_encoder is not None:
            if context_mask is None:
                raise ValueError("`context_mask` is required when text connector is enabled.")
            self._ensure_text_encoder_device(device=context.device)
            with torch.no_grad():
                video_context, audio_context, connector_mask = self.text_encoder._run_connectors(
                    context,
                    context_mask,
                )
            context = video_context
            context_mask = connector_mask
            audio_context_mask = context_mask

        video = Modality(
            latent=video_latent,
            timesteps=video_timesteps,
            positions=video_positions,
            context=context,
            context_mask=context_mask,
            enabled=True,
        )

        audio = None
        if audio_latent is not None:
            if audio_timesteps is None or audio_positions is None:
                raise ValueError("`audio_timesteps` and `audio_positions` are required when `audio_latent` is provided.")
            audio = Modality(
                latent=audio_latent,
                timesteps=audio_timesteps,
                positions=audio_positions,
                context=audio_context if audio_context is not None else context,
                context_mask=audio_context_mask if audio_context_mask is not None else context_mask,
                enabled=True,
            )

        video_pred, audio_pred = self.transformer(video=video, audio=audio, perturbations=None)
        video_pred = video_pred.float()
        if audio_pred is not None:
            audio_pred = audio_pred.float()

        video_loss = self._masked_mse(video_pred, video_target_velocity, loss_mask)
        audio_loss = None
        if audio_pred is not None and audio_target_velocity is not None:
            audio_loss = (audio_pred.float() - audio_target_velocity.float()).pow(2).mean()
            loss = video_loss + audio_loss
        else:
            loss = video_loss
        loss = loss.float()

        return LTX2ModelOutput(loss=loss, video_pred=video_pred, audio_pred=audio_pred)

    @staticmethod
    def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        diff = (pred - target).pow(2)
        if mask is None:
            return diff.mean()

        mask_f = mask.unsqueeze(-1).float()
        loss = diff.mul(mask_f).div(mask_f.mean())
        return loss.mean()
