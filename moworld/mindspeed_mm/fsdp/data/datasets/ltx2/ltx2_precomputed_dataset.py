from __future__ import annotations

import importlib
import logging
import os
import random
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import Dataset

if "ltx_core" not in sys.modules:
    sys.modules["ltx_core"] = importlib.import_module("mindspeed_mm.fsdp.models.ltx2.ltx_core")

from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier, get_pixel_coords
from ltx_core.types import AudioLatentShape, SpatioTemporalScaleFactors, VideoLatentShape

from mindspeed_mm.fsdp.utils.register import data_register

logger = logging.getLogger(__name__)


@dataclass
class LTX2PrecomputedBasicArgs:
    dataset_dir: str = "/data/ltx2"
    latents_dir: str = "latents"
    conditions_dir: str = "conditions"
    max_samples: int | None = None
    fps: float = 24.0
    first_frame_conditioning_p: float = 0.0
    with_audio: bool = False
    audio_latents_dir: str = "audio_latents"
    audio_channels: int = 8
    audio_mel_bins: int = 16
    audio_prompt_key: str = "audio_prompt_embeds"
    # Align with upstream ltx-trainer where default is shifted_logit_normal.
    timestep_sampling_mode: Literal["uniform", "shifted_logit_normal"] = "shifted_logit_normal"
    timestep_sampling_params: dict[str, float] = field(default_factory=dict)


@data_register.register("ltx2_precomputed")
class LTX2PrecomputedDataset(Dataset):
    """Load LTX2 precomputed latents/conditions and emit training-ready velocity targets.

    Expected layout:
      {root}/.precomputed/latents/*.pt
      {root}/.precomputed/conditions/*.pt
    """

    def __init__(self, basic_param, preprocess_param=None, dataset_param=None, **kwargs):
        _ = (preprocess_param, kwargs)
        resolved_basic_param = self._resolve_basic_param(basic_param, dataset_param)
        self.args = LTX2PrecomputedBasicArgs(**resolved_basic_param)
        self._video_patchifier = VideoLatentPatchifier(patch_size=1)
        self._audio_patchifier = AudioPatchifier(patch_size=1)
        self._video_scale_factors = SpatioTemporalScaleFactors.default()

        root = Path(self.args.dataset_dir).expanduser().resolve()
        if (root / ".precomputed").is_dir():
            root = root / ".precomputed"
        self.latents_root = root / self.args.latents_dir
        self.conditions_root = root / self.args.conditions_dir
        self.audio_latents_root = root / self.args.audio_latents_dir if self.args.with_audio else None
        if not self.latents_root.is_dir():
            raise FileNotFoundError(f"Latents directory not found: {self.latents_root}")
        if not self.conditions_root.is_dir():
            raise FileNotFoundError(f"Conditions directory not found: {self.conditions_root}")
        if self.args.with_audio and (self.audio_latents_root is None or not self.audio_latents_root.is_dir()):
            raise FileNotFoundError(f"Audio latents directory not found: {self.audio_latents_root}")

        latent_files = sorted(self.latents_root.rglob("*.pt"))
        if not latent_files:
            raise ValueError(f"No latent .pt files found under: {self.latents_root}")

        pairs: list[tuple[Path, Path, Path | None]] = []
        for latent_path in latent_files:
            rel = latent_path.relative_to(self.latents_root)
            cond_path = self.conditions_root / rel
            if not cond_path.exists() and latent_path.name.startswith("latent_"):
                cond_path = self.conditions_root / rel.with_name(f"condition_{latent_path.stem[7:]}.pt")
            if not cond_path.exists():
                continue

            audio_latent_path: Path | None = None
            if self.args.with_audio:
                audio_latent_path = self.audio_latents_root / rel
                if not audio_latent_path.exists():
                    continue
            pairs.append((latent_path, cond_path, audio_latent_path))

        if not pairs:
            raise ValueError("No matched (latents, conditions) file pairs were found.")

        if self.args.max_samples is not None:
            pairs = pairs[: self.args.max_samples]
        pairs = self._trim_pairs_for_data_parallel(pairs)
        self.sample_pairs = pairs

    @staticmethod
    def _get_world_size() -> int:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_world_size())
        env_world_size = os.getenv("WORLD_SIZE")
        if env_world_size is not None:
            try:
                return int(env_world_size)
            except ValueError:
                return 1
        return 1

    @classmethod
    def _trim_pairs_for_data_parallel(cls, pairs: list[tuple[Path, Path, Path | None]]) -> list[tuple[Path, Path, Path | None]]:
        """Drop tail samples so per-rank sample counts stay aligned with drop_last=True."""
        world_size = max(cls._get_world_size(), 1)
        if world_size <= 1:
            return pairs
        usable = (len(pairs) // world_size) * world_size
        if usable == 0:
            raise ValueError(
                f"Dataset has {len(pairs)} samples, smaller than world size {world_size}. "
                "Cannot shard evenly for distributed training."
            )
        if usable != len(pairs):
            dropped = len(pairs) - usable
            logger.warning(
                f"Trimming {dropped} tail samples from LTX2 dataset for even DP sharding: {len(pairs)} -> {usable} (world_size={world_size})"
            )
            return pairs[:usable]
        return pairs

    @staticmethod
    def _to_dict(obj: Any) -> dict[str, Any]:
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        return {}

    @classmethod
    def _resolve_basic_param(cls, basic_param: Any, dataset_param: Any) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        dataset_param_dict = cls._to_dict(dataset_param)

        # Preferred location: data.dataset_param.ltx2_dataset_custom
        dataset_custom = dataset_param_dict.get("ltx2_dataset_custom", {})
        if isinstance(dataset_custom, dict):
            merged.update(dataset_custom)

        # Highest priority: basic_parameters
        if isinstance(basic_param, dict):
            merged.update(basic_param)

        valid_keys = {f.name for f in fields(LTX2PrecomputedBasicArgs)}
        resolved = {k: v for k, v in merged.items() if k in valid_keys}
        return resolved

    def __len__(self) -> int:
        return len(self.sample_pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one raw sample in upstream ltx-trainer style.

        The actual training input preparation (noise/timesteps/positions/targets)
        is done in ``prepare_training_inputs`` during the train step.
        """
        latent_path, cond_path, audio_latent_path = self.sample_pairs[idx]
        latent_data = self._safe_torch_load(latent_path)
        cond_data = self._safe_torch_load(cond_path)
        latent_data = self._normalize_video_latents(latent_data)
        sample: dict[str, Any] = {
            "latents": latent_data,
            "conditions": cond_data,
            "idx": idx,
        }
        if self.args.with_audio:
            if audio_latent_path is None:
                raise ValueError("Audio training is enabled but audio latent path is missing.")
            audio_data = self._safe_torch_load(audio_latent_path)
            sample["audio_latents"] = audio_data
        return sample

    # according to ltx-trainer/src/ltx_trainer/timestep_samplers.py
    def _sample_sigmas(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        mode = self.args.timestep_sampling_mode
        params = self.args.timestep_sampling_params or {}
        if mode == "uniform":
            min_value = float(params.get("min_value", 0.0))
            max_value = float(params.get("max_value", 1.0))
            if max_value <= min_value:
                raise ValueError(
                    f"`timestep_sampling_params.max_value` ({max_value}) must be larger than "
                    f"`min_value` ({min_value}) for uniform mode."
                )
            sigmas = torch.rand((batch_size,), device=device, dtype=torch.float32) * (max_value - min_value) + min_value
            return sigmas

        if mode == "shifted_logit_normal":
            std = float(params.get("std", 1.0))
            min_tokens = float(params.get("min_tokens", 1024))
            max_tokens = float(params.get("max_tokens", 4096))
            min_shift = float(params.get("min_shift", 0.95))
            max_shift = float(params.get("max_shift", 2.05))
            if max_tokens <= min_tokens:
                raise ValueError(
                    f"`timestep_sampling_params.max_tokens` ({max_tokens}) must be larger than "
                    f"`min_tokens` ({min_tokens}) for shifted_logit_normal mode."
                )
            m = (max_shift - min_shift) / (max_tokens - min_tokens)
            b = min_shift - m * min_tokens
            shift = m * float(seq_len) + b
            sigmas = torch.sigmoid(torch.randn((batch_size,), device=device, dtype=torch.float32) * std + shift)
            return sigmas

        raise ValueError(
            f"Unsupported timestep_sampling_mode: {mode}. "
            f"Expected one of ['uniform', 'shifted_logit_normal']."
        )

    def _prepare_base_inputs(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Prepare deterministic inputs; stochastic noise/timestep sampling happens in train step."""
        def _first_scalar(x: Any) -> float:
            if isinstance(x, torch.Tensor):
                if x.numel() == 0:
                    raise ValueError("Expected non-empty tensor for scalar metadata.")
                return float(x.reshape(-1)[0].item())
            if isinstance(x, (list, tuple)):
                if len(x) == 0:
                    raise ValueError("Expected non-empty list/tuple for scalar metadata.")
                return _first_scalar(x[0])
            if isinstance(x, (int, float)):
                return float(x)
            raise TypeError(f"Unsupported scalar metadata type: {type(x).__name__}")

        latents = batch["latents"]
        video_latents = latents["latents"]
        # [B, C, F, H, W] -> [B, seq_len, C] with patch_size=1
        video_latents = self._video_patchifier.patchify(video_latents)

        num_frames = int(_first_scalar(latents["num_frames"]))
        height = int(_first_scalar(latents["height"]))
        width = int(_first_scalar(latents["width"]))
        fps_tensor = latents.get("fps", None)
        fps = float(_first_scalar(fps_tensor)) if fps_tensor is not None else float(self.args.fps)

        batch_size = video_latents.shape[0]
        dtype = video_latents.dtype
        device = video_latents.device

        video_positions = self._build_video_positions(
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
            dtype=dtype,
            batch_size=batch_size,
            device=device,
        )
        prepared: dict[str, torch.Tensor] = {
            "video_latent_clean": video_latents,
            "video_positions": video_positions,
            "video_height": torch.tensor(height, dtype=torch.int64, device=device),
            "video_width": torch.tensor(width, dtype=torch.int64, device=device),
        }
        prepared.update(self._extract_condition_inputs(batch["conditions"]))
        if "idx" in batch:
            idx_data = batch["idx"]
            if isinstance(idx_data, torch.Tensor):
                prepared["sample_idx"] = idx_data.to(device=device, dtype=torch.int64)
            elif isinstance(idx_data, list):
                prepared["sample_idx"] = torch.tensor(idx_data, dtype=torch.int64, device=device)

        if self.args.with_audio:
            if "audio_latents" not in batch:
                raise ValueError("Audio training is enabled but `audio_latents` is missing in batch.")
            audio_latents = self._extract_audio_latents(batch["audio_latents"]).to(dtype)
            audio_positions = self._build_audio_positions(
                num_steps=audio_latents.shape[1],
                dtype=dtype,
                batch_size=batch_size,
                device=device,
            )

            prepared.update(
                {
                    "audio_latent_clean": audio_latents,
                    "audio_positions": audio_positions,
                }
            )

        return prepared

    def _prepare_training_inputs(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Prepare stochastic model inputs in upstream ltx-trainer style at train step time.

        Expects batch to already have been processed by _prepare_base_inputs (i.e., contains
        'video_latent_clean', 'video_positions', 'context', 'context_mask', etc.).
        """
        video_latents = batch["video_latent_clean"]
        batch_size = video_latents.shape[0]
        seq_len = video_latents.shape[1]
        device = video_latents.device

        sigmas = self._sample_sigmas(batch_size, seq_len, device=device)
        video_noise = torch.randn_like(video_latents)
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_video = (1.0 - sigmas_expanded) * video_latents + sigmas_expanded * video_noise
        video_targets = video_noise - video_latents

        height = int(batch["video_height"].reshape(-1)[0].item())
        width = int(batch["video_width"].reshape(-1)[0].item())
        conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=seq_len,
            height=height,
            width=width,
            device=device,
            first_frame_conditioning_p=self.args.first_frame_conditioning_p,
        )
        noisy_video = torch.where(conditioning_mask.unsqueeze(-1), video_latents, noisy_video)
        video_timesteps = self._create_per_token_timesteps(conditioning_mask, sigmas)
        video_loss_mask = ~conditioning_mask

        prepared: dict[str, torch.Tensor] = {
            "video_latent": noisy_video,
            "video_target_velocity": video_targets,
            "video_timesteps": video_timesteps,
            "video_positions": batch["video_positions"],
            "context": batch["context"],
            "context_mask": batch["context_mask"],
            "loss_mask": video_loss_mask,
        }
        if "sample_idx" in batch:
            prepared["sample_idx"] = batch["sample_idx"]

        if self.args.with_audio and "audio_latent_clean" in batch:
            audio_latents = batch["audio_latent_clean"]
            audio_seq_len = audio_latents.shape[1]
            audio_noise = torch.randn_like(audio_latents)
            noisy_audio = (1.0 - sigmas_expanded) * audio_latents + sigmas_expanded * audio_noise
            audio_targets = audio_noise - audio_latents
            # `expand` creates a view with shared storage (stride 0) which may break DataLoader pin_memory
            audio_timesteps = sigmas.view(-1, 1).expand(-1, audio_seq_len).clone()
            prepared.update(
                {
                    "audio_latent": noisy_audio,
                    "audio_target_velocity": audio_targets,
                    "audio_timesteps": audio_timesteps,
                    "audio_positions": batch["audio_positions"],
                }
            )

        return prepared

    @staticmethod
    def _create_per_token_timesteps(conditioning_mask: torch.Tensor, sampled_sigma: torch.Tensor) -> torch.Tensor:
        # Same rationale as above: avoid returning an expanded view that can crash pin_memory.
        expanded_sigma = sampled_sigma.view(-1, 1).expand_as(conditioning_mask).clone()
        return torch.where(conditioning_mask, torch.zeros_like(expanded_sigma), expanded_sigma)

    @staticmethod
    def _create_first_frame_conditioning_mask(
        batch_size: int,
        sequence_length: int,
        height: int,
        width: int,
        device: torch.device,
        first_frame_conditioning_p: float = 0.0,
    ) -> torch.Tensor:
        conditioning_mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool, device=device)
        first_frame_end_idx = height * width
        if first_frame_conditioning_p > 0 and random.random() < first_frame_conditioning_p:
            if first_frame_end_idx < sequence_length:
                conditioning_mask[:, :first_frame_end_idx] = True
        return conditioning_mask

    def collate_fn(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        def _collate_values(values: list[Any], key_path: str) -> Any:
            first = values[0]
            if isinstance(first, torch.Tensor):
                if any((not isinstance(v, torch.Tensor)) or (v.shape != first.shape) for v in values):
                    raise ValueError(f"Shape mismatch for `{key_path}`")
                return torch.stack(values, dim=0)
            if isinstance(first, dict):
                out: dict[str, Any] = {}
                for k in first.keys():
                    out[k] = _collate_values([v[k] for v in values], f"{key_path}.{k}" if key_path else k)
                return out
            if isinstance(first, bool):
                return torch.tensor(values, dtype=torch.bool)
            if isinstance(first, int):
                return torch.tensor(values, dtype=torch.int64)
            if isinstance(first, float):
                return torch.tensor(values, dtype=torch.float32)
            return values

        raw_batch = _collate_values(features, "")

        # Two-stage preparation to mirror upstream ltx-trainer:
        # - Upstream `PrecomputedDataset` yields raw {"latents","conditions"}.
        # - Upstream `TextToVideoStrategy.prepare_training_inputs(...)` then patchifies latents, samples sigmas,
        #   adds noise, builds per-token timesteps, targets, masks, and positions.
        # Here we split the same logic into:
        #   1) `prepare_base_inputs`: deterministic parts (patchify + positions + context/context_mask).
        #   2) `prepare_training_inputs`: stochastic parts (sigma/noise/targets/masks) for the actual train step.
        base_inputs = self._prepare_base_inputs(raw_batch)
        return self._prepare_training_inputs(base_inputs)

    @staticmethod
    def _safe_torch_load(path: Path) -> dict:
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            return torch.load(path, map_location="cpu")

    @staticmethod
    def _normalize_video_latents(latent_data: dict[str, Any]) -> dict[str, Any]:
        data = dict(latent_data)
        latents = data["latents"]
        if latents.dim() == 2:
            num_frames = int(data["num_frames"])
            height = int(data["height"])
            width = int(data["width"])
            channels = latents.shape[-1]
            latents = latents.reshape(num_frames, height, width, channels).permute(3, 0, 1, 2).contiguous()
            data["latents"] = latents
        elif latents.dim() != 4:
            raise ValueError(f"Unsupported latent shape: {tuple(latents.shape)}")
        return data

    def _build_video_positions(
        self,
        num_frames: int,
        height: int,
        width: int,
        fps: float,
        dtype: torch.dtype,
        batch_size: int = 1,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        latent_coords = self._video_patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape(
                frames=num_frames,
                height=height,
                width=width,
                batch=batch_size,
                channels=128,
            ),
            device=device,
        )
        pixel_coords = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=self._video_scale_factors,
            causal_fix=True,
        ).to(dtype)
        pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / max(fps, 1e-6)
        return pixel_coords

    def _extract_audio_latents(self, audio_data: dict[str, Any]) -> torch.Tensor:
        latents = audio_data["latents"]
        if latents.dim() == 2:
            if latents.shape[-1] != self.args.audio_channels * self.args.audio_mel_bins:
                raise ValueError(
                    "2D audio latents expected shape [T, C*F], "
                    f"got last_dim={latents.shape[-1]}, expected {self.args.audio_channels * self.args.audio_mel_bins}."
                )
            return latents.unsqueeze(0)
        if latents.dim() == 3:
            if latents.shape[0] == self.args.audio_channels and latents.shape[-1] == self.args.audio_mel_bins:
                # [C, T, F] -> [1, T, C*F]
                return self._audio_patchifier.patchify(latents.unsqueeze(0))
            if latents.shape[-1] == self.args.audio_channels * self.args.audio_mel_bins:
                # Already patchified format [B, T, C*F]
                return latents
            raise ValueError(
                "3D audio latents expected either [C, T, F] or [B, T, C*F], "
                f"got shape: {tuple(latents.shape)}."
            )
        if latents.dim() == 4:
            # [B, C, T, F] -> [B, T, C*F]
            return self._audio_patchifier.patchify(latents)
        raise ValueError(f"Unsupported audio latent shape: {tuple(latents.shape)}")

    def _build_audio_positions(
        self,
        num_steps: int,
        dtype: torch.dtype,
        batch_size: int = 1,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        latent_coords = self._audio_patchifier.get_patch_grid_bounds(
            output_shape=AudioLatentShape(
                frames=num_steps,
                mel_bins=self.args.audio_mel_bins,
                batch=batch_size,
                channels=self.args.audio_channels,
            ),
            device=device,
        )
        return latent_coords.to(dtype)

    @staticmethod
    def _normalize_context(context: torch.Tensor) -> torch.Tensor:
        if context.dim() == 2:
            context = context.unsqueeze(0)
        elif context.dim() != 3:
            raise ValueError(f"Unsupported context shape: {tuple(context.shape)}")
        return context.to(torch.float32)

    @staticmethod
    def _normalize_context_mask(context: torch.Tensor, context_mask: torch.Tensor | None) -> torch.Tensor:
        if context_mask is None:
            context_mask = torch.ones(context.shape[:2], dtype=torch.bool, device=context.device)
        if context_mask.dim() == 1:
            context_mask = context_mask.unsqueeze(0)
        elif context_mask.dim() != 2:
            raise ValueError(f"Unsupported context mask shape: {tuple(context_mask.shape)}")
        return context_mask

    def _extract_condition_inputs(self, cond_data: dict[str, Any]) -> dict[str, torch.Tensor]:
        if "prompt_embeds" not in cond_data:
            raise KeyError("Condition file must contain `prompt_embeds` for LTX2 training.")

        context = self._normalize_context(cond_data["prompt_embeds"])
        context_mask = self._normalize_context_mask(context, cond_data.get("prompt_attention_mask"))
        return {
            "context": context,
            "context_mask": context_mask,
        }
