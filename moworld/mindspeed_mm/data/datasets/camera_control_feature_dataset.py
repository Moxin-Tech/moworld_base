import os

import numpy as np
import torch

from mindspeed_mm.data.datasets.mm_base_dataset import MMBaseDataset
from mindspeed_mm.data.data_utils.constants import (
    FILE_INFO,
    PROMPT_IDS,
    PROMPT_MASK,
    VIDEO,
    CAMERA_CONTROL_DIRECTION,
    CAM_FROM_WORLDS,
    INTRINSICS,
    ORI_H,
    ORI_W,
    START_FRAME,
    END_FRAME,
)


def _squeeze_leading_dim(tensor: torch.Tensor, key: str) -> torch.Tensor:
    """Remove DiffSynth's leading batch dim (=1) from a tensor."""
    if key in _SQUEEZE_IF_5D and tensor.ndim == 5 and tensor.shape[0] == 1:
        return tensor.squeeze(0)
    if key in _SQUEEZE_IF_3D:
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            return tensor.squeeze(0)
        if tensor.ndim == 2 and tensor.shape[0] == 1:
            return tensor.squeeze(0)
    return tensor


def camera_control_feature_collate_fn(batch):
    """Stack each encoder's tensor features across samples.

    Each encoder contributes one output tensor with a leading batch dimension.
    For ordinary equal-length lists of compatible tensors, this matches
    PyTorch default_collate. Other fields are passed to default_collate.
    """
    # Separate list-valued keys from normal Tensor/scalar keys
    list_keys = {
        k for k in batch[0]
        if isinstance(batch[0][k], (list, tuple))
        and all(isinstance(v, torch.Tensor) for v in batch[0][k])
    }

    # Build a sub-batch without the list keys and delegate to default_collate
    sub_batch = [{k: v for k, v in sample.items() if k not in list_keys} for sample in batch]
    result = torch.utils.data.default_collate(sub_batch)

    # Manually collate list[Tensor] keys: stack per-encoder across batch
    for key in list_keys:
        num_encoders = len(batch[0][key])
        stacked = []
        for enc_idx in range(num_encoders):
            stacked.append(torch.stack([sample[key][enc_idx] for sample in batch]))
        result[key] = stacked

    return result

# Mapping from DiffSynth .pth key names to MM framework key names
_DIFFSYNTH_TO_MM_KEY_MAP = {
    "input_latents": VIDEO,
    "context": PROMPT_IDS,
    "prompt_ids": PROMPT_IDS,
    "prompt_mask": PROMPT_MASK,
    "y": "i2v_vae_feature",
    "camera_control_direction": CAMERA_CONTROL_DIRECTION,
    "cam_from_worlds": CAM_FROM_WORLDS,
    "intrinsics": INTRINSICS,
    "ori_h": ORI_H,
    "ori_w": ORI_W,
    "start_frame": START_FRAME,
    "end_frame": END_FRAME,
}

# Keys from .pth that should be dropped (not used in MM training loop)
_DIFFSYNTH_DROP_KEYS = {
    "input_video", "latents", "noise", "height", "width", "num_frames",
    "cfg_scale", "tiled", "tile_size", "tile_stride", "scenario",
    "rand_device", "use_gradient_checkpointing", "use_gradient_checkpointing_offload",
    "cfg_merge", "vace_scale", "positive",
}

# Keys whose 5D tensors need squeeze(0) to remove the DiffSynth batch dim
_SQUEEZE_IF_5D = {VIDEO, "i2v_vae_feature", "control_camera_latents_input"}
# Keys whose 3D tensors need squeeze(0) to remove the DiffSynth batch dim
_SQUEEZE_IF_3D = {PROMPT_IDS, PROMPT_MASK, CAM_FROM_WORLDS, INTRINSICS}

# Scalar metadata keys stored as Python ints in .pth — convert to 0-dim tensors
_SCALAR_KEYS = {ORI_H, ORI_W, START_FRAME, END_FRAME}


class CameraControlFeatureDataset(MMBaseDataset):
    """Dataset for loading pre-encoded .pt/.pth feature files for control-camera training.

    Designed for MoWorld training where input data consists
    of pre-encoded VAE latents and text embeddings saved as .pth files from the
    DiffSynth feature extraction pipeline.

    Handles DiffSynth-specific tensor shapes (leading batch dim=1) and key naming
    conventions. Scalar metadata (ori_h, ori_w, start_frame, end_frame) are
    preserved as 0-dim tensors for compatibility with default_collate.
    """

    # Custom collate function to handle list[Tensor] values from .pth files.
    # Access via CameraControlFeatureDataset.collate_fn or dataset.collate_fn.
    collate_fn = staticmethod(camera_control_feature_collate_fn)

    def __init__(self, basic_param: dict):
        super().__init__(**basic_param)

    def __getitem__(self, index: int) -> dict:
        sample = self.data_samples[index]

        feature_file_path = sample[FILE_INFO]
        if self.data_folder:
            feature_file_path = os.path.join(self.data_folder, feature_file_path)

        feature_data = self._load_feature(feature_file_path)
        mapped_data = self._map_keys(feature_data)

        examples = {}
        for key, value in mapped_data.items():
            if isinstance(value, torch.Tensor):
                value = _squeeze_leading_dim(value, key)
                examples[key] = value
            elif isinstance(value, np.ndarray):
                # DiffSynth stores camera extrinsics/intrinsics as NumPy arrays.
                # Convert them here instead of silently dropping camera control.
                examples[key] = torch.from_numpy(value).float()
            elif isinstance(value, (list, tuple)) and all(isinstance(v, torch.Tensor) for v in value):
                # Multi-encoder features (e.g. context = [umt5_embed, clip_embed]).
                # Squeeze each element independently, keep list structure.
                examples[key] = [_squeeze_leading_dim(v, key) for v in value]
            elif key in _SCALAR_KEYS and value is not None:
                # Convert scalar int/float metadata to 0-dim tensor
                examples[key] = torch.tensor(int(value), dtype=torch.long)
            elif key == CAMERA_CONTROL_DIRECTION and value is not None:
                # Keep camera_control_direction as-is (string or list).
                # default_collate handles strings by creating a list across the batch.
                examples[key] = value

        # If .pth does not contain pre-computed control_camera_latents_input,
        # fall back to loading camera metadata from jsonl entry (.npy paths)
        has_precomputed_camera = "control_camera_latents_input" in examples
        if not has_precomputed_camera:
            self._load_camera_metadata(sample, examples)

        return examples

    def _load_camera_metadata(self, sample: dict, examples: dict):
        """Load camera extrinsics/intrinsics from .npy paths in the jsonl entry."""
        cam_path = sample.get("cam_from_worlds_path")
        intrin_path = sample.get("intrinsics_path")

        if cam_path and os.path.exists(cam_path):
            cam_data = np.load(cam_path)
            examples[CAM_FROM_WORLDS] = torch.from_numpy(cam_data).float()
        if intrin_path and os.path.exists(intrin_path):
            intrin_data = np.load(intrin_path)
            examples[INTRINSICS] = torch.from_numpy(intrin_data).float()

        for key in _SCALAR_KEYS:
            if key not in examples:
                val = sample.get(key)
                if val is not None:
                    examples[key] = torch.tensor(int(val), dtype=torch.long)

        camera_dir = sample.get(CAMERA_CONTROL_DIRECTION)
        if camera_dir is not None:
            examples[CAMERA_CONTROL_DIRECTION] = camera_dir

    def _load_feature(self, feature_path: str) -> dict:
        if feature_path.endswith((".pt", ".pth")):
            return torch.load(feature_path, map_location=torch.device('cpu'), weights_only=False)
        raise NotImplementedError(f"Unsupported file format: {feature_path}. Only .pt/.pth files are supported.")

    @staticmethod
    def _map_keys(feature_data: dict) -> dict:
        mapped = {}
        for key, value in feature_data.items():
            if key in _DIFFSYNTH_DROP_KEYS:
                continue
            mm_key = _DIFFSYNTH_TO_MM_KEY_MAP.get(key, key)
            mapped[mm_key] = value
        return mapped

