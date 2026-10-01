"""Offline raw-video to latent extraction for the MoWorld control-camera model."""

import hashlib
import os

import torch
from megatron.training import print_rank_0

from mindspeed_mm.data.data_utils.constants import (
    CAM_FROM_WORLDS,
    END_FRAME,
    FILE_INFO,
    INTRINSICS,
    ORI_H,
    ORI_W,
    PROMPT_IDS,
    PROMPT_MASK,
    START_FRAME,
    VIDEO,
)
from mindspeed_mm.tools.feature_extraction.get_sora_feature import FeatureExtractor


CAMERA_METADATA_KEYS = (
    CAM_FROM_WORLDS,
    INTRINSICS,
    ORI_H,
    ORI_W,
    START_FRAME,
    END_FRAME,
)


class MoWorldFeatureExtractor(FeatureExtractor):
    """Use native MindSpeed-MM encoders and retain camera metadata in each feature file."""

    @staticmethod
    def _generate_safe_filename(file_path):
        normalized = os.path.abspath(str(file_path))
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
        stem = os.path.basename(normalized).replace(".", "_")
        return f"{stem}_{digest}.pth"

    def _extract_single(self, batch):
        if not batch:
            raise ValueError("Received empty batch")

        video = batch.pop(VIDEO).to(self.device, dtype=self.ae_dtype)
        prompt_ids = batch.pop(PROMPT_IDS)
        prompt_mask = batch.pop(PROMPT_MASK)
        file_names = batch.pop(FILE_INFO)

        encoder_kwargs = {}
        if "first_frame" in batch:
            encoder_kwargs["first_frame"] = batch.pop("first_frame").to(
                self.device, dtype=self.ae_dtype
            )

        metadata = {}
        for key in CAMERA_METADATA_KEYS:
            if key in batch:
                metadata[key] = batch.pop(key)

        latents, latents_dict = self.vae.encode(video, **encoder_kwargs)
        latents_dict = dict(latents_dict or {})
        latents_dict.update(metadata)
        prompt, prompt_mask = self.text_encoder.encode(prompt_ids, prompt_mask)
        return file_names, latents, latents_dict, prompt, prompt_mask


if __name__ == "__main__":
    print_rank_0("Starting MoWorld feature extraction")
    MoWorldFeatureExtractor().extract_all()
    print_rank_0("MoWorld feature extraction completed")
