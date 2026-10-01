from typing import Dict, Any, Tuple, List

import torch
import mindspeed.megatron_adaptor
from megatron.training import get_args, print_rank_0

from mindspeed_mm.tools.feature_extraction.get_sora_feature import FeatureExtractor
from mindspeed_mm.data.data_utils.constants import (
    FILE_INFO,
    PROMPT_IDS,
    PROMPT_MASK,
    VIDEO,
    VIDEO_MASK,
)
from mindspeed_mm.utils.utils import get_dtype


class HunyuanFeatureExtractor(FeatureExtractor):
    def _extract_single(
        self,
        batch: Dict[str, Any]
    ) -> Tuple[List[str], torch.Tensor, Dict[str, Any], Any, Any]:
        if not batch:
            raise ValueError("Received empty batch")

        video = batch.pop(VIDEO).to(self.device, dtype=self.ae_dtype)
        prompt_ids = batch.pop(PROMPT_IDS)
        prompt_mask = batch.pop(PROMPT_MASK)
        file_names = batch.pop(FILE_INFO)

        # extract feature
        kwargs = {}
        latents, latents_dict = self.vae.encode(video, **batch)

        if latents_dict is not None:
            kwargs.update(latents_dict)

        prompt, prompt_mask = self.text_encoder.encode(prompt_ids, prompt_mask, **kwargs)

        return file_names, latents, latents_dict, prompt, prompt_mask


if __name__ == "__main__":
    # Initialize and run feature extraction
    print_rank_0("Starting feature extraction process")
    extractor = HunyuanFeatureExtractor()
    extractor.extract_all()
    print_rank_0("Feature extraction completed successfully")
