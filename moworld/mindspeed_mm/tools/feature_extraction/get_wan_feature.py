import copy
import json
import os
import random
import time
from typing import List, Optional, Union

import mindspeed.megatron_adaptor
import torch
import torch.distributed
from megatron.core import mpu
from megatron.training import get_args, print_rank_0
from megatron.training.initialize import initialize_megatron, set_jit_fusion_options
from numpy import save

from mindspeed_mm.configs.config import merge_mm_args, mm_extra_args_provider
from mindspeed_mm.data import build_mm_dataloader, build_mm_dataset
from mindspeed_mm.data.data_utils.constants import (
    FILE_INFO,
    PROMPT_IDS,
    PROMPT_MASK,
    VIDEO,
    VIDEO_MASK,
)
from mindspeed_mm.data.data_utils.transform_pipeline import get_transforms
from mindspeed_mm.data.datasets.t2v_dataset import T2VDataset
from mindspeed_mm.models.ae import AEModel
from mindspeed_mm.models.text_encoder import TextEncoder
from mindspeed_mm.tools.profiler import Profiler
from mindspeed_mm.utils.utils import get_device, get_dtype, is_npu_available
from mindspeed_mm.tools.feature_extraction.get_sora_feature import FeatureExtractor

if is_npu_available():
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    torch.npu.config.allow_internal_format = False


class WanTextVideoDataset(T2VDataset):
    def __init__(
        self,
        task,
        basic_param,
        vid_img_process: dict,
        use_text_processer: bool = False,
        enable_text_preprocessing: bool = True,
        text_preprocess_methods: Optional[Union[dict, List[dict]]] = None,
        tokenizer_config: Optional[Union[dict, List[dict]]] = None,
        **kwargs,
    ):
        video_only_transforms = vid_img_process.get("train_pipeline", {}).get("video_only", None)
        if video_only_transforms is None:
            raise ValueError('"video_only" key not found in vid_img_process["train_pipeline"]')

        video_and_first_frame_transforms = vid_img_process.get("train_pipeline", {}).get("video_and_first_frame", None)
        if video_and_first_frame_transforms is None:
            raise ValueError('"video_and_first_frame" key not found in vid_img_process["train_pipeline"]')

        video_only_preprocess = {"video": video_only_transforms}
        vid_img_process["train_pipeline"] = {"video": video_and_first_frame_transforms}

        super().__init__(
            basic_param=basic_param,
            vid_img_process=vid_img_process,
            use_text_processer=use_text_processer,
            enable_text_preprocessing=enable_text_preprocessing,
            text_preprocess_methods=text_preprocess_methods,
            tokenizer_config=tokenizer_config
        )

        self.video_only_preprocess = get_transforms(
            is_video=True,
            train_pipeline=video_only_preprocess,
            transform_size={"max_height": vid_img_process['max_height'], "max_width": vid_img_process['max_width']}
        )
        self.task = task  # t2v or i2v

    def __getitem__(self, index):
        example = {}
        sample = self.data_samples[index]
        file_path = sample["path"]
        if not os.path.exists(file_path):
            raise AssertionError(f"file {file_path} do not exist!")

        vframes = self.video_reader(file_path)
        video = self.video_processer(vframes=vframes, **sample)

        if self.task == "i2v":
            first_frame = video[:, 0, :, :]  # c t h w
            example["first_frame"] = first_frame

        video = self.video_only_preprocess(video)
        example[VIDEO] = video

        text = sample["cap"]
        if not isinstance(text, list):
            text = [text]
        text = [random.choice(text)]

        if self.use_text_processer:
            prompt_ids, prompt_mask = self.get_text_processer(text)
            example[PROMPT_IDS], example[PROMPT_MASK] = prompt_ids, prompt_mask
        else:
            example["text"] = text
        # for feature extract, trace source file name
        example[FILE_INFO] = file_path
        return example


class WanFeatureExtractor(FeatureExtractor):
    def _prepare_data(self):
        args = get_args()
        task = args.mm.model.task if hasattr(args.mm.model, "task") else "t2v"

        dataset_param = args.mm.data.dataset_param.to_dict()
        dataset = WanTextVideoDataset(
            task,
            dataset_param["basic_parameters"],
            dataset_param["preprocess_parameters"],
            **dataset_param
        )
        dataloader = build_mm_dataloader(
            dataset,
            args.mm.data.dataloader_param,
            process_group=mpu.get_data_parallel_group(),
            dataset_param=args.mm.data.dataset_param,
        )

        return dataset, dataloader


if __name__ == "__main__":
    # Initialize and run feature extraction
    print_rank_0("Starting feature extraction process")
    extractor = WanFeatureExtractor()
    extractor.extract_all()
    print_rank_0("Feature extraction completed successfully")
