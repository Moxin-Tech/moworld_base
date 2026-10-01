# Copyright (c) 2024 Huawei Technologies Co., Ltd.


import os
import random
from typing import Union, List, Optional
import warnings
import copy
import sys

import torch
import numpy as np
from megatron.core import mpu

from mindspeed_mm.data.data_utils.utils import map_target_fps
from mindspeed_mm.data.data_utils.constants import (
    CAPTIONS,
    FILE_INFO,
    FILE_REJECTED_INFO,
    PROMPT_IDS,
    PROMPT_MASK,
    TEXT,
    VIDEO,
    VIDEO_REJECTED,
    IMG_FPS,
    VIDEO_MASK,
    SCORE,
    SCORE_REJECTED,
    SORA_MODEL_PROTECTED_KEYS
)
from mindspeed_mm.data.data_utils.utils import (
    ImageProcesser,
    TextProcesser
)
from mindspeed_mm.data.data_utils.video_reader import VideoReader
from mindspeed_mm.data.data_utils.video_processor import VideoProcessor
from mindspeed_mm.data.datasets.mm_base_dataset import MMBaseDataset
from mindspeed_mm.models.text_encoder import Tokenizer
from mindspeed_mm.data.data_utils.data_transform import (
    MaskGenerator,
    add_aesthetic_notice_image,
    add_aesthetic_notice_video
)


T2VOutputData = {
    VIDEO: [],
    TEXT: [],
    PROMPT_IDS: [],
    PROMPT_MASK: []
}


class T2VDataset(MMBaseDataset):
    """
    A mutilmodal dataset for text-to-video task based on MMBaseDataset

    Args: some parameters from dataset_param_dict in config.
        basic_param(dict): some basic parameters such as data_path, data_folder, etc.
        vid_img_process(dict): some data preprocessing parameters
        use_text_processer(bool): whether text preprocessing
        tokenizer_config(dict or list(dict)): the config of tokenizer or a list of configs for multi tokenizers
        vid_img_fusion_by_splicing(bool):  videos and images are fused by splicing
        use_img_num(int): the number of fused images
        use_img_from_vid(bool): sampling some images from video
    """

    def __init__(
        self,
        basic_param: dict,
        vid_img_process: dict,
        use_text_processer: bool = False,
        enable_text_preprocessing: bool = True,
        text_preprocess_methods: Optional[Union[dict, List[dict]]] = None,
        use_clean_caption: bool = True,
        support_chinese: bool = False,
        tokenizer_config: Optional[Union[dict, List[dict]]] = None,
        vid_img_fusion_by_splicing: bool = False,
        use_img_num: int = 0,
        use_img_from_vid: bool = True,
        **kwargs,
    ):
        super().__init__(**basic_param)
        self.use_text_processer = use_text_processer
        self.enable_text_preprocessing = enable_text_preprocessing
        self.vid_img_fusion_by_splicing = vid_img_fusion_by_splicing
        self.use_img_num = use_img_num
        self.use_img_from_vid = use_img_from_vid

        self.cfg = vid_img_process.pop("cfg", 0.1)
        self.image_processer_type = vid_img_process.pop("image_processer_type", "image2video")
        self.use_aesthetic = vid_img_process.pop("use_aesthetic", False)
        self.video_reader_type = vid_img_process.pop("video_reader_type", "torchvision")
        self.image_reader_type = vid_img_process.pop("image_reader_type", "torchvision")

        # Initialize processing components
        self.video_reader = VideoReader(video_reader_type=self.video_reader_type)
        self.video_processer = VideoProcessor.create(**vid_img_process)

        self.num_frames = vid_img_process.get("num_frames", 16)
        self.max_height = vid_img_process.get("max_height", 480)
        self.max_width = vid_img_process.get("max_width", 640)
        self.max_hxw = vid_img_process.get("max_hxw", None)
        self.min_hxw = vid_img_process.get("min_hxw", None)
        if self.max_hxw is not None and self.min_hxw is None:
            self.min_hxw = self.max_hxw // 4
        self.train_pipeline = vid_img_process.get("train_pipeline", None)
        transform_size = {
            "max_height": self.max_height,
            "max_width": self.max_width,
            "max_hxw": self.max_hxw,
            "min_hxw": self.min_hxw
        }

        # Create image processor with configuration
        self.image_processer = ImageProcesser(
            num_frames=self.num_frames,
            train_pipeline=self.train_pipeline,
            image_reader_type=self.image_reader_type,
            image_processer_type=self.image_processer_type,
            transform_size=transform_size
        )

        # Initialize text processing components if enabled
        if self.use_text_processer and tokenizer_config is not None:
            self.tokenizer = Tokenizer(tokenizer_config).get_tokenizer()
            self.text_processer = TextProcesser(
                tokenizer=self.tokenizer,
                enable_text_preprocessing=self.enable_text_preprocessing,
                text_preprocess_methods=text_preprocess_methods,
                use_clean_caption=use_clean_caption,
                support_chinese=support_chinese,
                cfg=self.cfg,
            )

         # Data validation
        self.data_samples = self.video_processer.select_valid_data(self.data_samples)

    def __getitem__(self, index):
        try:
            data = self.getitem(index)
            return data
        except Exception as e:
            if self.data_storage_mode == "standard":
                path = self.data_samples[index][FILE_INFO]
                print(f"Data {path}: the error is {e}")
            else:
                print(f"the error is {e}")
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))

    def __len__(self):
        return len(self.data_samples)

    def getitem(self, index):
        # init output data
        examples = copy.deepcopy(T2VOutputData)

        if self.data_storage_mode == "standard":
            sample = self.data_samples[index]
            file_path, texts = sample[FILE_INFO], sample[CAPTIONS]
            if self.data_folder:
                file_path = os.path.join(self.data_folder, file_path)

        elif self.data_storage_mode == "combine":
            sample = self.data_samples[index]
            file_path = sample["path"]
            texts = sample["cap"]

        else:
            raise NotImplementedError(
                f"Not support now: data_storage_mode={self.data_storage_mode}."
            )

        # Generic media processing pipeline
        file_type = self.get_type(file_path)

        # Image/video processing
        if file_type == "image":
            video_value = self.image_processer(file_path)
        elif file_type == "video":
            vframes = self.video_reader(file_path)
            video_value = self.video_processer(vframes=vframes, **sample)
            examples["first_frame"] = video_value[:, 0, :, :]
            if self.vid_img_fusion_by_splicing:
                video_value = self.get_vid_img_fusion(video_value)
        examples[VIDEO] = video_value

        # Text processing
        if isinstance(texts, (list, tuple)) and len(texts) > 1:
            texts = random.choice(texts)  # Random selection from multiple options

        # Handle aesthetic scoring
        if self.use_aesthetic:
            aes = sample.get('aesthetic') or sample.get('aes')
            if aes is not None:
                if file_type == "video":
                    texts = add_aesthetic_notice_video(texts, aes)
                elif file_type == "image":
                    texts = add_aesthetic_notice_image(texts, aes)

        # Text tokenization
        if self.use_text_processer:
            prompt_ids, prompt_mask = self.get_text_processer(texts)
            examples[PROMPT_IDS], examples[PROMPT_MASK] = prompt_ids, prompt_mask

        # DPO (Direct Preference Optimization) handling
        if FILE_REJECTED_INFO in sample.keys():
            rejected_video_path = os.path.join(self.data_folder, sample[FILE_REJECTED_INFO])

            rejected_file_type = self.get_type(rejected_video_path)
            if rejected_file_type == "image":
                rejected_video_value = self.image_processer(rejected_video_path)
            elif rejected_file_type == "video":
                rejected_vframes = self.video_reader(rejected_video_path)
                rejected_video_value = self.video_processer(vframes=rejected_vframes, **sample)
                if self.vid_img_fusion_by_splicing:
                    rejected_video_value = self.get_vid_img_fusion(rejected_video_value)

            examples[VIDEO_REJECTED] = rejected_video_value
            examples[SCORE] = sample[SCORE]
            examples[SCORE_REJECTED] = sample[SCORE_REJECTED]

        # for feature extract, trace source file name
        examples[FILE_INFO] = file_path

        return examples

    def get_data_from_feature_data(self, feature_path):
        if feature_path.endswith(".pt"):
            return torch.load(feature_path, map_location=torch.device('cpu'))
        raise NotImplementedError("Not implemented.")

    def get_value_from_vid_or_img(self, path):
        file_type = self.get_type(path)
        if file_type == "video":
            vframes = self.video_reader(path)
            video_value = self.video_processer(vframes=vframes)
        elif file_type == "image":
            video_value = self.image_processer(path)
        return video_value

    def get_vid_img_fusion(self, video_value):
        if self.use_img_num != 0 and self.use_img_from_vid:
            select_image_idx = np.linspace(
                0, self.num_frames - 1, self.use_img_num, dtype=int
            )
            if self.num_frames < self.use_img_num:
                raise AssertionError(
                    "The num_frames must be larger than the use_img_num."
                )
            images = video_value[:, select_image_idx]  # c, num_img, h, w
            video_value = torch.cat(
                [video_value, images], dim=1
            )  # c, num_frame+num_img, h, w
            return video_value
        elif self.use_img_num != 0 and not self.use_img_from_vid:
            raise NotImplementedError("Not support now.")
        else:
            raise NotImplementedError

    def get_text_processer(self, texts):
        prompt_ids, prompt_mask = self.text_processer(texts)
        if self.vid_img_fusion_by_splicing and self.use_img_from_vid:
            if not isinstance(prompt_ids, list):
                prompt_ids = torch.stack(
                    [prompt_ids] * (1 + self.use_img_num)
                )  # 1+self.use_img_num, l
                prompt_mask = torch.stack(
                    [prompt_mask] * (1 + self.use_img_num)
                )  # 1+self.use_img_num, l
            else:
                prompt_ids = [
                    torch.stack([_prompt_ids] * (1 + self.use_img_num))
                    for _prompt_ids in prompt_ids
                ]
                prompt_mask = [
                    torch.stack([_prompt_mask] * (1 + self.use_img_num))
                    for _prompt_mask in prompt_mask
                ]
        if self.vid_img_fusion_by_splicing and not self.use_img_from_vid:
            raise NotImplementedError("Not support now.")

        return (prompt_ids, prompt_mask)


class DynamicVideoTextDataset(MMBaseDataset):
    """
    A mutilmodal dataset for variable text-to-video task based on MMBaseDataset

    Args: some parameters from dataset_param_dict in config.
        basic_param(dict): some basic parameters such as data_path, data_folder, etc.
        vid_img_process(dict): some data preprocessing parameters
        use_text_processer(bool): whether text preprocessing
        tokenizer_config(dict): the config of tokenizer
        vid_img_fusion_by_splicing(bool):  videos and images are fused by splicing
        use_img_num(int): the number of fused images
        use_img_from_vid(bool): sampling some images from video
    """

    def __init__(
        self,
        basic_param: dict,
        vid_img_process: dict,
        use_text_processer: bool = False,
        enable_text_preprocessing: bool = True,
        use_clean_caption: bool = True,
        tokenizer_config: Union[dict, None] = None,
        vid_img_fusion_by_splicing: bool = False,
        use_img_num: int = 0,
        use_img_from_vid: bool = True,
        dummy_text_feature=False,
        text_add_fps: bool = False,
        fps_max: int = sys.maxsize,
        **kwargs,
    ):
        super().__init__(**basic_param)
        self.use_text_processer = use_text_processer
        self.vid_img_fusion_by_splicing = vid_img_fusion_by_splicing
        self.use_img_num = use_img_num
        self.use_img_from_vid = use_img_from_vid

        self.video_processor_type = vid_img_process.get("video_processor_type")
        self.num_frames = vid_img_process.get("num_frames", 16)
        self.frame_interval = vid_img_process.get("frame_interval", 1)
        self.resolution = vid_img_process.get("resolution", (256, 256))

        self.train_pipeline = vid_img_process.get("train_pipeline", None)
        self.video_reader_type = vid_img_process.get("video_reader_type", "torchvision")
        self.image_reader_type = vid_img_process.get("image_reader_type", "torchvision")
        self.video_reader = VideoReader(video_reader_type=self.video_reader_type)
        self.text_add_fps = text_add_fps
        self.fps_max = fps_max

        self.video_processer = VideoProcessor.create(
            video_processor_type=self.video_processor_type,
            num_frames=self.num_frames,
            frame_interval=self.frame_interval,
            train_pipeline=self.train_pipeline,
        )
        self.image_processer = ImageProcesser(
            num_frames=self.num_frames,
            train_pipeline=self.train_pipeline,
            image_reader_type=self.image_reader_type,
        )

        if "video_mask_ratios" in kwargs:
            self.video_mask_generator = MaskGenerator(kwargs["video_mask_ratios"])
        else:
            self.video_mask_generator = None

        if self.use_text_processer and tokenizer_config is not None:
            self.tokenizer = Tokenizer(tokenizer_config).get_tokenizer()
            self.text_processer = TextProcesser(
                tokenizer=self.tokenizer,
                use_clean_caption=use_clean_caption,
                enable_text_preprocessing=enable_text_preprocessing
            )

        self.data_samples["id"] = np.arange(len(self.data_samples))
        self.dummy_text_feature = dummy_text_feature
        self.get_text = "text" in self.data_samples.columns

    def get_data_info(self, index):
        T = self.data.iloc[index]["num_frames"]
        H = self.data.iloc[index]["height"]
        W = self.data.iloc[index]["width"]

    def get_value_from_vid_or_img(self, num_frames, video_or_image_path, image_size, frame_interval):
        file_type = self.get_type(video_or_image_path)

        video_fps = 24  # default fps
        if file_type == "video":
            # loading
            vframes = self.video_reader(video_or_image_path)
            video_fps = vframes.get_video_fps()

            video_fps = video_fps // frame_interval

            video = self.video_processer(vframes, num_frames=num_frames, frame_interval=frame_interval,
                                         image_size=image_size)  # T C H W
        else:
            # loading
            image = pil_loader(video_or_image_path)
            video_fps = IMG_FPS

            # transform
            image = self.image_processer(image)

            # repeat
            video = image.unsqueeze(0)

        return video, video_fps

    def __getitem__(self, index):
        index, num_frames, height, width = [int(val) for val in index.split("-")]
        sample = self.data_samples.iloc[index]
        frame_interval = self.get_frame_interval(sample)
        video_or_image_path = sample["path"]
        if self.data_folder:
            video_or_image_path = os.path.join(self.data_folder, video_or_image_path)

        video, video_fps = self.get_value_from_vid_or_img(num_frames, video_or_image_path, image_size=(height, width), frame_interval=frame_interval)
        ar = height / width

        ret = {
            "video": video,
            "video_mask": None,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "ar": ar,
            "fps": video_fps,
        }

        if self.video_mask_generator is not None:
            ret["video_mask"] = self.video_mask_generator.get_mask(video)

        if self.get_text:
            prompt_ids, prompt_mask = self.get_text_processer(sample["text"])
            ret["prompt_ids"] = prompt_ids
            ret["prompt_mask"] = prompt_mask

        if self.dummy_text_feature:
            text_len = 50
            ret["prompt_ids"] = torch.zeros((1, text_len, 1152))
            ret["prompt_mask"] = text_len
        ret[FILE_INFO] = video_or_image_path
        return ret

    def get_text_processer(self, texts):
        prompt_ids, prompt_mask = self.text_processer(texts)
        return prompt_ids, prompt_mask

    def get_frame_interval(self, sample):
        if self.text_add_fps:
            new_fps, frame_interval = map_target_fps(sample["fps"], self.fps_max)
            if "text" in sample:
                postfixs = []
                if new_fps != 0 and self.fps_max < 999:
                    postfixs.append(f"{new_fps} FPS")
                postfix = " " + ", ".join(postfixs) + "." if postfixs else ""
                sample["text"] = sample["text"] + postfix
        else:
            frame_interval = self.frame_interval
        return frame_interval
