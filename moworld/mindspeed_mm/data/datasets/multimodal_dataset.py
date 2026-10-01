# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------


import os
import copy
import random
from typing import Dict, List, Union
import PIL.Image

import torch
from megatron.training import get_args, print_rank_0
from mindspeed_mm.data.data_utils.processing_deepseek_vl_v2 import DeepseekVLV2Processor
from mindspeed_mm.data.data_utils.constants import MODEL_CONSTANTS
from mindspeed_mm.data.data_utils.utils import preprocess
from mindspeed_mm.data.datasets.mm_base_dataset import MMBaseDataset
from mindspeed_mm.models.text_encoder import Tokenizer
from mindspeed_mm.data.data_utils.multimodal_image_video_preprocess import get_multimodal_image_video_preprocessor


class MultiModalChatDataset(MMBaseDataset):
    """
    A multimodal dataset for supervised fine-tuning based on MMBaseDataset.

    Args:
        basic_param (dict): Basic parameters such as data_path, data_folder, etc.
        img_process (dict): some data preprocessing parameters.
        constants (dict): some data preprocessing constants.
        use_text_processer (bool): whether text preprocessing
        tokenizer_config (dict): The config of tokenizer.
        is_multimodal (bool): Flag to indicate if the model is multimodal (handles both text and images).
        mm_use_im_start_end (bool): Flag to indicate if the image start and end tokens should be used.
        template_name (str): The name of the template to be used.
        image_size (int): The size to which images will be resized.
        down_sample_ratio (float): The ratio by which to downsample the images.
        patch_size (int): The size of the patches to be used for processing images.
        group_by_length (bool): Flag to indicate if data should be grouped by length.
        dynamic_image_size (bool): Flag to indicate if the image size should be dynamically adjusted.
        use_thumbnail (bool): Flag to indicate if thumbnails should be used for images.
        min_dynamic_patch (int): The minimum number of dynamic patches.
        max_dynamic_patch (int): The maximum number of dynamic patches.
        repeat_time (float): The number of times to repeat the data processing.
    """

    def __init__(
            self,
            basic_param: dict,
            img_process: dict,
            use_text_processer: bool = False,
            tokenizer_config: Union[dict, None] = None,
            is_multimodal: bool = True,
            mm_use_im_start_end: bool = True,
            template_name: str = "",
            image_size: int = 224,
            down_sample_ratio: float = 0.5,
            patch_size: int = 14,
            group_by_length: bool = False,
            dynamic_image_size: bool = False,
            use_thumbnail: bool = False,
            min_dynamic_patch: int = 1,
            max_dynamic_patch: int = 6,
            min_num_frame: int = 4, # for video data
            max_num_frame: int = 12, # for video data
            sampling_method: str = "rand", # for video data
            repeat_time: float = 1.0,
            **kwargs,
    ):
        super().__init__(**basic_param)
        self.use_text_processer = use_text_processer
        self.template_name = template_name
        self.image_size = image_size
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.min_num_frame = min_num_frame
        self.max_num_frame = max_num_frame
        self.sampling_method = sampling_method
        self.patch_size = patch_size
        self.down_sample_ratio = down_sample_ratio
        self.num_image_token = int((self.image_size // self.patch_size) ** 2 * (self.down_sample_ratio ** 2))

        if repeat_time < 1:
            # If repeat_time is less than 1, select a portion of the data
            self.data_samples = self.data_samples[:int(len(self.data_samples) * repeat_time)]
        if repeat_time > 1:
            # Repeat the list if repeat_time is greater than 1
            self.data_samples = self.data_samples * repeat_time

        self.is_multimodal = is_multimodal
        self.mm_use_im_start_end = mm_use_im_start_end
        self.train_pipeline = img_process.get("train_pipeline", None)
        self.image_reader_type = img_process.get("image_reader_type", "torchvision")

        self.tokenizer = Tokenizer(tokenizer_config).get_tokenizer()
        self.tokenizer.model_max_length = get_args().seq_length
        self.img_video_processor = self._init_image_video_processor()

    def __getitem__(self, index):
        return self.getitem(index)

    def __len__(self):
        return len(self.data_samples)

    def _init_image_video_processor(self):
        return get_multimodal_image_video_preprocessor(
            template_name=self.template_name,
            train_pipeline=self.train_pipeline,
            image_reader_type=self.image_reader_type,
            tokenizer=self.tokenizer,
            dynamic_image_size=self.dynamic_image_size,
            patch_size=self.patch_size,
            image_size=self.image_size,
            min_dynamic_patch=self.min_dynamic_patch,
            max_dynamic_patch=self.max_dynamic_patch,
            use_thumbnail=self.use_thumbnail,
            min_num_frame=self.min_num_frame,
            max_num_frame=self.max_num_frame,
            sampling_method=self.sampling_method
            )

    @staticmethod
    def _init_return_dict():
        return {
            "pixel_values": None,
            "image_flags": None,
            "input_ids": None,
            "labels": None,
            "attention_mask": None
        }

    def _filter_return_dict_keys(self, ret):
        allowed_keys = list(self._init_return_dict().keys())
        keys_to_remove = [key for key in list(ret.keys()) if key not in allowed_keys]
        for key in keys_to_remove:
            ret.pop(key, None)

    def get_path(self, data_path):
        return os.path.join(self.data_folder, data_path)

    def multi_modal_get_item(self, data_item):
        # Ensure the first conversation contains an image placeholder
        if "<image>" not in data_item["conversations"][0]["value"]:
            data_item["conversations"][0]["value"] = "<image>\n" + data_item["conversations"][0]["value"]

        ret = self._init_return_dict()
        image_path = self.get_path(data_item["image"])
        ret_img = self.img_video_processor(image_path=image_path, mode='single_image', num_image=1)
        ret.update(ret_img)
        num_image_patches = ret["pixel_values"].size(0)

        ret_tokenizer = preprocess(
            template_name=self.template_name,
            sources=copy.deepcopy([data_item["conversations"]]),
            tokenizer=self.tokenizer,
            num_image_token_list=[self.num_image_token * num_image_patches],
            group_by_length=self.group_by_length,
            is_multimodal=self.is_multimodal,
            mm_use_im_start_end=self.mm_use_im_start_end
        )
        ret.update(ret_tokenizer)
        ret["image_flags"] = torch.tensor([1] * num_image_patches, dtype=torch.long)
        self._filter_return_dict_keys(ret)

        return ret

    def multi_modal_multi_image_get_item(self, data_item):
        total_pixel_values, num_image_token_list = [], []
        num_images = len(data_item["image"])

        for image_path in data_item["image"]:
            image_path = self.get_path(image_path)

            cur_pixel_values = self.img_video_processor(image_path=image_path, mode='multi_image', num_image=num_images)['pixel_values']
            total_pixel_values += cur_pixel_values
            num_image_token_list.append(self.num_image_token * len(cur_pixel_values))

        total_pixel_values = torch.stack(total_pixel_values)
        num_patches = total_pixel_values.size(0)

        ret = self._init_return_dict()
        ret.update({"pixel_values": total_pixel_values})

        ret_tokenizer = preprocess(
            template_name=self.template_name,
            sources=copy.deepcopy([data_item["conversations"]]),
            tokenizer=self.tokenizer,
            num_image_token_list=num_image_token_list,
            group_by_length=self.group_by_length,
            is_multimodal=self.is_multimodal,
            mm_use_im_start_end=self.mm_use_im_start_end,
            num_image=num_images
        )

        ret.update(ret_tokenizer)
        ret["image_flags"] = torch.tensor([1] * num_patches, dtype=torch.long)
        self._filter_return_dict_keys(ret)

        image_end_token_id = self.tokenizer.convert_tokens_to_ids(MODEL_CONSTANTS[self.template_name]["IMG_END_TOKEN"])
        if (ret["input_ids"] == image_end_token_id).sum() != num_images:
            raise ValueError(f"image tokens are truncated, this dataset is {self.data_path}")

        return ret

    def pure_text_get_item(self, data_item):
        pass

    def video_get_item(self, data_item):
        # Ensure the first conversation contains a video placeholder
        if "<video>" not in data_item["conversations"][0]["value"]:
            data_item["conversations"][0]["value"] = "<video>\n" + data_item["conversations"][0]["value"]

        ret = self._init_return_dict()
        video_path = self.get_path(data_item["video"])
        ret_video = self.img_video_processor(video_path=video_path, clip=data_item.get("clip", None))
        ret.update(ret_video)
        num_image_patches = ret["pixel_values"].size(0)

        # Generate special tokens for each video frame
        special_tokens = "\n".join(["Frame-{}: <image>".format(i + 1) for i in range(len(ret["image_list"]))])
        data_item["conversations"][0]["value"] = data_item["conversations"][0]["value"].replace(
            "<video>\n", special_tokens + "\n")

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token] * num_image_patches
        ret_tokenizer = preprocess(
            self.template_name,
            sources=[copy.deepcopy(data_item["conversations"])],
            tokenizer=self.tokenizer,
            num_image_token_list=num_image_tokens,
            group_by_length=self.group_by_length,
            is_multimodal=self.is_multimodal,
            mm_use_im_start_end=self.mm_use_im_start_end,
            num_image=num_image_patches
        )
        ret.update(ret_tokenizer)
        ret["image_flags"] = torch.tensor([1] * num_image_patches, dtype=torch.long)
        self._filter_return_dict_keys(ret)

        return ret

    def getitem(self, index):
        index = index % len(self.data_samples)
        try_cnt, max_try = 0, 10
        while True:
            if try_cnt == max_try:
                raise InterruptedError(f"MultiModalChatDataset failed to get item after {max_try} times")
            try:
                data_item = copy.deepcopy(self.data_samples[index])
                if "image" in data_item and len(data_item["image"]) != 0:
                    if isinstance(data_item["image"], list):
                        ret = self.multi_modal_multi_image_get_item(data_item)
                    else:
                        ret = self.multi_modal_get_item(data_item)
                elif "video" in data_item and data_item["video"] is not None and data_item["video"] != "":
                    ret = self.video_get_item(data_item)
                else:
                    raise AssertionError(f"Inference data type must be image or video.")
                return ret
            except Exception as e:
                try_cnt += 1
                print_rank_0(f"Error: {e}")
                index = random.randint(0, len(self.data_samples) - 1)


class DeepSeekVLDataset(MMBaseDataset):
    def __init__(
        self,
        basic_param: dict,
        processor_path: str,
        repeat_time: float = 1.0,
        group_by_length: bool = False,
        **kwargs
    ):
        super().__init__(**basic_param)
        self.processor = DeepseekVLV2Processor.from_pretrained(processor_path)
        self.group_by_length = group_by_length

        if repeat_time < 1:
            # If repeat_time is less than 1, select a portion of the data
            self.data_samples = self.data_samples[:int(len(self.data_samples) * repeat_time)]
        if repeat_time > 1:
            # Repeat the list if repeat_time is greater than 1
            self.data_samples = self.data_samples * repeat_time

    def __getitem__(self, index):
        return self.getitem(index)

    def __len__(self):
        return len(self.data_samples)

    def load_pil_images(self, conversations: List[Dict[str, str]]):
        """

        Args:
            conversations (List[Dict[str, str]]): the conversations with a list of messages. An example is :
                [
                    {
                        "role": "User",
                        "content": "<image>\nExtract all information from this image and convert them into markdown format.",
                        "images": ["./examples/table_datasets.png"]
                    },
                    {"role": "Assistant", "content": ""},
                ]

        Returns:
            pil_images (List[PIL.Image.Image]): the list of PIL images.

        """

        pil_images = []

        for message in conversations:
            if "images" not in message:
                continue

            for image_path in message["images"]:
                image_path = os.path.join(self.data_folder, image_path)
                with PIL.Image.open(image_path) as pil_img:
                    pil_img = pil_img.convert("RGB")
                    pil_images.append(pil_img)

        return pil_images

    def multi_modal_get_item(self, data_item):
        conversation = data_item["conversations"]
        pil_images = self.load_pil_images(conversation)

        rets = self.processor.__call__(
            conversations=conversation,
            images=pil_images,
            force_batchify=False, # must set to False for training
            inference_mode=False,
            system_prompt="",
            group_by_length=self.group_by_length,
            max_length=get_args().seq_length
        )

        return {
            "input_ids": rets.input_ids,
            "labels": rets.target_ids,
            "images": rets.images,
            "images_seq_mask": rets.images_seq_mask,
            "images_spatial_crop": rets.images_spatial_crop
        }

    def getitem(self, index):
        index = index % len(self.data_samples)
        try_cnt, max_try = 0, 10
        while True:
            if try_cnt == max_try:
                raise InterruptedError(f"MultiModalChatDataset failed to get item after {max_try} times")
            try:
                data_item = copy.deepcopy(self.data_samples[index])
                ret = self.multi_modal_get_item(data_item)
                return ret
            except Exception as e:
                try_cnt += 1
                print_rank_0(f"Error: {e}")
                index = random.randint(0, len(self.data_samples) - 1)
