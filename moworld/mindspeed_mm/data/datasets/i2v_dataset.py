import os
import random
from typing import Union
import copy

import torch

from mindspeed_mm.data.data_utils.constants import (
    CAPTIONS,
    FILE_INFO,
    IMG_FPS,
    PROMPT_IDS,
    PROMPT_MASK,
    TEXT,
    VIDEO,
    MASKED_VIDEO
)
from mindspeed_mm.data.data_utils.data_transform import (
    MaskGenerator,
    add_aesthetic_notice_image,
    add_aesthetic_notice_video,
)
from mindspeed_mm.data.data_utils.transform_pipeline import get_transforms
from mindspeed_mm.data.datasets.t2v_dataset import T2VDataset
from mindspeed_mm.utils.mask_utils import STR_TO_TYPE, MaskProcessor


def type_ratio_normalize(mask_type_ratio_dict):
    total = sum(mask_type_ratio_dict.values())
    length = len(mask_type_ratio_dict)
    if total == 0:
        return {k: 1.0 / length for k in mask_type_ratio_dict.keys()}
    return {k: v / total for k, v in mask_type_ratio_dict.items()}


I2VOutputData = {
    VIDEO: [],
    MASKED_VIDEO: [],
    TEXT: [],
    PROMPT_IDS: [],
    PROMPT_MASK: [],
}


class I2VDataset(T2VDataset):

    def __init__(
        self,
        basic_param: dict,
        vid_img_process: dict,
        use_text_processer: bool = False,
        use_clean_caption: bool = True,
        support_chinese: bool = False,
        tokenizer_config: Union[dict, None] = None,
        vid_img_fusion_by_splicing: bool = False,
        use_img_num: int = 0,
        use_img_from_vid: bool = True,
        mask_type_ratio_dict_video: Union[dict, None] = None,
        mask_type_ratio_dict_image: Union[dict, None] = None,
        default_text_ratio: float = 0.5,
        min_clear_ratio: float = 0.0,
        max_clear_ratio: float = 1.0,
        **kwargs,
    ):

        if vid_img_process.get("num_frames") != 1:
            self.mask_type_ratio_dict_video = mask_type_ratio_dict_video if mask_type_ratio_dict_video is not None else {
                'i2v': 1.0}
            self.mask_type_ratio_dict_video = {STR_TO_TYPE[k]: v for k, v in self.mask_type_ratio_dict_video.items()}
            self.mask_type_ratio_dict_video = type_ratio_normalize(self.mask_type_ratio_dict_video)

        self.mask_type_ratio_dict_image = mask_type_ratio_dict_image if mask_type_ratio_dict_image is not None else {
            'clear': 1.0}
        self.mask_type_ratio_dict_image = {STR_TO_TYPE[k]: v for k, v in self.mask_type_ratio_dict_image.items()}
        self.mask_type_ratio_dict_image = type_ratio_normalize(self.mask_type_ratio_dict_image)

        self.mask_processor = MaskProcessor(
            max_height=vid_img_process.get("max_height"),
            max_width=vid_img_process.get("max_width"),
            min_clear_ratio=min_clear_ratio,
            max_clear_ratio=max_clear_ratio,
        )

        self.train_pipeline_after_resize = vid_img_process.pop("train_pipeline_after_resize", None)
        self.video_transforms_after_resize = get_transforms(is_video=True, train_pipeline=self.train_pipeline_after_resize) or (lambda x: x)
        self.image_transforms_after_resize = get_transforms(
            is_video=False, train_pipeline=self.train_pipeline_after_resize
        ) or (lambda x: x)

        self.default_text_ratio = default_text_ratio

        # t2v dataset config
        super().__init__(
            basic_param=basic_param,
            vid_img_process=vid_img_process,
            use_text_processer=use_text_processer,
            use_clean_caption=use_clean_caption,
            support_chinese=support_chinese,
            tokenizer_config=tokenizer_config,
            vid_img_fusion_by_splicing=vid_img_fusion_by_splicing,
            use_img_num=use_img_num,
            use_img_from_vid=use_img_from_vid,
            **kwargs,
        )

    def getitem(self, index):
        # init output data
        examples = copy.deepcopy(I2VOutputData)

        if self.data_storage_mode == "combine":
            sample = self.data_samples[index]
            file_path = sample["path"]
            texts = sample["cap"]
        elif self.data_storage_mode == "standard":
            sample = self.data_samples[index]
            file_path, texts = sample[FILE_INFO], sample[CAPTIONS]
            if self.data_folder:
                file_path = os.path.join(self.data_folder, file_path)
        else:
            raise NotImplementedError(
                f"Not support now: data_storage_mode={self.data_storage_mode}."
            )

        # get video or img
        file_type = self.get_type(file_path)
        if file_type == "image":
            video_value = self.image_processer(file_path)
            video_value = video_value.transpose(0, 1)
            transforms_after_resize = self.image_transforms_after_resize
        elif file_type == "video":
            vframes = self.video_reader(file_path)
            video_value = self.video_processer(vframes=vframes, **sample)
            if self.vid_img_fusion_by_splicing:
                video_value = self.get_vid_img_fusion(video_value)
            video_value = video_value.permute(1, 0, 2, 3)
            transforms_after_resize = self.video_transforms_after_resize

        inpaint_cond_data = self.mask_processor(video_value, mask_type_ratio_dict=self.mask_type_ratio_dict_video)
        mask, masked_video = inpaint_cond_data['mask'], inpaint_cond_data['masked_pixel_values']

        video_value = transforms_after_resize(video_value)  # T C H W -> T C H W
        masked_video = transforms_after_resize(masked_video)  # T C H W -> T C H W

        video_value = torch.cat([video_value, masked_video, mask], dim=1)  # T 2C+1 H W

        video_value = video_value.transpose(0, 1)  # T C H W -> C T H W

        examples[VIDEO] = video_value

        # get text tokens
        if (isinstance(texts, list) or isinstance(texts, tuple)) and len(texts) > 1:
            texts = random.choice(texts)

        if self.use_aesthetic:
            if sample.get('aesthetic', None) is not None or sample.get('aes', None) is not None:
                aes = sample.get('aesthetic', None) or sample.get('aes', None)
                if file_type == "video":
                    texts = add_aesthetic_notice_video(texts, aes)
                elif file_type == "image":
                    texts = add_aesthetic_notice_image(texts, aes)

        if self.use_text_processer:
            prompt_ids, prompt_mask = self.get_text_processer(texts)
            examples[PROMPT_IDS], examples[PROMPT_MASK] = (
                prompt_ids,
                prompt_mask,
            )

        # for feature extract, trace source file name
        examples[FILE_INFO] = file_path

        return examples
