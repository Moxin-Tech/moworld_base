import os
import copy
import random
from typing import Union

import numpy as np
import torch

from mindspeed_mm.data.data_utils.constants import (
    CAPTIONS,
    FILE_INFO,
    PROMPT_IDS,
    PROMPT_MASK,
    TEXT,
    VIDEO,
    VIDEO_MASK,
)
from mindspeed_mm.data.datasets.i2v_dataset import I2VDataset, I2VOutputData
from mindspeed_mm.data.data_utils.data_transform import (
    add_aesthetic_notice_image,
    add_aesthetic_notice_video,
)


CAMERA_CONTROL_DIRECTION = "camera_control_direction"
CAM_FROM_WORLDS = "cam_from_worlds"
INTRINSICS = "intrinsics"
ORI_H = "ori_h"
ORI_W = "ori_w"
START_FRAME = "start_frame"
END_FRAME = "end_frame"


CameraControlOutputData = dict(I2VOutputData)
CameraControlOutputData[CAMERA_CONTROL_DIRECTION] = []
CameraControlOutputData[CAM_FROM_WORLDS] = []
CameraControlOutputData[INTRINSICS] = []
CameraControlOutputData[ORI_H] = []
CameraControlOutputData[ORI_W] = []
CameraControlOutputData[START_FRAME] = []
CameraControlOutputData[END_FRAME] = []


class CameraControlDataset(I2VDataset):
    """Dataset for MoWorld training with wan2.2-control-camera.

    Extends I2VDataset to additionally load camera extrinsics (cam_from_worlds.npy)
    and intrinsics (intrinsics.npy) from the same directory as the video data.
    """

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
            mask_type_ratio_dict_video=mask_type_ratio_dict_video,
            mask_type_ratio_dict_image=mask_type_ratio_dict_image,
            default_text_ratio=default_text_ratio,
            min_clear_ratio=min_clear_ratio,
            max_clear_ratio=max_clear_ratio,
            **kwargs,
        )

    def getitem(self, index):
        examples = copy.deepcopy(CameraControlOutputData)

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

        # load camera data before video processing (need ori resolution)
        root_dir = file_path if os.path.isdir(file_path) else os.path.dirname(file_path)
        cam_from_worlds_path = os.path.join(root_dir, "cam_from_worlds.npy")
        if not os.path.exists(cam_from_worlds_path):
            cam_from_worlds_path = os.path.join(root_dir, "cam_from_world.npy")
        intrinsics_path = os.path.join(root_dir, "intrinsics.npy")

        cam_from_worlds = None
        intrinsics = None
        if os.path.exists(cam_from_worlds_path):
            cam_from_worlds = np.load(cam_from_worlds_path)
        if os.path.exists(intrinsics_path):
            intrinsics = np.load(intrinsics_path)

        # parse start_frame / end_frame from sample
        start_frame = sample.get("start_frame", 0)
        end_frame = sample.get("end_frame", None)

        # get video or image
        file_type = self.get_type(file_path)
        ori_h, ori_w = None, None
        if file_type == "image":
            video_value = self.image_processer(file_path)
            video_value = video_value.transpose(0, 1)
            transforms_after_resize = self.image_transforms_after_resize
            # capture original resolution before transform
            from PIL import Image
            with Image.open(file_path) as img:
                ori_w, ori_h = img.size
        elif file_type == "video":
            vframes = self.video_reader(file_path)
            video_value = self.video_processer(vframes=vframes, **sample)
            if self.vid_img_fusion_by_splicing:
                video_value = self.get_vid_img_fusion(video_value)
            video_value = video_value.permute(1, 0, 2, 3)
            transforms_after_resize = self.video_transforms_after_resize
            # try to get original resolution from metadata
            ori_h = sample.get("ori_h", None)
            ori_w = sample.get("ori_w", None)
            if ori_h is None or ori_w is None:
                # fallback: try first frame
                try:
                    from decord import VideoReader
                    vr = VideoReader(file_path)
                    first_frame = vr[0].asnumpy()
                    ori_h, ori_w = first_frame.shape[:2]
                except Exception:
                    pass

        # compute end_frame if not provided
        if end_frame is None:
            num_frames = self.num_frames
            if file_type == "video" and hasattr(video_value, 'shape'):
                # T C H W -> num temporal frames
                num_frames = video_value.shape[0]
            end_frame = start_frame + num_frames

        inpaint_cond_data = self.mask_processor(video_value, mask_type_ratio_dict=self.mask_type_ratio_dict_video)
        mask, masked_video = inpaint_cond_data['mask'], inpaint_cond_data['masked_pixel_values']

        video_value = transforms_after_resize(video_value)
        masked_video = transforms_after_resize(masked_video)

        # Extract first frame (T C H W -> C H W) for I2V processor before concatenation
        examples["first_frame"] = video_value[0]

        video_value = torch.cat([video_value, masked_video, mask], dim=1)
        video_value = video_value.transpose(0, 1)  # T C H W -> C T H W

        examples[VIDEO] = video_value

        # camera data
        if cam_from_worlds is not None:
            examples[CAM_FROM_WORLDS] = torch.from_numpy(cam_from_worlds).float()
        if intrinsics is not None:
            examples[INTRINSICS] = torch.from_numpy(intrinsics).float()
        if ori_h is not None:
            examples[ORI_H] = ori_h
        if ori_w is not None:
            examples[ORI_W] = ori_w
        examples[START_FRAME] = start_frame
        examples[END_FRAME] = end_frame

        # build camera_control_direction from paths (the predictor's control_adapter.process_camera_coordinates
        # will handle loading from these paths)
        if os.path.exists(intrinsics_path) and os.path.exists(cam_from_worlds_path):
            examples[CAMERA_CONTROL_DIRECTION] = [intrinsics_path, cam_from_worlds_path]

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

        examples[FILE_INFO] = file_path

        return examples
