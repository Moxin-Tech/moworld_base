import copy
import torchvision.transforms as transforms

from mindspeed_mm.data.data_utils.data_transform import (
    AENorm,
    CenterCropArr,
    CenterCropResizeVideo,
    LongSideResizeVideo,
    ResizeVideo,
    RandomHorizontalFlipVideo,
    ResizeCrop,
    RandSizeCropArr,
    RandomSizedCrop,
    ResizeCropToFill,
    SpatialStrideCropVideo,
    ToTensorVideo,
    ToTensorAfterResize,
    UCFCenterCropVideo,
    Expand2Square,
    JpegDegradationSimulator,
    MaxHWResizeVideo,
    CenterCropVideo,
    AffineVideo,
    ResizeCropVideo,
    ResizeToFill
)

VIDEO_TRANSFORM_MAPPING = {
    "ToTensorVideo": ToTensorVideo,
    "ToTensorAfterResize": ToTensorAfterResize,
    "RandomHorizontalFlipVideo": RandomHorizontalFlipVideo,
    "UCFCenterCropVideo": UCFCenterCropVideo,
    "ResizeCrop": ResizeCrop,
    "RandSizeCropArr": RandSizeCropArr,
    "RandomSizedCrop": RandomSizedCrop,
    "CenterCropResizeVideo": CenterCropResizeVideo,
    "LongSideResizeVideo": LongSideResizeVideo,
    "ResizeVideo": ResizeVideo,
    "SpatialStrideCropVideo": SpatialStrideCropVideo,
    "norm_fun": transforms.Normalize,
    "ae_norm": AENorm,
    "MaxHWResizeVideo": MaxHWResizeVideo,
    "Resize": transforms.Resize,
    "CenterCropVideo": CenterCropVideo,
    "AffineVideo": AffineVideo,
    "ResizeCropVideo": ResizeCropVideo,
    "ResizeToFill": ResizeToFill
}


IMAGE_TRANSFORM_MAPPING = {
    "Lambda": transforms.Lambda,
    "ToTensorVideo": ToTensorVideo,
    "CenterCropResizeVideo": CenterCropResizeVideo,
    "CenterCropArr": CenterCropArr,
    "ResizeCropToFill": ResizeCropToFill,
    "RandomHorizontalFlip": transforms.RandomHorizontalFlip,
    "ToTensor": transforms.ToTensor,
    "norm_fun": transforms.Normalize,
    "ae_norm": AENorm,
    "DataAugment": JpegDegradationSimulator,
    "Pad2Square": Expand2Square,
    "Resize": transforms.Resize,
    "ResizeToFill": ResizeToFill,
}


INTERPOLATIONMODE_MAPPING = {
    "BICUBIC": transforms.InterpolationMode.BICUBIC,
    "BILINEAR": transforms.InterpolationMode.BILINEAR,
    "NEAREST": transforms.InterpolationMode.NEAREST,
    "NEAREST_EXACT": transforms.InterpolationMode.NEAREST_EXACT,
    "BOX": transforms.InterpolationMode.BOX,
    "HAMMING": transforms.InterpolationMode.HAMMING,
    "LANCZOS": transforms.InterpolationMode.LANCZOS,
}


def get_transforms(is_video=True, train_pipeline=None, image_size=None, transform_size=None):
    if train_pipeline is None:
        return None
    train_pipeline_info = (
        copy.deepcopy(train_pipeline.get("video", list()))
        if is_video
        else copy.deepcopy(train_pipeline.get("image", list()))
    )
    pipeline = []
    for pp_in in train_pipeline_info:
        param_info = pp_in.get("param", dict())

        # For dynamic datasets, users must specify the image_size parameter and apply transforms with the given value
        if image_size and "size" in param_info and param_info["size"] == "auto":
            param_info["size"] = image_size
        elif transform_size and param_info.get("transform_size", None) == "auto":
            param_info["transform_size"] = transform_size
        trans_type = pp_in.get("trans_type", "")
        trans_info = TransformMapping(
            is_video=is_video, trans_type=trans_type, param=param_info
        ).get_trans_func()
        pipeline.append(trans_info)
    output_transforms = transforms.Compose(pipeline)
    return output_transforms


class TransformMapping:
    """used for transforms mapping"""

    def __init__(self, is_video=True, trans_type="", param=None):
        self.is_video = is_video
        self.trans_type = trans_type
        self.param = param if param is not None else dict()

    def get_trans_func(self):
        if self.is_video:
            if self.trans_type in VIDEO_TRANSFORM_MAPPING:
                transforms_cls = VIDEO_TRANSFORM_MAPPING[self.trans_type]
                return transforms_cls(**self.param)
            else:
                raise NotImplementedError(
                    f"Unsupported video transform type: {self.trans_type}"
                )
        else:
            if self.trans_type in IMAGE_TRANSFORM_MAPPING:
                transforms_cls = IMAGE_TRANSFORM_MAPPING[self.trans_type]
                if self.trans_type == "Resize" and "interpolation" in self.param:
                    self.param["interpolation"] = INTERPOLATIONMODE_MAPPING[self.param["interpolation"]]
                return transforms_cls(**self.param)
            else:
                raise NotImplementedError(
                    f"Unsupported image transform type: {self.trans_type}"
                )
