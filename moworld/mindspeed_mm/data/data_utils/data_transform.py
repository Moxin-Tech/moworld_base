# Modified from Latte: https://github.com/Vchitect/Latte/blob/main/datasets/video_transforms.py

import io
import random
import numbers
import math

import torch
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
import torchvision.transforms as T


def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("Clip should be Tensor, but it is %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("Clip should be 4D, but it is %dD" % clip.dim())

    return True


def center_crop_arr(pil_image, image_size):
    while pil_image.size[0] >= 2 * image_size[0] and pil_image.size[1] >= 2 * image_size[1]:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = max(image_size[0] / pil_image.size[0], image_size[1] / pil_image.size[1])
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size[1]) // 2
    crop_x = (arr.shape[1] - image_size[0]) // 2
    return Image.fromarray(
        arr[crop_y: crop_y + image_size[1], crop_x: crop_x + image_size[0]]
    )


def to_tensor(clip):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
    """
    try:
        _is_tensor_video_clip(clip)
    except Exception as e:
        print(f"An error occurred: {e}")
    if not clip.dtype == torch.uint8:
        raise TypeError(
            "Clip tensor should have data type uint8, but it is %s" % str(clip.dtype)
        )

    return clip.float() / 255.0


def to_tensor_after_resize(clip):
    """
    Convert resized tensor to [0, 1]
    Args:
        clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W), but in [0, 1]
    """
    try:
        _is_tensor_video_clip(clip)
    except Exception as e:
        print(f"An error occurred: {e}")
    return clip.float() / 255.0


def hflip(clip):
    """
    Args:
        clip (torch.tensor): Video clip to be normalized. Size is (T, C, H, W)
    Returns:
        flipped clip (torch.tensor): Size is (T, C, H, W)
    """
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    return clip.flip(-1)


def crop(clip, i, j, h, w):
    """
    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
    """
    if len(clip.size()) != 4:
        raise ValueError("clip should be a 4D tensor")
    return clip[..., i: i + h, j: j + w]


def resize(clip, target_size, interpolation_mode, align_corners=False, antialias=False):
    if len(target_size) != 2:
        raise ValueError(
            f"target size should be tuple (height, width), instead got {target_size}"
        )
    return torch.nn.functional.interpolate(
        clip,
        size=target_size,
        mode=interpolation_mode,
        align_corners=align_corners,
        antialias=antialias,
    )


def resize_scale(clip, target_size, interpolation_mode):
    if len(target_size) != 2:
        raise ValueError(
            f"target size should be tuple (height, width), instead got {target_size}"
        )
    H, W = clip.size(-2), clip.size(-1)
    scale = target_size[0] / min(H, W)
    return torch.nn.functional.interpolate(
        clip, scale_factor=scale, mode=interpolation_mode, align_corners=False
    )


def center_crop(clip, crop_size):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    h, w = clip.size(-2), clip.size(-1)
    th, tw = crop_size
    if h < th or w < tw:
        raise ValueError("height and width must be no smaller than crop_size")

    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    return crop(clip, i, j, th, tw)


def center_crop_using_short_edge(clip):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    h, w = clip.size(-2), clip.size(-1)
    if h < w:
        th, tw = h, h
        i = 0
        j = int(round((w - tw) / 2.0))
    else:
        th, tw = w, w
        i = int(round((h - th) / 2.0))
        j = 0
    return crop(clip, i, j, th, tw)


def center_crop_th_tw(clip, th, tw, top_crop):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")

    h, w = clip.size(-2), clip.size(-1)
    tr = th / tw
    if h / w > tr:
        new_h = int(w * tr)
        new_w = w
    else:
        new_h = h
        new_w = int(h / tr)

    i = 0 if top_crop else int(round((h - new_h) / 2.0))
    j = int(round((w - new_w) / 2.0))
    return crop(clip, i, j, new_h, new_w)


def resize_crop_to_fill(clip, target_size, interpolate_parameters: dict = None):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    if interpolate_parameters is None:
        interpolate_parameters = {}
    if "interpolation_mode" not in interpolate_parameters:
        interpolate_parameters["interpolation_mode"] = "bilinear"
    h, w = clip.size(-2), clip.size(-1)
    th, tw = target_size[0], target_size[1]
    rh, rw = th / h, tw / w
    if rh > rw:
        sh, sw = th, round(w * rh)
        clip = resize(clip, (sh, sw), **interpolate_parameters)
        i = 0
        j = int(round(sw - tw) / 2.0)
    else:
        sh, sw = round(h * rw), tw
        clip = resize(clip, (sh, sw), **interpolate_parameters)
        i = int(round(sh - th) / 2.0)
        j = 0
    if i + th > clip.size(-2) or j + tw > clip.size(-1):
        raise AssertionError("size mismatch.")
    return crop(clip, i, j, th, tw)


def rand_size_crop_arr(pil_image, image_size):
    """
    Randomly crop image for height and width, ranging from image_size[0] to image_size[1]
    """
    arr = np.array(pil_image)

    # get random target h w
    height = random.randint(image_size[0], image_size[1])
    width = random.randint(image_size[0], image_size[1])
    # ensure that h w are factors of 8
    height = height - height % 8
    width = width - width % 8

    # get random start pos
    h_start = random.randint(0, max(len(arr) - height, 0))
    w_start = random.randint(0, max(len(arr[0]) - height, 0))

    # crop
    return Image.fromarray(arr[h_start: h_start + height, w_start: w_start + width])


def longsideresize(h, w, size, skip_low_resolution):
    if h <= size[0] and w <= size[1] and skip_low_resolution:
        return h, w

    if h / w > size[0] / size[1]:
        # hxw 720x1280  size 320x640  hw_raito 9/16 > size_ratio 8/16  neww=320/720*1280=568  newh=320
        w = int(size[0] / h * w)
        h = size[0]
    else:
        # hxw 720x1280  size 480x640  hw_raito 9/16 < size_ratio 12/16   newh=640/1280*720=360 neww=640
        # hxw 1080x1920  size 720x1280  hw_raito 9/16 = size_ratio 9/16   newh=1280/1920*1080=720 neww=1280
        h = int(size[1] / w * h)
        w = size[1]
    return h, w


def shortsideresize(h, w, size, skip_low_resolution):
    if h <= size[0] and w <= size[1] and skip_low_resolution:
        return h, w

    if h / w < size[0] / size[1]:
        w = int(size[0] / h * w)
        h = size[0]
    else:
        h = int(size[1] / w * h)
        w = size[1]
    return h, w


def calculate_centered_alignment(h: int, w: int, stride: int) -> tuple:
    """Calculate centered crop parameters for stride alignment.

    Computes crop dimensions and offsets to maintain center position while
    ensuring the output dimensions are multiples of the specified stride.

    Args:
        h: Original height of the input
        w: Original width of the input
        stride: Alignment requirement (must be > 0)

    Returns:
        tuple: (vertical_offset, horizontal_offset, aligned_height, aligned_width)
    """
    # Calculate aligned dimensions
    aligned_h = h // stride * stride
    aligned_w = w // stride * stride

    # Compute centering offsets
    vertical_offset = (h - aligned_h) // 2
    horizontal_offset = (w - aligned_w) // 2

    return (vertical_offset, horizontal_offset, aligned_h, aligned_w)


def maxhwresize(ori_height, ori_width, max_hxw):
    if ori_height * ori_width > max_hxw:
        scale_factor = np.sqrt(max_hxw / (ori_height * ori_width))
        new_height = int(ori_height * scale_factor)
        new_width = int(ori_width * scale_factor)
    else:
        new_height = ori_height
        new_width = ori_width
    return new_height, new_width


class AENorm:
    """
    Apply an ae_norm to a PIL image or video.
    """

    def __init__(self):
        pass

    @staticmethod
    def __call__(clip):
        """
        Apply the center crop to the input video.

        Args:
            video (clip): The input video.

        Returns:
            video: The ae_norm video.
        """

        clip = 2.0 * clip - 1.0
        return clip

    def __repr__(self) -> str:
        return self.__class__.__name__


class CenterCropArr:
    """
    Apply a center crop to a PIL image.
    """

    def __init__(self, size=(256, 256)):
        self.size = size

    def __call__(self, pil_image):
        """
        Apply the center crop to the input PIL image.

        Args:
            pil_image (PIL.Image): The input PIL image.

        Returns:
            PIL.Image: The center-cropped image.
        """
        return center_crop_arr(pil_image, self.size)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class ResizeCropToFill:
    """
    Apply a resize crop to a PIL image.
    """

    def __init__(self, size=256):
        self.size = size

    def __call__(self, pil_image):
        """
        Apply the resize crop to the input PIL image.

        Args:
            pil_image (PIL.Image): The input PIL image.

        Returns:
            PIL.Image: The resize-cropped image.
        """
        return resize_crop_to_fill(pil_image, self.size)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class BaseCrop:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class ResizeCrop(BaseCrop):
    def __call__(self, clip):
        clip = resize_crop_to_fill(clip, self.size)
        return clip


class RandSizeCropArr(BaseCrop):
    def __call__(self, clip):
        clip = rand_size_crop_arr(clip, self.size)
        return clip


class RandomSizedCrop(BaseCrop):
    def __call__(self, clip):
        i, j, h, w = self.get_params(clip)
        return crop(clip, i, j, h, w)

    def get_params(self, clip, multiples_of=8):
        h, w = clip.shape[-2:]

        # get random target h w
        th = random.randint(self.size[0], self.size[1])
        tw = random.randint(self.size[0], self.size[1])
        # ensure that h w are factors of 8
        th = th - th % multiples_of
        tw = tw - tw % multiples_of

        if h < th:
            th = h - h % multiples_of
        if w < tw:
            tw = w - w % multiples_of

        if w == tw and h == th:
            return 0, 0, h, w

        # get random start pos
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw


class ToTensorVideo:
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    """

    def __init__(self):
        pass

    @staticmethod
    def __call__(clip):
        """
        Args:
            clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
        """
        return to_tensor(clip)

    def __repr__(self) -> str:
        return self.__class__.__name__


class ToTensorAfterResize:
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    """

    def __init__(self):
        pass

    @staticmethod
    def __call__(clip):
        """
        Args:
            clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W), but in [0, 1]
        """
        return to_tensor_after_resize(clip)

    def __repr__(self) -> str:
        return self.__class__.__name__


class RandomHorizontalFlipVideo:
    """
    Flip the video clip along the horizontal direction with a given probability
    Args:
        p (float): probability of the clip being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Size is (T, C, H, W)
        Return:
            clip (torch.tensor): Size is (T, C, H, W)
        """
        if random.random() < self.p:
            clip = hflip(clip)
        return clip

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class SpatialStrideCropVideo:
    def __init__(self, stride):
        self.stride = stride

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: cropped video clip by stride.
                size is (T, C, OH, OW)
        """
        h, w = clip.shape[-2:]
        i, j, h, w = calculate_centered_alignment(h, w, self.stride)
        return crop(clip, i, j, h, w)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(stride={self.stride})"


class LongSideResizeVideo:
    """
    First use the long side,
    then resize to the specified size
    """

    def __init__(
            self,
            size,
            skip_low_resolution=False,
            interpolation_mode="bilinear",
            align_corners=False,
            antialias=False
    ):
        self.size = size
        self.skip_low_resolution = skip_low_resolution
        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners
        self.antialias = antialias

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: scale resized video clip.
        """
        _, _, h, w = clip.shape
        tr_h, tr_w = longsideresize(h, w, self.size, self.skip_low_resolution)
        if h == tr_h and w == tr_w:
            return clip
        resize_clip = resize(
            clip,
            target_size=(tr_h, tr_w),
            interpolation_mode=self.interpolation_mode,
            align_corners=self.align_corners,
            antialias=self.antialias
        )
        return resize_clip

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"


class MaxHWResizeVideo:
    '''
    First use the h*w,
    then resize to the specified size
    '''

    def __init__(
            self,
            transform_size=None,
            interpolation_mode="bilinear",
            align_corners=False,
            antialias=False
    ):
        if transform_size is None or "max_hxw" not in transform_size:
            raise ValueError("Missing required param: max_hxw in data transform.")
        self.max_hxw = transform_size["max_hxw"]
        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners
        self.antialias = antialias

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: scale resized video clip.
        """
        _, _, h, w = clip.shape
        tr_h, tr_w = maxhwresize(h, w, self.max_hxw)
        if h == tr_h and w == tr_w:
            return clip
        resize_clip = resize(
            clip,
            target_size=(tr_h, tr_w),
            interpolation_mode=self.interpolation_mode,
            align_corners=self.align_corners,
            antialias=self.antialias
        )
        return resize_clip

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.max_hxw}, interpolation_mode={self.interpolation_mode}"


class CenterCropResizeVideo:
    """
    First use the short side for cropping length,
    center crop video, then resize to the specified size
    """

    def __init__(
            self,
            transform_size=None,
            use_short_edge=False,
            top_crop=False,
            interpolation_mode="bilinear",
            align_corners=False,
            antialias=False,
    ):
        if transform_size is None or "max_height" not in transform_size or "max_width" not in transform_size:
            raise ValueError("Missing required param: max_height or max_width in data transform.")
        self.size = (transform_size["max_height"], transform_size["max_width"])
        self.use_short_edge = use_short_edge
        self.top_crop = top_crop
        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners
        self.antialias = antialias

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: scale resized / center cropped video clip.
                size is (T, C, crop_size, crop_size)
        """
        if self.use_short_edge:
            clip_center_crop = center_crop_using_short_edge(clip)
        else:
            clip_center_crop = center_crop_th_tw(
                clip, self.size[0], self.size[1], top_crop=self.top_crop
            )

        clip_center_crop_resize = resize(
            clip_center_crop,
            target_size=self.size,
            interpolation_mode=self.interpolation_mode,
            align_corners=self.align_corners,
            antialias=self.antialias,
        )
        return clip_center_crop_resize

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"


class ResizeVideo:
    def __init__(
        self,
        transform_size="auto",
        interpolation_mode="bilinear",
        skip_low_resolution=False,
        align_corners=False,
        antialias=False,
        mode="resize"  # resize / longside / shortside / hxw
    ):
        self.mode = mode
        if mode == 'hxw':
            self.transform_size = transform_size["max_hxw"] if isinstance(transform_size, dict) else transform_size
        elif mode in ["resize", "longside", "shortside"]:
            self.transform_size = (transform_size["max_height"], transform_size["max_width"]) if isinstance(transform_size, dict) else transform_size
        else:
            raise NotImplementedError(f"ResizeVideo only support mode `resize` / `longside` / `shortside` / `hxw`, {mode} is not implemented.")

        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners
        self.antialias = antialias
        self.skip_low_resolution = skip_low_resolution

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: scale resized video clip.
        """
        h, w = clip.shape[-2:]
        if self.mode == "hxw":
            tr_h, tr_w = maxhwresize(h, w, self.transform_size)
        elif self.mode == "resize":
            tr_h, tr_w = self.transform_size
        elif self.mode == "longside":
            tr_h, tr_w = longsideresize(h, w, self.transform_size, skip_low_resolution=self.skip_low_resolution)
        elif self.mode == "shortside":
            tr_h, tr_w = shortsideresize(h, w, self.transform_size, skip_low_resolution=self.skip_low_resolution)

        if h == tr_h and w == tr_w:
            return clip
        resize_clip = resize(
            clip,
            target_size=(tr_h, tr_w),
            interpolation_mode=self.interpolation_mode,
            align_corners=self.align_corners,
            antialias=self.antialias
        )
        return resize_clip

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.transform_size}, interpolation_mode={self.interpolation_mode})"


class UCFCenterCropVideo:
    """
    First scale to the specified size in equal proportion to the short edge,
    then center cropping
    """

    def __init__(
            self,
            size,
            interpolation_mode="bilinear",
    ):
        if isinstance(size, list):
            size = tuple(size)
        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError(
                    f"size should be tuple (height, width), instead got {size}"
                )
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: scale resized / center cropped video clip.
                size is (T, C, crop_size, crop_size)
        """
        clip_resize = resize_scale(
            clip=clip, target_size=self.size, interpolation_mode=self.interpolation_mode
        )
        clip_center_crop = center_crop(clip_resize, self.size)
        return clip_center_crop

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"


class TemporalRandomCrop:
    """Temporally crop the given frame indices at a random location.
    Args:
            size (int): Desired length of frames will be seen in the model.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, total_frames):
        rand_end = max(0, total_frames - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, total_frames)
        return begin_index, end_index


class Expand2Square:
    """
    Expand the given PIL image to a square by padding it with a background color.
    Args:
        mean (sequence): Sequence of means for each channel.
    """

    def __init__(self, mean):
        self.background_color = tuple(int(x * 255) for x in mean)

    def __call__(self, pil_img):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), self.background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), self.background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result


class JpegDegradationSimulator:
    """
    Degrade an image based on the JPEG quality.
    """

    def __init__(self, min_quality=75, max_quality=101):
        """
        Initialize the simulator with a list of qualities.
        """
        self.qualities = list(range(min_quality, max_quality))
        self.jpeg_degrade_functions = {
            quality: self._simulate_jpeg_degradation(quality)
            for quality in self.qualities
        }

    @staticmethod
    def _simulate_jpeg_degradation(quality):
        """
        Create a function to degrade an image based on the JPEG quality.
        """

        def jpeg_degrade(img):
            with io.BytesIO() as output:
                img.convert("RGB").save(output, format="JPEG", quality=quality)
                output.seek(0)  # Move the reading cursor to the start of the stream
                img_jpeg = Image.open(output).copy()  # Use .copy() to make sure the image is loaded in memory
            return img_jpeg

        return jpeg_degrade

    def __call__(self, img):
        """
        Apply a random JPEG degradation from the available qualities.
        """
        transform = T.RandomChoice([T.Lambda(self.jpeg_degrade_functions[quality]) for quality in self.qualities])
        return transform(img)

    def __repr__(self):
        """
        Represent the class instance.
        """
        return f"JpegDegradationSimulator(qualities={self.qualities})"


class CenterCropVideo:
    def __init__(
            self,
            size,
            interpolation_mode="bilinear",
    ):
        if isinstance(size, list):
            size = tuple(size)
        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError(
                    f"size should be tuple (height, width), instead got {size}"
                )
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: center cropped video clip.
                size is (T, C, crop_size, crop_size)
        """
        clip_center_crop = center_crop(clip, self.size)
        return clip_center_crop

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"


class ResizeCropVideo:
    def __init__(self, size, interpolation_mode="bilinear", antialias=False, align_corners=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.interpolation_mode = interpolation_mode
        self.antialias = antialias
        self.align_corners = align_corners
        self.interpolate_parameters = {
            "interpolation_mode": self.interpolation_mode,
            "antialias": self.antialias,
            "align_corners": self.align_corners,
        }

    def __call__(self, clip):
        clip = resize_crop_to_fill(clip, self.size, self.interpolate_parameters)
        return clip

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class ResizeToFill:
    def __init__(self, size, interpolation_mode="bilinear", antialias=False, align_corners=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.interpolation_mode = interpolation_mode
        self.antialias = antialias
        self.align_corners = align_corners
        self.interpolate_parameters = {
            "interpolation_mode": self.interpolation_mode,
            "antialias": self.antialias,
            "align_corners": self.align_corners,
        }

    def __call__(self, clip):
        if clip.shape[-2:] != self.size:
            canvas_height, canvas_width = self.size
            ref_height, ref_width = clip.shape[-2:]
            white_canvas = torch.ones((1, 3, canvas_height, canvas_width))  # [-1, 1]
            scale = min(canvas_height / ref_height, canvas_width / ref_width)
            new_height = int(ref_height * scale)
            new_width = int(ref_width * scale)
            clip = resize(clip, (new_height, new_width), **self.interpolate_parameters)
            top = (canvas_height - new_height) // 2
            left = (canvas_width - new_width) // 2
            white_canvas[:, :, top:top + new_height, left:left + new_width] = clip
            clip = white_canvas
        return clip


MAX_PIXELS = 14 * 14 * 9 * 1024


class MaxLongEdgeMinShortEdgeResize:
    """Resize the input image so that its longest side and shortest side are within a specified range,
    ensuring that both sides are divisible by a specified stride. This class also handles tensor conversion
    and normalization to provide a complete image transformation pipeline.
    """
    def __init__(
            self,
            max_image_size: int,
            min_image_size: int,
            image_stride: int,
            max_pixels: int = MAX_PIXELS,
            interpolation=T.InterpolationMode.BICUBIC,
            antialias: bool = True
    ):
        self.max_size = max_image_size
        self.min_size = min_image_size
        self.stride = image_stride
        self.max_pixels = max_pixels
        self.interpolation = interpolation
        self.antialias = antialias

        # Initialize transformation components
        self.to_tensor_transform = T.ToTensor()
        self.normalize_transform = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)

    def _make_divisible(self, value: int, stride: int) -> int:
        return max(stride, int(round(value / stride) * stride))

    def _apply_scale(self, width: int, height: int, scale: float) -> tuple:
        """Apply scaling and ensure dimensions are divisible by stride."""
        new_width = round(width * scale)
        new_height = round(height * scale)
        new_width = self._make_divisible(new_width, self.stride)
        new_height = self._make_divisible(new_height, self.stride)
        return new_width, new_height

    def __call__(self, img, img_num: int = 1):
        # Store original type to handle return type appropriately
        original_is_tensor = isinstance(img, torch.Tensor)

        if original_is_tensor:
            height, width = img.shape[-2:]
        else:
            width, height = img.size

        # Calculate initial scale based on size constraints
        scale = min(self.max_size / max(width, height), 1.0)
        scale = max(scale, self.min_size / min(width, height))
        new_width, new_height = self._apply_scale(width, height, scale)

        # Ensure the number of pixels does not exceed max_pixels
        if new_width * new_height > self.max_pixels / img_num:
            scale = self.max_pixels / img_num / (new_width * new_height)
            new_width, new_height = self._apply_scale(new_width, new_height, scale)

        # Ensure longest edge does not exceed max_size
        if max(new_width, new_height) > self.max_size:
            scale = self.max_size / max(new_width, new_height)
            new_width, new_height = self._apply_scale(new_width, new_height, scale)

        # Apply resize
        resized_img = F.resize(img, (new_height, new_width), self.interpolation, antialias=self.antialias)

        # Convert to tensor if input was PIL image
        if not original_is_tensor:
            tensor_img = self.to_tensor_transform(resized_img)
        else:
            tensor_img = resized_img

        # Apply normalization
        normalized_img = self.normalize_transform(tensor_img)

        return normalized_img


class AffineVideo:
    def __init__(self, gamma=2.0, beta=-1.0):
        self.gamma = gamma
        self.beta = beta

    def __call__(self, clip):
        clip = self.gamma * clip + self.beta
        return clip

    def __repr__(self) -> str:
        return self.__class__.__name__


class MaskGenerator:
    def __init__(self, mask_ratios):
        valid_mask_names = [
            "identity",
            "quarter_random",
            "quarter_head",
            "quarter_tail",
            "quarter_head_tail",
            "image_random",
            "image_head",
            "image_tail",
            "image_head_tail",
            "random",
            "interpolate",
        ]
        if not all(
                mask_name in valid_mask_names for mask_name in mask_ratios.keys()
        ):
            raise Exception(f"mask_name should be one of {valid_mask_names}, got {mask_ratios.keys()}")
        if not all(
                mask_ratio >= 0 for mask_ratio in mask_ratios.values()
        ):
            raise Exception(f"mask_ratio should be greater than or equal to 0, got {mask_ratios.values()}")
        if not all(
                mask_ratio <= 1 for mask_ratio in mask_ratios.values()
        ):
            raise Exception(f"mask_ratio should be less than or equal to 1, got {mask_ratios.values()}")
        # sum of mask_ratios should be 1
        if "identity" not in mask_ratios:
            mask_ratios["identity"] = 1.0 - sum(mask_ratios.values())
        if not math.isclose(
                sum(mask_ratios.values()), 1.0, abs_tol=1e-6
        ):
            raise Exception(f"sum of mask_ratios should be 1, got {sum(mask_ratios.values())}")
        self.mask_ratios = mask_ratios

    def get_mask(self, x):
        mask_type = random.random()
        mask_name = None
        prob_acc = 0.0
        for mask, mask_ratio in self.mask_ratios.items():
            prob_acc += mask_ratio
            if mask_type < prob_acc:
                mask_name = mask
                break

        vae_micro_frame_size = 17
        video_frames = x.shape[1]
        temporal_vae_downsample = 4
        num_frames = (video_frames // vae_micro_frame_size) * \
                     math.ceil(vae_micro_frame_size / temporal_vae_downsample)
        # Hardcoded condition_frames
        condition_frames_max = num_frames // 4

        mask = torch.ones(num_frames, dtype=torch.bool, device=x.device)
        if num_frames <= 1:
            return mask

        if mask_name == "quarter_random":
            random_size = random.randint(1, condition_frames_max)
            random_pos = random.randint(0, num_frames - random_size)
            mask[random_pos: random_pos + random_size] = 0
        elif mask_name == "image_random":
            random_size = 1
            random_pos = random.randint(0, num_frames - random_size)
            mask[random_pos: random_pos + random_size] = 0
        elif mask_name == "quarter_head":
            random_size = random.randint(1, condition_frames_max)
            mask[:random_size] = 0
        elif mask_name == "image_head":
            random_size = 1
            mask[:random_size] = 0
        elif mask_name == "quarter_tail":
            random_size = random.randint(1, condition_frames_max)
            mask[-random_size:] = 0
        elif mask_name == "image_tail":
            random_size = 1
            mask[-random_size:] = 0
        elif mask_name == "quarter_head_tail":
            random_size = random.randint(1, condition_frames_max)
            mask[:random_size] = 0
            mask[-random_size:] = 0
        elif mask_name == "image_head_tail":
            random_size = 1
            mask[:random_size] = 0
            mask[-random_size:] = 0
        elif mask_name == "interpolate":
            random_start = random.randint(0, 1)
            mask[random_start::2] = 0
        elif mask_name == "random":
            mask_ratio = random.uniform(0.1, 0.9)
            mask = torch.rand(num_frames, device=x.device) > mask_ratio
            # if mask is all False, set the last frame to True
            if not mask.any():
                mask[-1] = 1

        return mask


low_aesthetic_score_notices_video = [
    "This video has a low aesthetic quality.",
    "The beauty of this video is minimal.",
    "This video scores low in aesthetic appeal.",
    "The aesthetic quality of this video is below average.",
    "This video ranks low for beauty.",
    "The artistic quality of this video is lacking.",
    "This video has a low score for aesthetic value.",
    "The visual appeal of this video is low.",
    "This video is rated low for beauty.",
    "The aesthetic quality of this video is poor.",
]

high_aesthetic_score_notices_video = [
    "This video has a high aesthetic quality.",
    "The beauty of this video is exceptional.",
    "This video scores high in aesthetic value.",
    "With its harmonious colors and balanced composition.",
    "This video ranks highly for aesthetic quality",
    "The artistic quality of this video is excellent.",
    "This video is rated high for beauty.",
    "The aesthetic quality of this video is impressive.",
    "This video has a top aesthetic score.",
    "The visual appeal of this video is outstanding.",
]

low_aesthetic_score_notices_image = [
    "This image has a low aesthetic quality.",
    "The beauty of this image is minimal.",
    "This image scores low in aesthetic appeal.",
    "The aesthetic quality of this image is below average.",
    "This image ranks low for beauty.",
    "The artistic quality of this image is lacking.",
    "This image has a low score for aesthetic value.",
    "The visual appeal of this image is low.",
    "This image is rated low for beauty.",
    "The aesthetic quality of this image is poor.",
]

high_aesthetic_score_notices_image = [
    "This image has a high aesthetic quality.",
    "The beauty of this image is exceptional",
    "This photo scores high in aesthetic value.",
    "With its harmonious colors and balanced composition.",
    "This image ranks highly for aesthetic quality.",
    "The artistic quality of this photo is excellent.",
    "This image is rated high for beauty.",
    "The aesthetic quality of this image is impressive.",
    "This photo has a top aesthetic score.",
    "The visual appeal of this image is outstanding.",
]


def add_aesthetic_notice_video(caption, aesthetic_score):
    if aesthetic_score <= 4.25:
        notice = random.choice(low_aesthetic_score_notices_video)
        return random.choice([caption + ' ' + notice, notice + ' ' + caption])
    if aesthetic_score >= 5.75:
        notice = random.choice(high_aesthetic_score_notices_video)
        return random.choice([caption + ' ' + notice, notice + ' ' + caption])
    return caption


def add_aesthetic_notice_image(caption, aesthetic_score):
    if aesthetic_score <= 4.25:
        notice = random.choice(low_aesthetic_score_notices_image)
        return random.choice([caption + ' ' + notice, notice + ' ' + caption])
    if aesthetic_score >= 5.75:
        notice = random.choice(high_aesthetic_score_notices_image)
        return random.choice([caption + ' ' + notice, notice + ' ' + caption])
    return caption
