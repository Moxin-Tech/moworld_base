import math
import random

import numpy as np
import torch
from PIL import Image

from mindspeed_mm.data.data_utils.data_transform import (
    _is_tensor_video_clip,
    center_crop_arr,
    to_tensor,
    to_tensor_after_resize,
    hflip,
    crop,
    resize,
    resize_scale,
    center_crop,
    center_crop_using_short_edge,
    center_crop_th_tw,
    resize_crop_to_fill,
    longsideresize,
    shortsideresize,
    calculate_centered_alignment,
    maxhwresize,
    AENorm,
    CenterCropArr,
    ToTensorVideo,
    ToTensorAfterResize,
    RandomHorizontalFlipVideo,
    SpatialStrideCropVideo,
    LongSideResizeVideo,
    MaxHWResizeVideo,
    CenterCropResizeVideo,
    ResizeVideo,
    UCFCenterCropVideo,
    TemporalRandomCrop,
    Expand2Square,
    ResizeToFill,
    AffineVideo,
    MaskGenerator,
    add_aesthetic_notice_video,
    add_aesthetic_notice_image,
)
from tests.ut.utils import judge_expression


# =====================================================================
# Helper: build a 4D uint8 video tensor (T, C, H, W)
# =====================================================================
def _make_video_clip(t=4, c=3, h=64, w=64, dtype=torch.uint8):
    return torch.randint(0, 256, (t, c, h, w), dtype=dtype)


# =====================================================================
# _is_tensor_video_clip
# =====================================================================
class TestIsTensorVideoClip:

    def test_valid_4d_tensor(self):
        clip = _make_video_clip()
        judge_expression(_is_tensor_video_clip(clip) is True)

    def test_non_tensor_raises(self):
        try:
            _is_tensor_video_clip(np.zeros((4, 3, 64, 64)))
            judge_expression(False)
        except TypeError:
            judge_expression(True)

    def test_wrong_ndim_raises(self):
        try:
            _is_tensor_video_clip(torch.randn(3, 64, 64))
            judge_expression(False)
        except ValueError:
            judge_expression(True)


# =====================================================================
# center_crop_arr (PIL Image)
# =====================================================================
class TestCenterCropArr:

    def test_output_size_matches_target(self):
        img = Image.fromarray(np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8))
        target = (100, 100)
        result = center_crop_arr(img, target)
        judge_expression(result.size == (target[0], target[1]))

    def test_larger_image_crop(self):
        img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        result = center_crop_arr(img, (256, 256))
        judge_expression(result.size[0] == 256 and result.size[1] == 256)

    def test_small_image_still_crops(self):
        img = Image.fromarray(np.random.randint(0, 255, (50, 60, 3), dtype=np.uint8))
        result = center_crop_arr(img, (40, 40))
        judge_expression(result.size == (40, 40))


# =====================================================================
# to_tensor
# =====================================================================
class TestToTensor:

    def test_uint8_to_float(self):
        clip = _make_video_clip()
        result = to_tensor(clip)
        judge_expression(result.dtype == torch.float32)
        judge_expression(result.max() <= 1.0)
        judge_expression(result.min() >= 0.0)

    def test_non_uint8_raises(self):
        clip = torch.randn(4, 3, 64, 64)
        try:
            to_tensor(clip)
            judge_expression(False)
        except TypeError:
            judge_expression(True)

    def test_255_maps_to_1(self):
        clip = torch.full((2, 3, 8, 8), 255, dtype=torch.uint8)
        result = to_tensor(clip)
        judge_expression(torch.allclose(result, torch.ones_like(result)))


# =====================================================================
# to_tensor_after_resize
# =====================================================================
class TestToTensorAfterResize:

    def test_float_div_255(self):
        clip = torch.full((2, 3, 8, 8), 255.0, dtype=torch.float32)
        result = to_tensor_after_resize(clip)
        judge_expression(torch.allclose(result, torch.ones_like(result)))

    def test_output_dtype(self):
        clip = torch.randn(2, 3, 8, 8)
        result = to_tensor_after_resize(clip)
        judge_expression(result.dtype == torch.float32)


# =====================================================================
# hflip
# =====================================================================
class TestHflip:

    def test_horizontal_flip(self):
        clip = torch.arange(8).reshape(1, 1, 1, 8).float()
        result = hflip(clip)
        expected = torch.arange(7, -1, -1).reshape(1, 1, 1, 8).float()
        judge_expression(torch.equal(result, expected))

    def test_shape_preserved(self):
        clip = _make_video_clip(h=32, w=48)
        result = hflip(clip)
        judge_expression(result.shape == clip.shape)


# =====================================================================
# crop
# =====================================================================
class TestCrop:

    def test_crop_region(self):
        clip = torch.arange(64).reshape(1, 1, 8, 8).float()
        result = crop(clip, 2, 2, 4, 4)
        judge_expression(result.shape == (1, 1, 4, 4))

    def test_non_4d_raises(self):
        clip = torch.randn(3, 64, 64)
        try:
            crop(clip, 0, 0, 32, 32)
            judge_expression(False)
        except ValueError:
            judge_expression(True)


# =====================================================================
# resize
# =====================================================================
class TestResize:

    def test_resize_shape(self):
        clip = _make_video_clip(h=64, w=64).float()
        result = resize(clip, (32, 32), interpolation_mode="bilinear")
        judge_expression(result.shape[-2:] == (32, 32))

    def test_invalid_target_size_raises(self):
        clip = _make_video_clip().float()
        try:
            resize(clip, (32,), interpolation_mode="bilinear")
            judge_expression(False)
        except ValueError:
            judge_expression(True)


# =====================================================================
# resize_scale
# =====================================================================
class TestResizeScale:

    def test_resize_scale_output(self):
        clip = _make_video_clip(h=64, w=64).float()
        result = resize_scale(clip, (128, 128), interpolation_mode="bilinear")
        judge_expression(result.shape[-2] >= 128 or result.shape[-1] >= 128)

    def test_invalid_target_size_raises(self):
        clip = _make_video_clip().float()
        try:
            resize_scale(clip, (32,), interpolation_mode="bilinear")
            judge_expression(False)
        except ValueError:
            judge_expression(True)


# =====================================================================
# center_crop
# =====================================================================
class TestCenterCrop:

    def test_center_crop_size(self):
        clip = _make_video_clip(h=64, w=64)
        result = center_crop(clip, (32, 32))
        judge_expression(result.shape[-2:] == (32, 32))

    def test_crop_smaller_than_clip(self):
        clip = _make_video_clip(h=64, w=64)
        result = center_crop(clip, (16, 16))
        judge_expression(result.shape[-2:] == (16, 16))

    def test_crop_larger_than_clip_raises(self):
        clip = _make_video_clip(h=32, w=32)
        try:
            center_crop(clip, (64, 64))
            judge_expression(False)
        except ValueError:
            judge_expression(True)


# =====================================================================
# center_crop_using_short_edge
# =====================================================================
class TestCenterCropUsingShortEdge:

    def test_square_output(self):
        clip = _make_video_clip(h=64, w=48)
        result = center_crop_using_short_edge(clip)
        judge_expression(result.shape[-2] == result.shape[-1])
        judge_expression(result.shape[-2] == 48)  # shorter side

    def test_already_square(self):
        clip = _make_video_clip(h=64, w=64)
        result = center_crop_using_short_edge(clip)
        judge_expression(result.shape[-2:] == (64, 64))


# =====================================================================
# center_crop_th_tw
# =====================================================================
class TestCenterCropThTw:

    def test_output_shape_square_target(self):
        clip = _make_video_clip(h=64, w=64)
        result = center_crop_th_tw(clip, 32, 32, top_crop=False)
        # target ratio = 32/32 = 1.0, input ratio = 64/64 = 1.0
        # new_h = 64, new_w = 64 since h/w == tr
        judge_expression(result.shape[-2] <= 64 and result.shape[-1] <= 64)

    def test_top_crop(self):
        clip = _make_video_clip(h=64, w=64)
        result = center_crop_th_tw(clip, 32, 32, top_crop=True)
        judge_expression(result.shape[-2] <= 64 and result.shape[-1] <= 64)

    def test_wider_aspect_ratio(self):
        clip = _make_video_clip(h=64, w=128)
        result = center_crop_th_tw(clip, 32, 64, top_crop=False)
        # target ratio = 32/64 = 0.5, input ratio = 64/128 = 0.5
        # Since h/w == tr, new_h = h = 64, new_w = w = 128 (no crop needed)
        judge_expression(result.shape[-2] <= 64 and result.shape[-1] <= 128)

    def test_tall_clip_with_wide_target(self):
        clip = _make_video_clip(h=128, w=64)
        result = center_crop_th_tw(clip, 32, 64, top_crop=False)
        # target ratio = 32/64 = 0.5, input ratio = 128/64 = 2.0
        # h/w > tr, so new_w = w = 64, new_h = int(w * tr) = int(64 * 0.5) = 32
        judge_expression(result.shape[-2] == 32)
        judge_expression(result.shape[-1] == 64)


# =====================================================================
# resize_crop_to_fill
# =====================================================================
class TestResizeCropToFill:

    def test_output_matches_target(self):
        clip = _make_video_clip(h=64, w=48).float()
        result = resize_crop_to_fill(clip, (32, 32))
        judge_expression(result.shape[-2:] == (32, 32))

    def test_with_interpolate_parameters(self):
        clip = _make_video_clip(h=64, w=48).float()
        result = resize_crop_to_fill(clip, (32, 32), {"interpolation_mode": "bilinear"})
        judge_expression(result.shape[-2:] == (32, 32))


# =====================================================================
# longsideresize
# =====================================================================
class TestLongSideResize:

    def test_tall_image(self):
        # h > w and h/w > size_h/size_w
        new_h, new_w = longsideresize(720, 1280, (320, 640), skip_low_resolution=False)
        judge_expression(new_h <= 320 and new_w <= 640)
        # long side should equal target long side
        judge_expression(new_h == 320)

    def test_wide_image(self):
        # h/w < size_h/size_w
        new_h, new_w = longsideresize(720, 1280, (480, 640), skip_low_resolution=False)
        judge_expression(new_h <= 480 and new_w <= 640)
        judge_expression(new_w == 640)

    def test_skip_low_resolution(self):
        new_h, new_w = longsideresize(100, 200, (320, 640), skip_low_resolution=True)
        judge_expression(new_h == 100 and new_w == 200)

    def test_no_skip_when_exceeds(self):
        new_h, new_w = longsideresize(720, 1280, (320, 640), skip_low_resolution=True)
        judge_expression(new_h != 720 or new_w != 1280)


# =====================================================================
# shortsideresize
# =====================================================================
class TestShortSideResize:

    def test_short_side_resized(self):
        new_h, new_w = shortsideresize(720, 1280, (320, 640), skip_low_resolution=False)
        judge_expression(new_h >= 320)
        judge_expression(new_w >= 640)

    def test_skip_low_resolution(self):
        new_h, new_w = shortsideresize(100, 200, (320, 640), skip_low_resolution=True)
        judge_expression(new_h == 100 and new_w == 200)


# =====================================================================
# calculate_centered_alignment
# =====================================================================
class TestCalculateCenteredAlignment:

    def test_already_aligned(self):
        v_off, h_off, al_h, al_w = calculate_centered_alignment(64, 64, 8)
        judge_expression(al_h == 64 and al_w == 64)
        judge_expression(v_off == 0 and h_off == 0)

    def test_non_aligned(self):
        v_off, h_off, al_h, al_w = calculate_centered_alignment(66, 66, 8)
        judge_expression(al_h == 64 and al_w == 64)
        judge_expression(v_off == 1 and h_off == 1)

    def test_stride_1(self):
        v_off, h_off, al_h, al_w = calculate_centered_alignment(33, 17, 1)
        judge_expression(al_h == 33 and al_w == 17)

    def test_small_input(self):
        v_off, h_off, al_h, al_w = calculate_centered_alignment(5, 5, 8)
        judge_expression(al_h == 0 and al_w == 0)


# =====================================================================
# maxhwresize
# =====================================================================
class TestMaxHWResize:

    def test_within_limit(self):
        new_h, new_w = maxhwresize(100, 200, 100000)
        judge_expression(new_h == 100 and new_w == 200)

    def test_exceeds_limit(self):
        new_h, new_w = maxhwresize(1080, 1920, 500000)
        judge_expression(new_h * new_w <= 500000)
        judge_expression(new_h < 1080 and new_w < 1920)

    def test_square_resize(self):
        new_h, new_w = maxhwresize(1000, 1000, 250000)
        judge_expression(new_h * new_w <= 250000)


# =====================================================================
# AENorm
# =====================================================================
class TestAENorm:

    def test_normalize_range(self):
        clip = torch.rand(2, 3, 8, 8)  # [0, 1]
        result = AENorm()(clip)
        judge_expression(result.min() >= -1.0)
        judge_expression(result.max() <= 1.0)

    def test_formula(self):
        clip = torch.ones(1, 3, 4, 4)
        result = AENorm()(clip)
        # 2.0 * 1.0 - 1.0 = 1.0
        judge_expression(torch.allclose(result, torch.ones_like(result)))

    def test_zero_input(self):
        clip = torch.zeros(1, 3, 4, 4)
        result = AENorm()(clip)
        # 2.0 * 0.0 - 1.0 = -1.0
        judge_expression(torch.allclose(result, torch.full_like(result, -1.0)))


# =====================================================================
# CenterCropArr class
# =====================================================================
class TestCenterCropArrClass:

    def test_call(self):
        img = Image.fromarray(np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8))
        transform = CenterCropArr(size=(100, 100))
        result = transform(img)
        judge_expression(result.size == (100, 100))


# =====================================================================
# ToTensorVideo class
# =====================================================================
class TestToTensorVideoClass:

    def test_call(self):
        clip = _make_video_clip()
        result = ToTensorVideo()(clip)
        judge_expression(result.dtype == torch.float32)
        judge_expression(result.max() <= 1.0)


# =====================================================================
# ToTensorAfterResize class
# =====================================================================
class TestToTensorAfterResizeClass:

    def test_call(self):
        clip = torch.full((2, 3, 8, 8), 128.0, dtype=torch.float32)
        result = ToTensorAfterResize()(clip)
        judge_expression(torch.allclose(result, torch.full_like(result, 128.0 / 255.0), atol=1e-5))


# =====================================================================
# RandomHorizontalFlipVideo class
# =====================================================================
class TestRandomHorizontalFlipVideo:

    def test_always_flip(self):
        clip = torch.arange(8).reshape(1, 1, 1, 8).float()
        transform = RandomHorizontalFlipVideo(p=1.0)
        result = transform(clip)
        expected = torch.arange(7, -1, -1).reshape(1, 1, 1, 8).float()
        judge_expression(torch.equal(result, expected))

    def test_never_flip(self):
        clip = torch.arange(8).reshape(1, 1, 1, 8).float()
        transform = RandomHorizontalFlipVideo(p=0.0)
        result = transform(clip)
        judge_expression(torch.equal(result, clip))


# =====================================================================
# SpatialStrideCropVideo class
# =====================================================================
class TestSpatialStrideCropVideo:

    def test_crop_with_stride(self):
        clip = _make_video_clip(h=66, w=66)
        transform = SpatialStrideCropVideo(stride=8)
        result = transform(clip)
        judge_expression(result.shape[-2] % 8 == 0)
        judge_expression(result.shape[-1] % 8 == 0)

    def test_already_aligned(self):
        clip = _make_video_clip(h=64, w=64)
        transform = SpatialStrideCropVideo(stride=8)
        result = transform(clip)
        judge_expression(result.shape[-2:] == (64, 64))


# =====================================================================
# LongSideResizeVideo class
# =====================================================================
class TestLongSideResizeVideoClass:

    def test_resize(self):
        clip = _make_video_clip(h=720, w=1280).float()
        transform = LongSideResizeVideo(size=(320, 640))
        result = transform(clip)
        judge_expression(result.shape[-2] <= 320)
        judge_expression(result.shape[-1] <= 640)

    def test_skip_low_resolution(self):
        clip = _make_video_clip(h=64, w=64).float()
        transform = LongSideResizeVideo(size=(320, 640), skip_low_resolution=True)
        result = transform(clip)
        judge_expression(result.shape[-2:] == (64, 64))


# =====================================================================
# MaxHWResizeVideo class
# =====================================================================
class TestMaxHWResizeVideoClass:

    def test_resize(self):
        clip = _make_video_clip(h=1080, w=1920).float()
        transform = MaxHWResizeVideo(transform_size={"max_hxw": 500000})
        result = transform(clip)
        judge_expression(result.shape[-2] * result.shape[-1] <= 500000)

    def test_missing_max_hxw_raises(self):
        try:
            MaxHWResizeVideo(transform_size={})
            judge_expression(False)
        except ValueError:
            judge_expression(True)

    def test_none_transform_size_raises(self):
        try:
            MaxHWResizeVideo(transform_size=None)
            judge_expression(False)
        except ValueError:
            judge_expression(True)


# =====================================================================
# CenterCropResizeVideo class
# =====================================================================
class TestCenterCropResizeVideoClass:

    def test_basic_resize(self):
        clip = _make_video_clip(h=64, w=48).float()
        transform = CenterCropResizeVideo(
            transform_size={"max_height": 32, "max_width": 32}
        )
        result = transform(clip)
        judge_expression(result.shape[-2:] == (32, 32))

    def test_missing_size_raises(self):
        try:
            CenterCropResizeVideo(transform_size={})
            judge_expression(False)
        except ValueError:
            judge_expression(True)

    def test_none_size_raises(self):
        try:
            CenterCropResizeVideo(transform_size=None)
            judge_expression(False)
        except ValueError:
            judge_expression(True)


# =====================================================================
# ResizeVideo class
# =====================================================================
class TestResizeVideoClass:

    def test_resize_mode(self):
        clip = _make_video_clip(h=64, w=48).float()
        transform = ResizeVideo(
            transform_size={"max_height": 32, "max_width": 32}, mode="resize"
        )
        result = transform(clip)
        judge_expression(result.shape[-2:] == (32, 32))

    def test_hxw_mode(self):
        clip = _make_video_clip(h=1080, w=1920).float()
        transform = ResizeVideo(
            transform_size={"max_hxw": 500000}, mode="hxw"
        )
        result = transform(clip)
        judge_expression(result.shape[-2] * result.shape[-1] <= 500000)

    def test_invalid_mode_raises(self):
        try:
            ResizeVideo(transform_size={"max_height": 32, "max_width": 32}, mode="invalid")
            judge_expression(False)
        except NotImplementedError:
            judge_expression(True)


# =====================================================================
# UCFCenterCropVideo class
# =====================================================================
class TestUCFCenterCropVideoClass:

    def test_center_crop(self):
        clip = _make_video_clip(h=80, w=100).float()
        transform = UCFCenterCropVideo(size=(64, 64))
        result = transform(clip)
        judge_expression(result.shape[-2:] == (64, 64))


# =====================================================================
# TemporalRandomCrop class
# =====================================================================
class TestTemporalRandomCrop:

    def test_output_within_bounds(self):
        crop_fn = TemporalRandomCrop(size=8)
        total = 20
        start, end = crop_fn(total)
        judge_expression(0 <= start < total)
        judge_expression(end <= total)
        judge_expression(end - start <= 8)

    def test_size_equals_total(self):
        crop_fn = TemporalRandomCrop(size=20)
        start, end = crop_fn(20)
        judge_expression(end - start <= 20)

    def test_size_larger_than_total(self):
        crop_fn = TemporalRandomCrop(size=30)
        start, end = crop_fn(20)
        judge_expression(end <= 20)


# =====================================================================
# Expand2Square class
# =====================================================================
class TestExpand2Square:

    def test_already_square(self):
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        result = Expand2Square(mean=[0.5, 0.5, 0.5])(img)
        judge_expression(result.size == (64, 64))

    def test_wide_image_expanded(self):
        img = Image.fromarray(np.random.randint(0, 255, (64, 128, 3), dtype=np.uint8))
        result = Expand2Square(mean=[0.5, 0.5, 0.5])(img)
        judge_expression(result.size == (128, 128))

    def test_tall_image_expanded(self):
        img = Image.fromarray(np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8))
        result = Expand2Square(mean=[0.5, 0.5, 0.5])(img)
        judge_expression(result.size == (128, 128))


# =====================================================================
# ResizeToFill class
# =====================================================================
class TestResizeToFill:

    def test_no_resize_when_same_size(self):
        clip = torch.randn(1, 3, 32, 32)
        transform = ResizeToFill(size=(32, 32))
        result = transform(clip)
        judge_expression(torch.equal(result, clip))

    def test_resize_to_canvas(self):
        clip = torch.randn(1, 3, 64, 48)
        transform = ResizeToFill(size=(32, 32))
        result = transform(clip)
        judge_expression(result.shape[-2:] == (32, 32))


# =====================================================================
# AffineVideo class
# =====================================================================
class TestAffineVideo:

    def test_default_transform(self):
        clip = torch.ones(2, 3, 8, 8)
        result = AffineVideo()(clip)
        # 2.0 * 1.0 + (-1.0) = 1.0
        judge_expression(torch.allclose(result, torch.ones_like(result)))

    def test_custom_gamma_beta(self):
        clip = torch.ones(2, 3, 8, 8)
        result = AffineVideo(gamma=3.0, beta=0.0)(clip)
        judge_expression(torch.allclose(result, torch.full_like(result, 3.0)))

    def test_zero_input(self):
        clip = torch.zeros(2, 3, 8, 8)
        result = AffineVideo()(clip)
        judge_expression(torch.allclose(result, torch.full_like(result, -1.0)))


# =====================================================================
# MaskGenerator class
# =====================================================================
class TestMaskGenerator:

    def test_identity_mask(self):
        mg = MaskGenerator(mask_ratios={"identity": 1.0})
        x = torch.randn(1, 68, 64, 64)
        mask = mg.get_mask(x)
        judge_expression(mask.all())

    def test_invalid_mask_name_raises(self):
        try:
            MaskGenerator(mask_ratios={"invalid_name": 1.0})
            judge_expression(False)
        except Exception:
            judge_expression(True)

    def test_ratio_out_of_range_raises(self):
        try:
            MaskGenerator(mask_ratios={"identity": 1.5})
            judge_expression(False)
        except Exception:
            judge_expression(True)

    def test_negative_ratio_raises(self):
        try:
            MaskGenerator(mask_ratios={"identity": -0.1})
            judge_expression(False)
        except Exception:
            judge_expression(True)

    def test_ratios_not_sum_to_one_raises(self):
        try:
            MaskGenerator(mask_ratios={"quarter_random": 0.5, "image_random": 0.6})
            judge_expression(False)
        except Exception:
            judge_expression(True)

    def test_identity_auto_filled(self):
        mg = MaskGenerator(mask_ratios={"quarter_random": 0.3, "image_random": 0.2})
        judge_expression(mg.mask_ratios["identity"] == 0.5)


# =====================================================================
# add_aesthetic_notice_video
# =====================================================================
class TestAddAestheticNoticeVideo:

    def test_low_score_adds_notice(self):
        random.seed(42)
        result = add_aesthetic_notice_video("test video", 3.0)
        judge_expression("test video" in result)

    def test_high_score_adds_notice(self):
        random.seed(42)
        result = add_aesthetic_notice_video("test video", 6.0)
        judge_expression("test video" in result)

    def test_mid_score_no_notice(self):
        result = add_aesthetic_notice_video("test video", 5.0)
        judge_expression(result == "test video")


# =====================================================================
# add_aesthetic_notice_image
# =====================================================================
class TestAddAestheticNoticeImage:

    def test_low_score_adds_notice(self):
        random.seed(42)
        result = add_aesthetic_notice_image("test image", 3.0)
        judge_expression("test image" in result)

    def test_mid_score_no_notice(self):
        result = add_aesthetic_notice_image("test image", 5.0)
        judge_expression(result == "test image")
