import math
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from tests.ut.utils import judge_expression


# =====================================================================
# Helper: mock vframes object for video processors
# =====================================================================
class MockVFrames:
    """Mock video frame container that doesn't read local files."""

    def __init__(self, frames=None, fps=24.0, timestamps=None):
        """
        Args:
            frames: torch.Tensor in (T, C, H, W) format
            fps: frames per second
            timestamps: list of (start, end) tuples for each frame
        """
        self._frames = frames
        self._fps = fps
        self._timestamps = timestamps

    def get_len(self):
        return self._frames.shape[0]

    def get_video_fps(self):
        return self._fps

    def get_batch(self, indices):
        if isinstance(indices, (list, np.ndarray)):
            indices = list(indices)
        return self._frames[indices]

    def get_frame_timestamp(self, i):
        if self._timestamps is not None:
            return self._timestamps[i]
        return np.array([i / self._fps, (i + 1) / self._fps], dtype=np.float32)


def _make_mock_vframes(t=16, c=3, h=480, w=640, fps=24.0):
    frames = torch.randint(0, 256, (t, c, h, w), dtype=torch.uint8)
    return MockVFrames(frames=frames, fps=fps)


# =====================================================================
# CogVideoXProcessor._pad_last_frame
# =====================================================================
class TestPadLastFrame:

    def _get_processor(self):
        """Create CogVideoXProcessor without full init to test _pad_last_frame."""
        from mindspeed_mm.data.data_utils.video_processor import CogVideoXProcessor
        with patch.dict('os.environ', {
            'LOCAL_RANK': '0', 'RANK': '0', 'WORLD_SIZE': '1', 'LOCAL_WORLD_SIZE': '1'
        }):
            with patch('torch.distributed.init_process_group'), \
                 patch('torch.cuda.set_device'), \
                 patch('torch.cuda.current_device', return_value=0):
                proc = CogVideoXProcessor.__new__(CogVideoXProcessor)
                return proc

    def test_pad_short_tensor(self):
        proc = self._get_processor()
        tensor = torch.randn(5, 3, 64, 64)
        result = proc._pad_last_frame(tensor, 8)
        judge_expression(result.shape[0] == 8)
        # Padded frames should equal the last frame
        judge_expression(torch.equal(result[5], result[4]))
        judge_expression(torch.equal(result[7], result[4]))

    def test_no_pad_when_equal(self):
        proc = self._get_processor()
        tensor = torch.randn(8, 3, 64, 64)
        result = proc._pad_last_frame(tensor, 8)
        judge_expression(result.shape[0] == 8)
        judge_expression(torch.equal(result, tensor))

    def test_truncate_when_longer(self):
        proc = self._get_processor()
        tensor = torch.randn(12, 3, 64, 64)
        result = proc._pad_last_frame(tensor, 8)
        judge_expression(result.shape[0] == 8)
        judge_expression(torch.equal(result, tensor[:8]))


# =====================================================================
# CogVideoXProcessor._resize_for_rectangle_crop
# =====================================================================
class TestResizeForRectangleCrop:

    def _get_processor(self):
        from mindspeed_mm.data.data_utils.video_processor import CogVideoXProcessor
        with patch.dict('os.environ', {
            'LOCAL_RANK': '0', 'RANK': '0', 'WORLD_SIZE': '1', 'LOCAL_WORLD_SIZE': '1'
        }):
            with patch('torch.distributed.init_process_group'), \
                 patch('torch.cuda.set_device'), \
                 patch('torch.cuda.current_device', return_value=0):
                proc = CogVideoXProcessor.__new__(CogVideoXProcessor)
                return proc

    def test_center_crop_output_size(self):
        proc = self._get_processor()
        # T, C, H, W with large spatial dims
        arr = torch.randint(0, 256, (1, 16, 480, 640), dtype=torch.uint8).float()
        result = proc._resize_for_rectangle_crop(arr, [240, 320], reshape_mode="center")
        judge_expression(result.shape[-2:] == (240, 320))

    def test_center_crop_wide_input(self):
        proc = self._get_processor()
        arr = torch.randint(0, 256, (1, 16, 480, 960), dtype=torch.uint8).float()
        result = proc._resize_for_rectangle_crop(arr, [240, 320], reshape_mode="center")
        judge_expression(result.shape[-2:] == (240, 320))


# =====================================================================
# OpensoraplanVideoProcessor.find_closest_y
# =====================================================================
class TestFindClosestY:

    def _get_processor(self):
        from mindspeed_mm.data.data_utils.video_processor import OpensoraplanVideoProcessor
        with patch.dict('os.environ', {
            'LOCAL_RANK': '0', 'RANK': '0', 'WORLD_SIZE': '1', 'LOCAL_WORLD_SIZE': '1'
        }):
            with patch('torch.distributed.init_process_group'), \
                 patch('torch.cuda.set_device'), \
                 patch('torch.cuda.current_device', return_value=0), \
                 patch('mindspeed_mm.utils.dpcp_utils.get_max_cp_size', return_value=1), \
                 patch('mindspeed_mm.data.data_utils.utils.get_value_from_args', return_value=1), \
                 patch('mindspeed_mm.data.data_utils.utils.cal_gradient_accumulation_size', return_value=1), \
                 patch('mindspeed_mm.data.data_utils.transform_pipeline.get_transforms', return_value=lambda x: x):
                proc = OpensoraplanVideoProcessor.__new__(OpensoraplanVideoProcessor)
                proc.min_num_frames = 29
                proc.ae_stride_t = 4
                proc.sp_size = 1
                return proc

    def test_valid_y_returned(self):
        proc = self._get_processor()
        y = proc.find_closest_y(33, vae_stride_t=4, model_ds_t=1)
        judge_expression(y != -1)
        judge_expression((y - 1) % 4 == 0)

    def test_too_short_returns_minus1(self):
        proc = self._get_processor()
        y = proc.find_closest_y(10, vae_stride_t=4, model_ds_t=1)
        judge_expression(y == -1)

    def test_y_always_satisfies_vae_stride(self):
        proc = self._get_processor()
        for x in [29, 40, 50, 60, 100, 200]:
            y = proc.find_closest_y(x, vae_stride_t=4, model_ds_t=1)
            if y != -1:
                judge_expression((y - 1) % 4 == 0)
                judge_expression(y <= x)
                judge_expression(y >= 29)

    def test_model_ds_t_4(self):
        proc = self._get_processor()
        y = proc.find_closest_y(61, vae_stride_t=4, model_ds_t=4)
        if y != -1:
            # (y-1)//4 + 1 must be divisible by 4
            judge_expression(((y - 1) // 4 + 1) % 4 == 0)


# =====================================================================
# OpensoraplanVideoProcessor._validate_caption
# =====================================================================
class TestValidateCaption:

    def _get_processor(self):
        from mindspeed_mm.data.data_utils.video_processor import OpensoraplanVideoProcessor
        with patch.dict('os.environ', {
            'LOCAL_RANK': '0', 'RANK': '0', 'WORLD_SIZE': '1', 'LOCAL_WORLD_SIZE': '1'
        }):
            with patch('torch.distributed.init_process_group'), \
                 patch('torch.cuda.set_device'), \
                 patch('torch.cuda.current_device', return_value=0), \
                 patch('mindspeed_mm.utils.dpcp_utils.get_max_cp_size', return_value=1), \
                 patch('mindspeed_mm.data.data_utils.utils.get_value_from_args', return_value=1), \
                 patch('mindspeed_mm.data.data_utils.utils.cal_gradient_accumulation_size', return_value=1), \
                 patch('mindspeed_mm.data.data_utils.transform_pipeline.get_transforms', return_value=lambda x: x):
                proc = OpensoraplanVideoProcessor.__new__(OpensoraplanVideoProcessor)
                return proc

    def test_valid_caption(self):
        proc = self._get_processor()
        from mindspeed_mm.data.data_utils.utils import DataStats
        stats = DataStats()
        sample = {"cap": "A cat playing piano"}
        judge_expression(proc._validate_caption(sample, stats) is True)

    def test_missing_caption(self):
        proc = self._get_processor()
        from mindspeed_mm.data.data_utils.utils import DataStats
        stats = DataStats()
        sample = {}
        judge_expression(proc._validate_caption(sample, stats) is False)

    def test_none_caption(self):
        proc = self._get_processor()
        from mindspeed_mm.data.data_utils.utils import DataStats
        stats = DataStats()
        sample = {"cap": None}
        judge_expression(proc._validate_caption(sample, stats) is False)


# =====================================================================
# OpensoraplanVideoProcessor._filter_resolution
# =====================================================================
class TestFilterResolution:

    def _get_processor(self):
        from mindspeed_mm.data.data_utils.video_processor import OpensoraplanVideoProcessor
        with patch.dict('os.environ', {
            'LOCAL_RANK': '0', 'RANK': '0', 'WORLD_SIZE': '1', 'LOCAL_WORLD_SIZE': '1'
        }):
            with patch('torch.distributed.init_process_group'), \
                 patch('torch.cuda.set_device'), \
                 patch('torch.cuda.current_device', return_value=0), \
                 patch('mindspeed_mm.utils.dpcp_utils.get_max_cp_size', return_value=1), \
                 patch('mindspeed_mm.data.data_utils.utils.get_value_from_args', return_value=1), \
                 patch('mindspeed_mm.data.data_utils.utils.cal_gradient_accumulation_size', return_value=1), \
                 patch('mindspeed_mm.data.data_utils.transform_pipeline.get_transforms', return_value=lambda x: x):
                proc = OpensoraplanVideoProcessor.__new__(OpensoraplanVideoProcessor)
                return proc

    def test_valid_aspect_ratio(self):
        proc = self._get_processor()
        judge_expression(proc._filter_resolution(480, 640, max_h_div_w_ratio=17 / 16, min_h_div_w_ratio=8 / 16) is True)

    def test_too_tall(self):
        proc = self._get_processor()
        judge_expression(proc._filter_resolution(800, 200, max_h_div_w_ratio=17 / 16, min_h_div_w_ratio=8 / 16) is False)

    def test_too_wide(self):
        proc = self._get_processor()
        judge_expression(proc._filter_resolution(200, 800, max_h_div_w_ratio=17 / 16, min_h_div_w_ratio=8 / 16) is False)

    def test_square_passes(self):
        proc = self._get_processor()
        judge_expression(proc._filter_resolution(640, 640, max_h_div_w_ratio=17 / 16, min_h_div_w_ratio=8 / 16) is True)


# =====================================================================
# RewardVideoProcessor: round_by_factor / ceil_by_factor / floor_by_factor
# =====================================================================
class TestRewardMathUtils:

    def _get_processor(self):
        from mindspeed_mm.data.data_utils.video_processor import RewardVideoProcessor
        with patch.dict('os.environ', {
            'LOCAL_RANK': '0', 'RANK': '0', 'WORLD_SIZE': '1', 'LOCAL_WORLD_SIZE': '1'
        }):
            with patch('torch.distributed.init_process_group'), \
                 patch('torch.cuda.set_device'), \
                 patch('torch.cuda.current_device', return_value=0):
                proc = RewardVideoProcessor.__new__(RewardVideoProcessor)
                return proc

    def test_round_by_factor(self):
        proc = self._get_processor()
        # round(10/4)*4 = round(2.5)*4 = 2*4 = 8 (banker's rounding)
        judge_expression(proc.round_by_factor(10, 4) == 8)
        judge_expression(proc.round_by_factor(12, 4) == 12)
        judge_expression(proc.round_by_factor(7, 2) == 8)
        # round(1/2)*2 = round(0.5)*2 = 0*2 = 0 (banker's rounding)
        judge_expression(proc.round_by_factor(1, 2) == 0)
        judge_expression(proc.round_by_factor(8, 4) == 8)

    def test_ceil_by_factor(self):
        proc = self._get_processor()
        judge_expression(proc.ceil_by_factor(9, 4) == 12)
        judge_expression(proc.ceil_by_factor(12, 4) == 12)
        judge_expression(proc.ceil_by_factor(13, 4) == 16)

    def test_floor_by_factor(self):
        proc = self._get_processor()
        judge_expression(proc.floor_by_factor(9, 4) == 8)
        judge_expression(proc.floor_by_factor(12, 4) == 12)
        judge_expression(proc.floor_by_factor(15, 4) == 12)


# =====================================================================
# RewardVideoProcessor.get_sample_nframes
# =====================================================================
class TestGetSampleNframes:

    def _get_processor(self, **kwargs):
        from mindspeed_mm.data.data_utils.video_processor import RewardVideoProcessor
        with patch.dict('os.environ', {
            'LOCAL_RANK': '0', 'RANK': '0', 'WORLD_SIZE': '1', 'LOCAL_WORLD_SIZE': '1'
        }):
            with patch('torch.distributed.init_process_group'), \
                 patch('torch.cuda.set_device'), \
                 patch('torch.cuda.current_device', return_value=0):
                proc = RewardVideoProcessor.__new__(RewardVideoProcessor)
                proc.sample_nframe = kwargs.get("sample_nframe", None)
                proc.fps = kwargs.get("fps", 2.0)
                proc.min_frames = kwargs.get("min_frames", 4)
                proc.max_frames = kwargs.get("max_frames", 768)
                proc.frame_factor = kwargs.get("frame_factor", 2)
                return proc

    def test_fixed_sample_nframe(self):
        proc = self._get_processor(sample_nframe=16)
        result = proc.get_sample_nframes(total_frames=100, video_fps=24)
        judge_expression(result <= 100)
        judge_expression(result % 2 == 0)

    def test_sample_nframe_exceeds_total(self):
        proc = self._get_processor(sample_nframe=200)
        result = proc.get_sample_nframes(total_frames=50, video_fps=24)
        judge_expression(result <= 50)

    def test_auto_nframes(self):
        proc = self._get_processor(fps=2.0, min_frames=4, max_frames=768)
        result = proc.get_sample_nframes(total_frames=100, video_fps=24)
        judge_expression(result % 2 == 0)
        judge_expression(result >= 4)
        judge_expression(result <= 100)

    def test_nframes_divisible_by_factor(self):
        proc = self._get_processor(sample_nframe=10, frame_factor=2)
        result = proc.get_sample_nframes(total_frames=50, video_fps=24)
        judge_expression(result % 2 == 0)


# =====================================================================
# RewardVideoProcessor.get_frame_size
# =====================================================================
class TestGetFrameSize:

    def _get_processor(self, **kwargs):
        from mindspeed_mm.data.data_utils.video_processor import RewardVideoProcessor
        with patch.dict('os.environ', {
            'LOCAL_RANK': '0', 'RANK': '0', 'WORLD_SIZE': '1', 'LOCAL_WORLD_SIZE': '1'
        }):
            with patch('torch.distributed.init_process_group'), \
                 patch('torch.cuda.set_device'), \
                 patch('torch.cuda.current_device', return_value=0):
                proc = RewardVideoProcessor.__new__(RewardVideoProcessor)
                proc.video_min_pixels = kwargs.get("video_min_pixels", 100352)
                proc.video_max_pixels = kwargs.get("video_max_pixels", None)
                proc.resized_height = kwargs.get("resized_height", None)
                proc.resized_width = kwargs.get("resized_width", None)
                proc.image_factor = kwargs.get("image_factor", 28)
                proc.frame_factor = kwargs.get("frame_factor", 2)
                return proc

    def test_normal_size(self):
        proc = self._get_processor()
        rh, rw = proc.get_frame_size(nframes=16, height=480, width=640)
        judge_expression(rh % 28 == 0)
        judge_expression(rw % 28 == 0)

    def test_large_image_resized_down(self):
        proc = self._get_processor()
        rh, rw = proc.get_frame_size(nframes=16, height=2000, width=3000)
        judge_expression(rh * rw <= 768 * 28 * 28)

    def test_small_image_resized_up(self):
        proc = self._get_processor(video_min_pixels=100352)
        rh, rw = proc.get_frame_size(nframes=16, height=28, width=28)
        judge_expression(rh >= 28 and rw >= 28)

    def test_resized_height_width_override(self):
        proc = self._get_processor(resized_height=224, resized_width=224)
        rh, rw = proc.get_frame_size(nframes=16, height=480, width=640)
        judge_expression(rh == 224)
        judge_expression(rw == 224)

    def test_output_divisible_by_image_factor(self):
        proc = self._get_processor()
        rh, rw = proc.get_frame_size(nframes=16, height=333, width=555)
        judge_expression(rh % 28 == 0)
        judge_expression(rw % 28 == 0)


# =====================================================================
# OpensoraplanVideoProcessor._process_resolution (dynamic resolution)
# =====================================================================
class TestProcessResolution:

    def _get_processor(self):
        from mindspeed_mm.data.data_utils.video_processor import OpensoraplanVideoProcessor
        with patch.dict('os.environ', {
            'LOCAL_RANK': '0', 'RANK': '0', 'WORLD_SIZE': '1', 'LOCAL_WORLD_SIZE': '1'
        }):
            with patch('torch.distributed.init_process_group'), \
                 patch('torch.cuda.set_device'), \
                 patch('torch.cuda.current_device', return_value=0), \
                 patch('mindspeed_mm.utils.dpcp_utils.get_max_cp_size', return_value=1), \
                 patch('mindspeed_mm.data.data_utils.utils.get_value_from_args', return_value=1), \
                 patch('mindspeed_mm.data.data_utils.utils.cal_gradient_accumulation_size', return_value=1), \
                 patch('mindspeed_mm.data.data_utils.transform_pipeline.get_transforms', return_value=lambda x: x):
                proc = OpensoraplanVideoProcessor.__new__(OpensoraplanVideoProcessor)
                proc.force_resolution = False
                proc.max_hxw = 480 * 832
                proc.min_hxw = 480 * 832 // 4
                proc.hw_stride = 32
                proc.hw_aspect_thr = 1.5
                proc.max_height = 480
                proc.max_width = 640
                return proc

    def test_valid_dynamic_resolution(self):
        proc = self._get_processor()
        from mindspeed_mm.data.data_utils.utils import DataStats
        stats = DataStats()
        sample = {"resolution": {"height": 480, "width": 640}}
        result = proc._process_resolution(sample, stats)
        judge_expression(result is True)
        judge_expression("sample_height" in sample["resolution"])
        judge_expression("sample_width" in sample["resolution"])
        judge_expression(sample["resolution"]["sample_height"] % 32 == 0)
        judge_expression(sample["resolution"]["sample_width"] % 32 == 0)

    def test_missing_resolution(self):
        proc = self._get_processor()
        from mindspeed_mm.data.data_utils.utils import DataStats
        stats = DataStats()
        sample = {"resolution": {}}
        result = proc._process_resolution(sample, stats)
        judge_expression(result is False)

    def test_negative_resolution(self):
        proc = self._get_processor()
        from mindspeed_mm.data.data_utils.utils import DataStats
        stats = DataStats()
        sample = {"resolution": {"height": -1, "width": 640}}
        result = proc._process_resolution(sample, stats)
        judge_expression(result is False)


# =====================================================================
# VideoProcessor.create (factory pattern with Registry)
# =====================================================================
class TestVideoProcessorCreate:

    def test_create_with_registry(self):
        from mindspeed_mm.data.data_utils.video_processor import VideoProcessor
        mock_cls = MagicMock(return_value=MagicMock())
        with patch('mindspeed_mm.utils.utils.Registry.get_class', return_value=mock_cls):
            result = VideoProcessor.create(video_processor_type="opensora_video_processor", num_frames=16)
            mock_cls.assert_called_once_with(num_frames=16)
