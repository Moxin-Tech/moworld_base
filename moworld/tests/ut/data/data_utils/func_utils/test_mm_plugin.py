import unittest
from typing import BinaryIO
from unittest.mock import patch, MagicMock

import numpy as np
import torch
from PIL import Image
from PIL.Image import Image as ImageObject

from mindspeed_mm.data.data_utils.func_utils.mm_plugin import (
    get_mm_plugin,
    BasePlugin,
    Qwen2VLPlugin,
    Qwen2OmniPlugin,
    Qwen3VLPlugin,
    Qwen3OmniPlugin,
    GLM4VPlugin,
    _make_batched_images,
    _check_video_is_nested_images,
    MMPluginMixin,
)
from tests.ut.utils import judge_expression


class TestUtilityFunctions(unittest.TestCase):
    def test_make_batched_images(self):
        """Test _make_batched_images function"""
        # Setup
        mock_images = [MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        imglens = [2, 3]

        # Execute
        result = _make_batched_images(mock_images, imglens)

        # Assert
        judge_expression(len(result) == 2)
        judge_expression(len(result[0]) == 2)
        judge_expression(len(result[1]) == 3)

    def test_check_video_is_nested_images_true(self):
        """Test _check_video_is_nested_images returns True for nested images"""
        # Setup
        mock_video = [MagicMock(spec=str), MagicMock(spec=BinaryIO), MagicMock(spec=dict), MagicMock(spec=ImageObject)]

        # Execute
        result = _check_video_is_nested_images(mock_video)

        # Assert
        judge_expression(result is True)

    def test_check_video_is_nested_images_false(self):
        """Test _check_video_is_nested_images returns False for non-nested images"""
        # Setup
        mock_video = "not_a_list"

        # Execute
        result = _check_video_is_nested_images(mock_video)

        # Assert
        judge_expression(result is False)


class TestGetMMPlugin(unittest.TestCase):
    def test_get_mm_plugin_valid_names(self):
        """Test get_mm_plugin with valid plugin names"""
        # Execute & Assert
        base_plugin = get_mm_plugin("base")
        judge_expression(isinstance(base_plugin, BasePlugin))

        qwen2_vl_plugin = get_mm_plugin("qwen2_vl")
        judge_expression(isinstance(qwen2_vl_plugin, Qwen2VLPlugin))

        qwen2_omni_plugin = get_mm_plugin("qwen2_omni")
        judge_expression(isinstance(qwen2_omni_plugin, Qwen2OmniPlugin))

        qwen3_vl_plugin = get_mm_plugin("qwen3_vl")
        judge_expression(isinstance(qwen3_vl_plugin, Qwen3VLPlugin))

        qwen3_omni_plugin = get_mm_plugin("qwen3_omni")
        judge_expression(isinstance(qwen3_omni_plugin, Qwen3OmniPlugin))

        glm4v_plugin = get_mm_plugin("glm4.1v")
        judge_expression(isinstance(glm4v_plugin, GLM4VPlugin))

    def test_get_mm_plugin_invalid_name(self):
        """Test get_mm_plugin with invalid plugin name raises ValueError"""
        # Execute & Assert
        try:
            get_mm_plugin("invalid_plugin")
            judge_expression(False)  # Should not reach here
        except ValueError:
            judge_expression(True)  # Expected exception


class TestMMPluginMixin(unittest.TestCase):
    def setUp(self):
        """Set up MMPluginMixin instance for testing"""
        self.mixin = MMPluginMixin(
            image_token="<image>", video_token="<video>", audio_token="<audio>"
        )

    def test_mm_plugin_mixin_initialization(self):
        """Test MMPluginMixin initialization"""
        # Assert
        judge_expression(self.mixin.image_token == "<image>")
        judge_expression(self.mixin.video_token == "<video>")
        judge_expression(self.mixin.audio_token == "<audio>")
        judge_expression(self.mixin.expand_mm_tokens is True)

    def test_mm_plugin_mixin_validate_input_image_without_image_processor(self):
        """Test MMPluginMixin._validate_input with image token but no image processor raises ValueError"""
        # Setup
        mock_processor = MagicMock()
        mock_processor.image_processor = None

        # Execute & Assert
        try:
            self.mixin._validate_input(mock_processor, ["image1"], [], [])
            judge_expression(False)  # Should not reach here
        except ValueError as e:
            judge_expression("Image processor was not found" in str(e))

    def test_mm_plugin_mixin_validate_input_video_without_video_processor(self):
        """Test MMPluginMixin._validate_input with video token but no video processor raises ValueError"""
        # Setup
        mock_processor = MagicMock()

        # Execute & Assert
        with patch('mindspeed_mm.data.data_utils.func_utils.mm_plugin.get_video_processor', return_value=None):
            try:
                self.mixin._validate_input(mock_processor, [], ["video1"], [])
                judge_expression(False)  # Should not reach here
            except ValueError as e:
                judge_expression("Video processor was not found" in str(e))

    def test_mm_plugin_mixin_validate_input_audio_without_feature_extractor(self):
        """Test MMPluginMixin._validate_input with audio token but no feature extractor raises ValueError"""
        # Setup
        mock_processor = MagicMock()
        mock_processor.feature_extractor = None

        # Execute & Assert
        try:
            self.mixin._validate_input(mock_processor, [], [], ["audio1"])
            judge_expression(False)  # Should not reach here
        except ValueError as e:
            judge_expression("Audio feature extractor was not found" in str(e))

    def test_mm_plugin_mixin_validate_input_all_processors_present(self):
        """Test MMPluginMixin._validate_input with all processors present - should pass without exceptions"""
        # Setup
        mock_processor = MagicMock()
        mock_processor.image_processor = MagicMock()
        mock_processor.feature_extractor = MagicMock()

        # Execute & Assert
        with patch('mindspeed_mm.data.data_utils.func_utils.mm_plugin.get_video_processor', return_value=MagicMock()):
            try:
                self.mixin._validate_input(mock_processor, ["image1"], ["video1"], ["audio1"])
                judge_expression(True)  # Should pass without exception
            except Exception as e:
                judge_expression(False)  # Should not reach here

    def test_mm_plugin_mixin_validate_messages_matching_counts(self):
        """Test MMPluginMixin._validate_messages with matching counts"""
        # Setup
        messages = [
            {"content": "This is an <image> and another <image>"},
            {"content": "This is a <video>"},
            {"content": "This is an <audio>"},
        ]
        images = ["img1", "img2"]
        videos = ["vid1"]
        audios = ["aud1"]

        # Execute
        MMPluginMixin._validate_messages(messages, images, videos, audios)

        # Assert - should not raise any exception
        judge_expression(True)

    def test_mm_plugin_mixin_validate_messages_mismatch_images(self):
        """Test MMPluginMixin._validate_messages with mismatched image counts"""
        # Setup
        messages = [{"content": "This is an <image> and another <image>"}]
        images = ["img1"]  # Only one image but two placeholders

        # Execute & Assert
        try:
            MMPluginMixin._validate_messages(messages, images, [], [])
            judge_expression(False)  # Should not reach here
        except ValueError as e:
            judge_expression("number of images does not match" in str(e))

    def test_mm_plugin_mixin_validate_messages_mismatch_videos(self):
        """Test MMPluginMixin._validate_messages with mismatched video counts"""
        # Setup
        messages = [{"content": "This is a <video> and another <video>"}]
        videos = ["vid1"]  # Only one video but two placeholders

        # Execute & Assert
        try:
            MMPluginMixin._validate_messages(messages, [], videos, [])
            judge_expression(False)  # Should not reach here
        except ValueError as e:
            judge_expression("number of videos does not match" in str(e))

    def test_mm_plugin_mixin_validate_messages_mismatch_audios(self):
        """Test MMPluginMixin._validate_messages with mismatched audio counts"""
        # Setup
        messages = [{"content": "This is an <audio> and another <audio>"}]
        audios = ["aud1"]  # Only one audio but two placeholders

        # Execute & Assert
        try:
            MMPluginMixin._validate_messages(messages, [], [], audios)
            judge_expression(False)  # Should not reach here
        except ValueError as e:
            judge_expression("number of audios does not match" in str(e))

    @patch("mindspeed_mm.data.data_utils.func_utils.mm_plugin.Image")
    def test_regularize_images_from_path(self, mock_image_module):
        """Test _regularize_images with image path"""
        # Setup
        mock_image_obj = MagicMock()
        mock_image_obj.width = 100
        mock_image_obj.height = 100
        mock_image_obj.mode = "RGB"
        mock_image_obj.__class__ = Image.Image

        mock_image_module.open.return_value.__enter__.return_value = mock_image_obj

        images = ["image_path.jpg"]

        # Execute
        result = self.mixin._regularize_images(
            images, image_max_pixels=768 * 768, image_min_pixels=32 * 32
        )

        # Assert
        judge_expression("images" in result)
        judge_expression(len(result["images"]) == 1)

    @patch("mindspeed_mm.data.data_utils.func_utils.mm_plugin.BytesIO")
    @patch("mindspeed_mm.data.data_utils.func_utils.mm_plugin.Image")
    def test_regularize_images_from_bytes(self, mock_image_module, mock_bytes_io):
        """Test _regularize_images with image bytes"""
        # Setup
        mock_image_obj = MagicMock()
        mock_image_obj.width = 100
        mock_image_obj.height = 100
        mock_image_obj.mode = "RGB"
        mock_image_obj.__class__ = Image.Image

        mock_image_module.open.return_value.__enter__.return_value = mock_image_obj

        images = [b"image_bytes"]

        # Execute
        result = self.mixin._regularize_images(
            images, image_max_pixels=768 * 768, image_min_pixels=32 * 32
        )

        # Assert
        judge_expression("images" in result)
        judge_expression(len(result["images"]) == 1)

    @patch("mindspeed_mm.data.data_utils.func_utils.mm_plugin.Image")
    def test_regularize_images_from_object(self, mock_image_module):
        """Test _regularize_images with ImageObject"""
        # Setup
        mock_image_obj = MagicMock()
        mock_image_obj.width = 100
        mock_image_obj.height = 100
        mock_image_obj.mode = "RGB"
        mock_image_obj.__class__ = Image.Image

        images = [mock_image_obj]

        # Execute
        result = self.mixin._regularize_images(
            images, image_max_pixels=768 * 768, image_min_pixels=32 * 32
        )

        # Assert
        judge_expression("images" in result)
        judge_expression(len(result["images"]) == 1)

    @patch("mindspeed_mm.data.data_utils.func_utils.mm_plugin.Image")
    def test_regularize_images_invalid_input(self, mock_image_module):
        """Test _regularize_images with invalid input raises ValueError"""
        # Setup
        images = [123]  # Invalid type

        # Execute & Assert
        try:
            self.mixin._regularize_images(
                images, image_max_pixels=768 * 768, image_min_pixels=32 * 32
            )
            judge_expression(False)  # Should not reach here
        except ValueError as e:
            judge_expression("Expect input is a list of images" in str(e))


class TestBasePlugin(unittest.TestCase):
    def setUp(self):
        """Set up BasePlugin instance for testing"""
        self.plugin = BasePlugin(
            image_token="<image>", video_token="<video>", audio_token="<audio>"
        )

    def test_base_plugin_initialization(self):
        """Test BasePlugin initialization"""
        # Assert
        judge_expression(isinstance(self.plugin, BasePlugin))
        judge_expression(isinstance(self.plugin, MMPluginMixin))

    def test_base_plugin_process_messages(self):
        """Test BasePlugin.process_messages"""
        # Setup
        messages = [{"content": "Hello world"}]
        images = []
        videos = []
        audios = []
        mock_processor = MagicMock()

        # Execute
        result = self.plugin.process_messages(
            messages, images, videos, audios, mock_processor
        )

        # Assert
        judge_expression(result == messages)

    def test_base_plugin_process_token_ids(self):
        """Test BasePlugin.process_token_ids"""
        # Setup
        input_ids = [1, 2, 3]
        labels = [1, 2, 3]
        images = []
        videos = []
        audios = []
        mock_tokenizer = MagicMock()
        mock_processor = MagicMock()

        # Execute
        result_ids, result_labels = self.plugin.process_token_ids(
            input_ids, labels, images, videos, audios, mock_tokenizer, mock_processor
        )

        # Assert
        judge_expression(result_ids == input_ids)
        judge_expression(result_labels == labels)

    def test_base_plugin_get_mm_inputs(self):
        """Test BasePlugin.get_mm_inputs"""
        # Setup
        images = []
        videos = []
        audios = []
        imglens = []
        vidlens = []
        audlens = []
        batch_ids = []
        mock_processor = MagicMock()

        # Execute
        result = self.plugin.get_mm_inputs(
            images, videos, audios, imglens, vidlens, audlens, batch_ids, mock_processor
        )

        # Assert
        judge_expression(isinstance(result, dict))


class TestGLM4VPlugin(unittest.TestCase):
    def setUp(self):
        self.plugin = GLM4VPlugin(image_token="<|image|>", video_token="<|video|>", audio_token=None)

    @patch("mindspeed_mm.data.data_utils.func_utils.mm_plugin.getattr")
    @patch.object(GLM4VPlugin, "_regularize_images")
    def test_get_mm_inputs_with_images_only(self, mock_regularize_images, mock_getattr):
        # Setup
        mock_image_processor = MagicMock()
        mock_image_processor.return_value = {
            "pixel_values": torch.tensor([1, 2, 3]),
            "image_grid_thw": torch.tensor([1, 2, 2])
        }
        mock_getattr.return_value = mock_image_processor

        mock_regularize_images.return_value = {
            "images": ["processed_image"]
        }

        images = ["image1"]
        videos = []
        audios = []
        mock_processor = MagicMock()

        # Execute
        result = self.plugin._get_mm_inputs(images, videos, audios, mock_processor)

        # Assert
        mock_regularize_images.assert_called_once()
        mock_image_processor.assert_called_once()
        judge_expression("pixel_values" in result)
        judge_expression(result["pixel_values"].tolist() == [1, 2, 3])

    @patch("mindspeed_mm.data.data_utils.func_utils.mm_plugin.getattr")
    @patch.object(GLM4VPlugin, "_regularize_videos")
    def test_get_mm_inputs_with_videos_only(self, mock_regularize_videos, mock_getattr):
        # Setup
        mock_video_processor = MagicMock()
        mock_video_processor.return_value = {
            "pixel_values": torch.tensor([4, 5, 6]),
            "video_grid_thw": torch.tensor([2, 2, 2]),
            "timestamps": [0, 1]
        }
        mock_getattr.return_value = mock_video_processor

        mock_regularize_videos.return_value = {
            "videos": [["frame1", "frame2"]]
        }

        images = []
        videos = ["video1"]
        audios = []
        mock_processor = MagicMock()

        # Execute
        result = self.plugin._get_mm_inputs(images, videos, audios, mock_processor)

        # Assert
        mock_regularize_videos.assert_called_once()
        mock_video_processor.assert_called_once()
        judge_expression("pixel_values" in result)
        judge_expression(result["pixel_values"].tolist() == [4, 5, 6])
        judge_expression("video_grid_thw" in result)
        judge_expression("timestamps" in result)

    @patch("mindspeed_mm.data.data_utils.func_utils.mm_plugin.deepcopy")
    @patch.object(GLM4VPlugin, "_get_mm_inputs")
    @patch("mindspeed_mm.data.data_utils.func_utils.mm_plugin.getattr")
    def test_process_messages_without_expand_tokens(self, mock_getattr, mock_get_mm_inputs, mock_deepcopy):
        # Setup
        self.plugin.expand_mm_tokens = False
        mock_image_processor = MagicMock()
        mock_getattr.return_value = mock_image_processor

        mock_deepcopy.return_value = [
            {"content": "This is an image <image> and a video <video>"}
        ]

        messages = [{"content": "This is an image <image> and a video <video>"}]
        images = ["image1"]
        videos = ["video1"]
        audios = []
        mock_processor = MagicMock()
        type(mock_processor).image_processor = mock_image_processor
        mock_image_processor.merge_size = 2

        # Execute
        result = self.plugin.process_messages(messages, images, videos, audios, mock_processor)

        # Assert
        judge_expression(isinstance(result, list))
        judge_expression("content" in result[0])
        judge_expression("<|begin_of_image|>" in result[0]["content"])
        judge_expression("<|end_of_image|>" in result[0]["content"])
        judge_expression("<|begin_of_video|>" in result[0]["content"])
        judge_expression("<|end_of_video|>" in result[0]["content"])

        # Reset
        self.plugin.expand_mm_tokens = True

    @patch("mindspeed_mm.data.data_utils.func_utils.mm_plugin.getattr")
    def test_get_mm_inputs_without_images_and_videos(self, mock_getattr):
        # Setup
        images = []
        videos = []
        audios = []
        mock_processor = MagicMock()

        # Execute
        result = self.plugin._get_mm_inputs(images, videos, audios, mock_processor)

        # Assert
        judge_expression(isinstance(result, dict))
        judge_expression(len(result) == 0)  # Should be empty dict

    def test_get_mm_inputs(self):
        # Setup
        images = []
        videos = []
        audios = []
        imglens = []
        vidlens = []
        audlens = []
        batch_ids = []
        mock_processor = MagicMock()

        # Execute
        result = self.plugin.get_mm_inputs(
            images, videos, audios, imglens, vidlens, audlens, batch_ids, mock_processor
        )

        # Assert
        judge_expression(isinstance(result, dict))

    @patch.object(GLM4VPlugin, "_regularize_images")
    def test_get_mm_inputs_image_processing_failure(self, mock_regularize_images):
        # Setup
        mock_regularize_images.side_effect = ValueError("Image processing failed")
        images = ["invalid_image"]
        videos = []
        audios = []
        mock_processor = MagicMock()

        # Execute
        result = None
        exception_raised = False
        try:
            result = self.plugin._get_mm_inputs(images, videos, audios, mock_processor)
        except ValueError:
            exception_raised = True

        # Assert
        judge_expression(result is None)
        judge_expression(exception_raised is True)

    @patch("mindspeed_mm.data.data_utils.func_utils.mm_plugin.getattr")
    @patch.object(GLM4VPlugin, "_regularize_videos")
    def test_get_mm_inputs_video_processing_failure(self, mock_regularize_videos, mock_getattr):
        # Setup
        mock_regularize_videos.side_effect = ValueError("Video processing failed")
        images = []
        videos = ["invalid_video"]
        audios = []
        mock_processor = MagicMock()

        # Execute
        result = None
        exception_raised = False
        try:
            result = self.plugin._get_mm_inputs(images, videos, audios, mock_processor)
        except ValueError:
            exception_raised = True

        # Assert
        judge_expression(result is None)
        judge_expression(exception_raised is True)

    @patch("mindspeed_mm.data.data_utils.func_utils.mm_plugin.getattr")
    @patch.object(GLM4VPlugin, "_regularize_images")
    @patch.object(GLM4VPlugin, "_regularize_videos")
    def test_get_mm_inputs_with_images_and_videos(self, mock_regularize_videos, mock_regularize_images, mock_getattr):
        # Setup
        mock_image_processor = MagicMock()
        mock_image_processor.return_value = {
            "pixel_values": torch.tensor([1, 2, 3]),
            "image_grid_thw": torch.tensor([1, 2, 2])
        }

        mock_video_processor = MagicMock()
        mock_video_processor.return_value = {
            "pixel_values_videos": torch.tensor([4, 5, 6]),
            "video_grid_thw": torch.tensor([2, 2, 2])
        }

        def getattr_side_effect(obj, name, default=None):
            if name == "image_processor":
                return mock_image_processor
            elif name == "video_processor":
                return mock_video_processor
            return default

        mock_getattr.side_effect = getattr_side_effect
        mock_regularize_images.return_value = {"images": ["processed_image"]}
        mock_regularize_videos.return_value = {"videos": [["frame1", "frame2"]]}

        images = ["image1"]
        videos = ["video1"]
        audios = []
        mock_processor = MagicMock()

        # Execute
        result = self.plugin._get_mm_inputs(images, videos, audios, mock_processor)

        # Assert
        mock_regularize_images.assert_called_once()
        mock_regularize_videos.assert_called_once()
        mock_image_processor.assert_called_once()
        mock_video_processor.assert_called_once()
        judge_expression("pixel_values" in result)
        judge_expression("pixel_values_videos" in result)
        judge_expression(len(result["pixel_values"]) == 3)
        judge_expression(len(result["pixel_values_videos"]) == 3)
        judge_expression("image_grid_thw" in result)
        judge_expression("video_grid_thw" in result)

    @patch("mindspeed_mm.data.data_utils.func_utils.mm_plugin.deepcopy")
    @patch.object(GLM4VPlugin, "_get_mm_inputs")
    @patch("mindspeed_mm.data.data_utils.func_utils.mm_plugin.getattr")
    def test_process_messages_with_images_and_videos(self, mock_getattr, mock_get_mm_inputs, mock_deepcopy):
        # Setup
        mock_image_processor = MagicMock()
        mock_getattr.return_value = mock_image_processor

        mock_get_mm_inputs.return_value = {
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
            "video_grid_thw": torch.tensor([[2, 2, 2]]),
            "timestamps": torch.tensor([0, 1])
        }

        mock_deepcopy.return_value = [
            {"content": "This is an image <image> and a video <video>"}
        ]

        messages = [{"content": "This is an image <image> and a video <video>"}]
        images = ["image1"]
        videos = ["video1"]
        audios = []
        mock_processor = MagicMock()
        type(mock_processor).image_processor = mock_image_processor
        mock_image_processor.merge_size = 2

        # Execute
        result = self.plugin.process_messages(messages, images, videos, audios, mock_processor)

        # Assert
        judge_expression(isinstance(result, list))
        judge_expression("content" in result[0])
        judge_expression("<|begin_of_image|>" in result[0]["content"])
        judge_expression("<|end_of_image|>" in result[0]["content"])
        judge_expression("<|begin_of_video|>" in result[0]["content"])
        judge_expression("<|end_of_video|>" in result[0]["content"])
