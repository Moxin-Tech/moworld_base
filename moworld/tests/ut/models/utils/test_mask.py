"""
Unit tests for mask_utils.py module.
Tests all mask generators and mask processor functionality.
"""

import torch
import random
from unittest.mock import patch, MagicMock
from abc import ABC, abstractmethod

from tests.ut.utils import judge_expression, TestConfig
from mindspeed_mm.utils.mask_utils import (
    MaskType,
    TYPE_TO_STR,
    STR_TO_TYPE,
    BaseMaskGenerator,
    T2IVMaskGenerator,
    I2VMaskGenerator,
    TransitionMaskGenerator,
    ContinuationMaskGenerator,
    ClearMaskGenerator,
    RandomTemporalMaskGenerator,
    MaskProcessor,
    MaskCompressor
)


class TestMaskType:
    """Test MaskType enum and related dictionaries."""

    def test_mask_type_enum(self):
        """Test that all expected mask types are present in the enum."""
        expected_types = [
            "t2iv", "i2v", "transition",
            "continuation", "clear", "random_temporal"
        ]

        for mask_type in MaskType:
            judge_expression(mask_type.name in expected_types)

        judge_expression(len(MaskType) == len(expected_types))

    def test_type_to_str(self):
        """Test TYPE_TO_STR dictionary mapping."""
        for mask_type in MaskType:
            judge_expression(TYPE_TO_STR[mask_type] == mask_type.name)

    def test_str_to_type(self):
        """Test STR_TO_TYPE dictionary mapping."""
        for mask_type in MaskType:
            judge_expression(STR_TO_TYPE[mask_type.name] == mask_type)


# 创建一个具体子类用于测试BaseMaskGenerator的非抽象方法
class ConcreteMaskGenerator(BaseMaskGenerator):
    """Concrete implementation of BaseMaskGenerator for testing."""

    def process(self, mask):
        """Implement abstract process method."""
        return mask


class TestBaseMaskGenerator:
    """Test BaseMaskGenerator abstract base class."""

    def test_create_system_mask(self):
        """Test create_system_mask method."""
        # 使用具体子类而不是直接实例化抽象基类
        generator = ConcreteMaskGenerator()

        # Test normal case
        num_frames, height, width = 4, 32, 32
        device = "cpu"
        dtype = torch.float32

        mask = generator.create_system_mask(num_frames, height, width, device, dtype)

        judge_expression(mask.shape == (num_frames, 1, height, width))
        judge_expression(mask.device.type == device)
        judge_expression(mask.dtype == dtype)
        judge_expression(torch.all(mask == 1.0))

        # Test with None values (should raise ValueError)
        try:
            generator.create_system_mask(None, height, width, device, dtype)
            judge_expression(False)  # Should not reach here
        except ValueError:
            pass

        try:
            generator.create_system_mask(num_frames, None, width, device, dtype)
            judge_expression(False)  # Should not reach here
        except ValueError:
            pass

        try:
            generator.create_system_mask(num_frames, height, None, device, dtype)
            judge_expression(False)  # Should not reach here
        except ValueError:
            pass

    def test_abstract_method_enforcement(self):
        """Test that BaseMaskGenerator cannot be instantiated directly."""
        try:
            # 这应该会抛出TypeError
            BaseMaskGenerator()
            judge_expression(False)  # 如果没有抛出异常，测试失败
        except TypeError as e:
            # 验证错误信息包含预期内容
            judge_expression("abstract class" in str(e).lower())
            judge_expression("process" in str(e).lower())


class TestT2IVMaskGenerator:
    """Test T2IVMaskGenerator class."""

    def test_process(self):
        """Test process method."""
        generator = T2IVMaskGenerator()

        # Create a test mask
        num_frames, height, width = 4, 32, 32
        mask = torch.ones(num_frames, 1, height, width)

        processed_mask = generator.process(mask)

        # T2IV mask should be all 1s
        judge_expression(torch.all(processed_mask == 1.0))
        judge_expression(processed_mask.shape == mask.shape)

    def test_call(self):
        """Test __call__ method."""
        generator = T2IVMaskGenerator()

        num_frames, height, width = 4, 32, 32
        mask = generator(num_frames, height, width, device="cpu")

        judge_expression(mask.shape == (num_frames, 1, height, width))
        judge_expression(torch.all(mask == 1.0))


class TestI2VMaskGenerator:
    """Test I2VMaskGenerator class."""

    def test_process(self):
        """Test process method."""
        generator = I2VMaskGenerator()

        # Create a test mask
        num_frames, height, width = 4, 32, 32
        mask = torch.ones(num_frames, 1, height, width)

        processed_mask = generator.process(mask)

        # I2V mask should have first frame as 0, others as 1
        judge_expression(torch.all(processed_mask[0] == 0.0))
        judge_expression(torch.all(processed_mask[1:] == 1.0))
        judge_expression(processed_mask.shape == mask.shape)


class TestTransitionMaskGenerator:
    """Test TransitionMaskGenerator class."""

    def test_process(self):
        """Test process method."""
        generator = TransitionMaskGenerator()

        # Create a test mask
        num_frames, height, width = 4, 32, 32
        mask = torch.ones(num_frames, 1, height, width)

        processed_mask = generator.process(mask)

        # Transition mask should have first and last frames as 0, others as 1
        judge_expression(torch.all(processed_mask[0] == 0.0))
        judge_expression(torch.all(processed_mask[-1] == 0.0))
        judge_expression(torch.all(processed_mask[1:-1] == 1.0))
        judge_expression(processed_mask.shape == mask.shape)


class TestContinuationMaskGenerator:
    """Test ContinuationMaskGenerator class."""

    def test_init(self):
        """Test __init__ method with default and custom parameters."""
        # Default parameters
        generator = ContinuationMaskGenerator()
        judge_expression(generator.min_clear_ratio == 0.0)
        judge_expression(generator.max_clear_ratio == 1.0)

        # Custom parameters
        min_ratio, max_ratio = 0.2, 0.8
        generator = ContinuationMaskGenerator(min_clear_ratio=min_ratio, max_clear_ratio=max_ratio)
        judge_expression(generator.min_clear_ratio == min_ratio)
        judge_expression(generator.max_clear_ratio == max_ratio)

    @patch('random.randint')
    def test_process(self, mock_randint):
        """Test process method with mocked random.randint."""
        # Mock the random.randint to return a fixed value
        mock_randint.return_value = 2

        generator = ContinuationMaskGenerator(min_clear_ratio=0.0, max_clear_ratio=1.0)

        # Create a test mask
        num_frames, height, width = 4, 32, 32
        mask = torch.ones(num_frames, 1, height, width)

        processed_mask = generator.process(mask)

        # Continuation mask should have first 2 frames as 0, others as 1
        judge_expression(torch.all(processed_mask[0:2] == 0.0))
        judge_expression(torch.all(processed_mask[2:] == 1.0))
        judge_expression(processed_mask.shape == mask.shape)

        # Verify that random.randint was called with correct arguments
        mock_randint.assert_called_once_with(0, 4)


class TestClearMaskGenerator:
    """Test ClearMaskGenerator class."""

    def test_process(self):
        """Test process method."""
        generator = ClearMaskGenerator()

        # Create a test mask
        num_frames, height, width = 4, 32, 32
        mask = torch.ones(num_frames, 1, height, width)

        processed_mask = generator.process(mask)

        # Clear mask should be all 0s
        judge_expression(torch.all(processed_mask == 0.0))
        judge_expression(processed_mask.shape == mask.shape)


class TestRandomTemporalMaskGenerator:
    """Test RandomTemporalMaskGenerator class."""

    def test_init(self):
        """Test __init__ method with default and custom parameters."""
        # Default parameters
        generator = RandomTemporalMaskGenerator()
        judge_expression(generator.min_clear_ratio == 0.0)
        judge_expression(generator.max_clear_ratio == 1.0)

        # Custom parameters
        min_ratio, max_ratio = 0.3, 0.7
        generator = RandomTemporalMaskGenerator(min_clear_ratio=min_ratio, max_clear_ratio=max_ratio)
        judge_expression(generator.min_clear_ratio == min_ratio)
        judge_expression(generator.max_clear_ratio == max_ratio)

    @patch('random.randint')
    @patch('random.sample')
    def test_process(self, mock_sample, mock_randint):
        """Test process method with mocked random functions."""
        # Mock the random functions to return fixed values
        mock_randint.return_value = 2
        mock_sample.return_value = [1, 3]

        generator = RandomTemporalMaskGenerator(min_clear_ratio=0.0, max_clear_ratio=1.0)

        # Create a test mask
        num_frames, height, width = 4, 32, 32
        mask = torch.ones(num_frames, 1, height, width)

        processed_mask = generator.process(mask)

        # Random temporal mask should have frames 1 and 3 as 0, others as 1
        judge_expression(torch.all(processed_mask[0] == 1.0))
        judge_expression(torch.all(processed_mask[1] == 0.0))
        judge_expression(torch.all(processed_mask[2] == 1.0))
        judge_expression(torch.all(processed_mask[3] == 0.0))
        judge_expression(processed_mask.shape == mask.shape)

        # Verify that random functions were called with correct arguments
        mock_randint.assert_called_once_with(0, 4)
        mock_sample.assert_called_once_with(range(4), 2)


class TestMaskProcessor:
    """Test MaskProcessor class."""

    def test_init(self):
        """Test __init__ method with default and custom parameters."""
        # Default parameters
        processor = MaskProcessor()
        judge_expression(processor.max_height == 640)
        judge_expression(processor.max_width == 640)
        judge_expression(processor.min_clear_ratio == 0.0)
        judge_expression(processor.max_clear_ratio == 1.0)

        # Custom parameters
        max_h, max_w = 512, 512
        min_ratio, max_ratio = 0.1, 0.9
        processor = MaskProcessor(
            max_height=max_h,
            max_width=max_w,
            min_clear_ratio=min_ratio,
            max_clear_ratio=max_ratio
        )
        judge_expression(processor.max_height == max_h)
        judge_expression(processor.max_width == max_w)
        judge_expression(processor.min_clear_ratio == min_ratio)
        judge_expression(processor.max_clear_ratio == max_ratio)

        # Verify that all mask generators are initialized
        for mask_type in MaskType:
            judge_expression(mask_type in processor.mask_generators)

    def test_get_mask(self):
        """Test get_mask method with all mask types."""
        processor = MaskProcessor()

        # Create test pixel values
        num_frames, channels, height, width = 4, 3, 32, 32
        pixel_values = torch.randn(num_frames, channels, height, width)

        # Test each mask type
        for mask_type in MaskType:
            mask = processor.get_mask(mask_type, pixel_values, device="cpu")

            # 兼容3维和4维输出
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)

            judge_expression(mask.shape == (num_frames, 1, height, width))
            judge_expression(mask.device == pixel_values.device)
            judge_expression(mask.dtype == pixel_values.dtype)

    @patch('random.choices')
    def test_call_with_mask_type_ratio_dict(self, mock_choices):
        """Test __call__ method with mask_type_ratio_dict parameter."""
        # Mock random.choices to return a fixed mask type
        mock_choices.return_value = [MaskType.transition]

        processor = MaskProcessor()

        # Create test pixel values
        num_frames, channels, height, width = 4, 3, 32, 32
        pixel_values = torch.randn(num_frames, channels, height, width)

        # Create mask type ratio dict
        mask_type_ratio_dict = {
            MaskType.t2iv: 0.2,
            MaskType.i2v: 0.3,
            MaskType.transition: 0.5
        }

        result = processor(pixel_values, mask_type_ratio_dict=mask_type_ratio_dict)

        judge_expression("mask" in result)
        judge_expression("masked_pixel_values" in result)
        judge_expression(result["masked_pixel_values"].shape == pixel_values.shape)

        # Verify that random.choices was called with correct arguments
        mock_choices.assert_called_once_with(
            list(mask_type_ratio_dict.keys()),
            list(mask_type_ratio_dict.values())
        )

    def test_call_with_invalid_parameters(self):
        """Test __call__ method with invalid parameters."""
        processor = MaskProcessor()

        # Create test pixel values
        num_frames, channels, height, width = 4, 3, 32, 32
        pixel_values = torch.randn(num_frames, channels, height, width)

        # Test with neither mask_type nor mask_type_ratio_dict
        try:
            processor(pixel_values)
            judge_expression(False)  # Should not reach here
        except ValueError:
            pass


class TestMaskCompressor:
    """Test MaskCompressor class."""

    def test_init(self):
        """Test __init__ method with default and custom parameters."""
        # Default parameters
        compressor = MaskCompressor()
        judge_expression(compressor.ae_stride_h == 8)
        judge_expression(compressor.ae_stride_w == 8)
        judge_expression(compressor.ae_stride_t == 4)

        # Custom parameters
        stride_h, stride_w, stride_t = 4, 4, 2
        compressor = MaskCompressor(
            ae_stride_h=stride_h,
            ae_stride_w=stride_w,
            ae_stride_t=stride_t
        )
        judge_expression(compressor.ae_stride_h == stride_h)
        judge_expression(compressor.ae_stride_w == stride_w)
        judge_expression(compressor.ae_stride_t == stride_t)
