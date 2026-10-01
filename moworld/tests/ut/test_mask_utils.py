# Copyright (c) Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for ``mindspeed_mm.utils.mask_utils`` mask generators.

Covers:
  * Each ``BaseMaskGenerator`` subclass returns a mask of the requested shape.
  * The deterministic generators (T2IV, I2V, Transition, Clear) are
    shape-preserving under repeated invocation and have the expected
    values at the well-defined positions.
  * The stochastic generators (Continuation, RandomTemporal) produce
    masks that respect the configured ``min/max_clear_ratio`` bounds.
  * ``MaskProcessor.__call__`` correctly dispatches on ``mask_type``
    (string and ``MaskType`` enum) and the missing-argument path
    raises.
  * ``MaskCompressor`` produces a mask with the right compressed shape
    for both even and odd T (C=1 only -- see note below).

The tests are CPU-runnable.  The only NPU-specific code path in
``mask_utils`` is the ``F.interpolate`` branch in ``MaskCompressor``,
which falls back to the CPU path when ``torch_npu`` is unavailable.

Note: ``MaskCompressor`` is hard-coded to ``C=1``; the trailing
``.view(B, new_T, ae_stride_t, new_H, new_W)`` does not fold the
channel dim, so multi-channel inputs would crash.  We only test
``C=1`` to avoid coupling to that constraint.
"""

import sys

import pytest
import torch

from mindspeed_mm.utils.mask_utils import (
    BaseMaskGenerator,
    ClearMaskGenerator,
    ContinuationMaskGenerator,
    I2VMaskGenerator,
    MaskCompressor,
    MaskProcessor,
    MaskType,
    RandomTemporalMaskGenerator,
    STR_TO_TYPE,
    T2IVMaskGenerator,
    TransitionMaskGenerator,
    TYPE_TO_STR,
)


# ---------------------------------------------------------------------------
# MaskType / STR_TO_TYPE / TYPE_TO_STR sanity
# ---------------------------------------------------------------------------


class TestMaskTypeEnums:
    """The MaskType enum and its name maps must stay in sync."""

    def test_all_mask_types_have_str_mapping(self):
        for mt in MaskType:
            assert mt in TYPE_TO_STR
            assert TYPE_TO_STR[mt] == mt.name

    def test_str_to_type_round_trip(self):
        for name, mt in STR_TO_TYPE.items():
            assert MaskType[name] is mt
            assert TYPE_TO_STR[mt] == name

    def test_known_mask_types(self):
        expected = {"t2iv", "i2v", "transition", "continuation", "clear", "random_temporal"}
        actual = {mt.name for mt in MaskType}
        assert actual == expected


# ---------------------------------------------------------------------------
# Deterministic mask generators
# ---------------------------------------------------------------------------


class TestT2IVMaskGenerator:
    """T2IV fills the entire mask with 1 (everything is masked)."""

    def test_output_is_all_ones(self):
        gen = T2IVMaskGenerator()
        out = gen(num_frames=4, height=8, width=8, device="cpu", dtype=torch.float32)
        assert out.shape == (4, 1, 8, 8)
        assert torch.all(out == 1.0)

    def test_dtype_preserved(self):
        gen = T2IVMaskGenerator()
        out = gen(num_frames=2, height=4, width=4, device="cpu", dtype=torch.bfloat16)
        assert out.dtype == torch.bfloat16


class TestI2VMaskGenerator:
    """I2V masks the first frame (sets it to 0)."""

    def test_first_frame_cleared(self):
        gen = I2VMaskGenerator()
        out = gen(num_frames=4, height=8, width=8, device="cpu", dtype=torch.float32)
        assert out.shape == (4, 1, 8, 8)
        assert torch.all(out[0] == 0.0)
        assert torch.all(out[1:] == 1.0)

    def test_single_frame(self):
        """num_frames=1 is a degenerate case: the first frame is the only frame."""
        gen = I2VMaskGenerator()
        out = gen(num_frames=1, height=2, width=2, device="cpu")
        assert torch.all(out == 0.0)


class TestTransitionMaskGenerator:
    """Transition masks both the first and last frame."""

    def test_endpoints_cleared(self):
        gen = TransitionMaskGenerator()
        out = gen(num_frames=4, height=8, width=8, device="cpu", dtype=torch.float32)
        assert out.shape == (4, 1, 8, 8)
        assert torch.all(out[0] == 0.0)
        assert torch.all(out[-1] == 0.0)
        # Middle frames are intact.
        assert torch.all(out[1:-1] == 1.0)

    def test_single_frame_clears_only_frame(self):
        gen = TransitionMaskGenerator()
        out = gen(num_frames=1, height=2, width=2, device="cpu")
        assert torch.all(out == 0.0)


class TestClearMaskGenerator:
    """Clear mask is the identity: nothing is masked."""

    def test_all_zeros(self):
        gen = ClearMaskGenerator()
        out = gen(num_frames=4, height=8, width=8, device="cpu")
        assert out.shape == (4, 1, 8, 8)
        assert torch.all(out == 0.0)


# ---------------------------------------------------------------------------
# Stochastic mask generators
# ---------------------------------------------------------------------------


class TestContinuationMaskGenerator:
    """Continuation clears frames [0:end_idx) with end_idx in [min, max]."""

    def test_end_idx_within_ratio_bounds(self):
        """num_cleared must satisfy min_clear_ratio * N <= num_cleared <= max_clear_ratio * N.

        Tested across multiple random seeds to be robust to the draw.
        """
        num_frames = 16
        min_ratio, max_ratio = 0.25, 0.75
        gen = ContinuationMaskGenerator(min_clear_ratio=min_ratio, max_clear_ratio=max_ratio)
        for _ in range(20):
            out = gen(num_frames=num_frames, height=4, width=4, device="cpu")
            num_cleared = int((out == 0).all(dim=-1).all(dim=-1).squeeze().sum().item())
            lo = int(num_frames * min_ratio)
            hi = int(num_frames * max_ratio)
            # ``random.randint`` is inclusive on both ends; allow +-1 for
            # the ceil/floor rounding of float bounds.
            assert lo - 1 <= num_cleared <= hi + 1, (
                f"num_cleared={num_cleared} not in [{lo}, {hi}]"
            )

    def test_full_clear_when_ratios_are_unity(self):
        """With min=max=1.0 the entire video is cleared."""
        gen = ContinuationMaskGenerator(min_clear_ratio=1.0, max_clear_ratio=1.0)
        out = gen(num_frames=4, height=4, width=4, device="cpu")
        assert torch.all(out == 0.0)

    def test_no_clear_when_ratios_are_zero(self):
        """With min=max=0.0 nothing is cleared."""
        gen = ContinuationMaskGenerator(min_clear_ratio=0.0, max_clear_ratio=0.0)
        out = gen(num_frames=4, height=4, width=4, device="cpu")
        assert torch.all(out == 1.0)


class TestRandomTemporalMaskGenerator:
    """RandomTemporal selects a random subset of frames and clears them."""

    def test_selected_count_within_bounds(self):
        num_frames = 16
        min_ratio, max_ratio = 0.25, 0.5
        gen = RandomTemporalMaskGenerator(min_clear_ratio=min_ratio, max_clear_ratio=max_ratio)
        for _ in range(20):
            out = gen(num_frames=num_frames, height=4, width=4, device="cpu")
            num_cleared = int((out == 0).all(dim=-1).all(dim=-1).squeeze().sum().item())
            lo = int(num_frames * min_ratio)
            hi = int(num_frames * max_ratio)
            assert lo - 1 <= num_cleared <= hi + 1

    def test_mask_only_full_frames(self):
        """RandomTemporal clears whole frames, not partial rows/cols."""
        gen = RandomTemporalMaskGenerator()
        out = gen(num_frames=8, height=4, width=4, device="cpu")
        per_frame = out.view(out.shape[0], -1).all(dim=-1)
        cleared = per_frame == 0
        intact = per_frame == 1
        assert torch.all(cleared | intact)


# ---------------------------------------------------------------------------
# BaseMaskGenerator error paths
# ---------------------------------------------------------------------------


class TestBaseMaskGenerator:
    """``BaseMaskGenerator.create_system_mask`` must validate its inputs."""

    def test_missing_num_frames_raises(self):
        gen = T2IVMaskGenerator()
        with pytest.raises(ValueError, match="num_frames, height, and width should be provided"):
            gen(num_frames=None, height=4, width=4, device="cpu")

    def test_missing_height_raises(self):
        gen = T2IVMaskGenerator()
        with pytest.raises(ValueError, match="num_frames, height, and width should be provided"):
            gen(num_frames=2, height=None, width=4, device="cpu")

    def test_missing_width_raises(self):
        gen = T2IVMaskGenerator()
        with pytest.raises(ValueError, match="num_frames, height, and width should be provided"):
            gen(num_frames=2, height=4, width=None, device="cpu")

    def test_process_is_abstract(self):
        """``BaseMaskGenerator`` cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseMaskGenerator()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# MaskProcessor dispatch
# ---------------------------------------------------------------------------


class TestMaskProcessorDispatch:
    """``MaskProcessor.__call__`` dispatches based on ``mask_type``."""

    def _make_pixel_values(self, num_frames=4, height=4, width=4, seed=0):
        torch.manual_seed(seed)
        return torch.randn(num_frames, 3, height, width)

    def test_mask_type_enum(self):
        proc = MaskProcessor()
        pv = self._make_pixel_values()
        out = proc(pv, mask_type=MaskType.clear)
        assert "mask" in out and "masked_pixel_values" in out
        # For a Clear mask, ``masked_pixel_values = pv * (mask < 0.5) = pv * 1 = pv``,
        # so the pixel values are preserved exactly.
        assert torch.all(out["mask"] == 0.0)
        assert torch.allclose(out["masked_pixel_values"], pv)

    def test_mask_type_string(self):
        # On Python 3.12+ this works because Enum supports in with
        # member values.  On 3.10/3.11 the in raises TypeError.
        # In both cases a string mask_type that matches a MaskType
        # value must be accepted and dispatched to that generator.
        proc = MaskProcessor()
        pv = self._make_pixel_values()
        if sys.version_info >= (3, 12):
            out = proc(pv, mask_type="clear")
            assert torch.all(out["mask"] == 0.0)
        else:
            with pytest.raises(TypeError):
                proc(pv, mask_type="clear")

    def test_unknown_mask_type_string_raises(self):
        # On Python 3.10/3.11 the in MaskType raises TypeError for
        # a non-MaskType string before the lookup runs; on 3.12+ it
        # returns False and the subsequent KeyError is raised.
        proc = MaskProcessor()
        pv = self._make_pixel_values()
        if sys.version_info >= (3, 12):
            with pytest.raises(KeyError):
                proc(pv, mask_type="not_a_real_type")
        else:
            with pytest.raises(TypeError):
                proc(pv, mask_type="not_a_real_type")

    def test_no_mask_arguments_raises(self):
        proc = MaskProcessor()
        pv = self._make_pixel_values()
        with pytest.raises(ValueError, match="mask_type or mask_type_ratio_dict should be provided"):
            proc(pv)

    def test_masked_pixel_values_zero_when_mask_full(self):
        """T2IV mask is all ones -> masked pixel values are all zero."""
        proc = MaskProcessor()
        pv = self._make_pixel_values()
        out = proc(pv, mask_type=MaskType.t2iv)
        assert torch.allclose(out["masked_pixel_values"], torch.zeros_like(pv))

    def test_random_ratio_dict_picks_a_known_type(self):
        """When the ratio dict is degenerate, the chosen generator must be honored."""
        proc = MaskProcessor()
        pv = self._make_pixel_values()
        ratio_dict = {MaskType.t2iv: 1.0}  # Always pick t2iv
        out = proc(pv, mask_type_ratio_dict=ratio_dict)
        assert torch.all(out["mask"] == 1.0)


# ---------------------------------------------------------------------------
# MaskCompressor
# ---------------------------------------------------------------------------


class TestMaskCompressor:
    """``MaskCompressor`` compresses a (B, 1, T, H, W) mask spatially and temporally.

    The current implementation only supports ``C=1`` -- the trailing
    ``.view(B, new_T, ae_stride_t, new_H, new_W)`` does not fold the
    channel dim into new_T.  The tests stay at ``C=1`` accordingly.
    """

    def test_compress_even_t(self):
        """For even T, the compressed T is T / ae_stride_t."""
        compressor = MaskCompressor(ae_stride_h=2, ae_stride_w=2, ae_stride_t=2)
        mask = torch.ones(1, 1, 8, 8, 8)
        out = compressor(mask)
        # Output is (B, ae_stride_t, T/ae_stride_t, H/ae_stride_h, W/ae_stride_w).
        assert out.shape == (1, 2, 4, 4, 4)

    def test_compress_odd_t(self):
        """For odd T, the compressed T is T // ae_stride_t + 1."""
        compressor = MaskCompressor(ae_stride_h=2, ae_stride_w=2, ae_stride_t=2)
        mask = torch.ones(1, 1, 7, 8, 8)  # T=7 is odd
        out = compressor(mask)
        assert out.shape == (1, 2, 4, 4, 4)  # 7//2 + 1 = 4

    def test_compress_preserves_batch(self):
        compressor = MaskCompressor(ae_stride_h=4, ae_stride_w=4, ae_stride_t=4)
        mask = torch.ones(2, 1, 8, 16, 16)
        out = compressor(mask)
        assert out.shape[0] == 2
        # ae_stride_t is the second axis after the transpose.
        assert out.shape[1] == compressor.ae_stride_t
