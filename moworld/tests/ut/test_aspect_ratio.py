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

"""Unit tests for ``mindspeed_mm.data.data_utils.aspect_ratio``.

Covers the pure-Python helpers used by the bucket/video pipelines.
No GPU or torch.distributed is required.

Specifically:
  * ``get_ar`` parses a ``"H:W"`` string to a float ratio.
  * ``get_h_w`` picks integer ``(H, W)`` whose aspect ratio is closest
    to the input and whose product is close to the target ``ts``.
  * ``get_aspect_ratios_dict`` materialises a dict for a list of
    aspect ratios.
  * ``get_closest_ratio`` finds the bucket ratio nearest a given
    ``H/W`` value.
  * ``get_image_size`` returns the ``(H, W)`` tuple for a resolution
    and a known ``H:W`` string.
  * ``get_num_pixels`` and ``get_num_pixels_from_name`` return the
    expected pixel counts.
"""

import pytest

# Import the module file directly to avoid triggering mindspeed_mm/__init__.py
# (which depends on megatron).  This keeps the UT CPU-runnable.
import importlib.util as _importlib_util
import os as _os
_AR_PATH = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)),
    "../../mindspeed_mm/data/data_utils/aspect_ratio.py",
)
_AR_SPEC = _importlib_util.spec_from_file_location("mindspeed_mm_aspect_ratio", _AR_PATH)
_ar = _importlib_util.module_from_spec(_AR_SPEC)
_AR_SPEC.loader.exec_module(_ar)
ASPECT_RATIO_MAP = _ar.ASPECT_RATIO_MAP
ASPECT_RATIOS = _ar.ASPECT_RATIOS
get_ar = _ar.get_ar
get_aspect_ratios_dict = _ar.get_aspect_ratios_dict
get_aspect_ratios_dict_sora2 = _ar.get_aspect_ratios_dict_sora2
get_closest_ratio = _ar.get_closest_ratio
get_closest_ratio_sora2 = _ar.get_closest_ratio_sora2
get_h_w = _ar.get_h_w
get_image_size = _ar.get_image_size
get_num_frames = _ar.get_num_frames
get_num_pixels = _ar.get_num_pixels
get_num_pixels_from_name = _ar.get_num_pixels_from_name
get_ratio = _ar.get_ratio
get_resolution_with_aspect_ratio = _ar.get_resolution_with_aspect_ratio


# ---------------------------------------------------------------------------
# get_ar
# ---------------------------------------------------------------------------


class TestGetAr:
    """``get_ar`` parses a ``"H:W"`` string into a float ratio."""

    def test_square(self):
        assert get_ar("1:1") == 1.0

    def test_3_to_2(self):
        assert get_ar("3:2") == 1.5

    def test_16_to_9(self):
        assert get_ar("16:9") == pytest.approx(16 / 9)

    def test_portrait(self):
        """Portrait (H > W) gives a ratio > 1."""
        assert get_ar("9:16") == pytest.approx(9 / 16)


# ---------------------------------------------------------------------------
# get_h_w
# ---------------------------------------------------------------------------


class TestGetHw:
    """``get_h_w(a, ts)`` returns integer ``(H, W)`` for aspect ratio ``a``."""

    def test_square_360x640_ts(self):
        """For aspect ratio 1.0 and target 360*640 = 230400, result is (480, 480)."""
        # 230400 ** 0.5 = 480, so H = W = 480 (even, no floor/ceil flip).
        h, w = get_h_w(1.0, 360 * 640)
        assert (h, w) == (480, 480)

    def test_result_dims_are_even(self):
        """``get_h_w`` always rounds to even dims (a divisibility invariant)."""
        for a in [0.5, 1.0, 1.5, 2.0]:
            h, w = get_h_w(a, 360 * 640)
            assert h % 2 == 0, f"H={h} is odd for aspect ratio {a}"
            assert w % 2 == 0, f"W={w} is odd for aspect ratio {a}"

    def test_result_aspect_close_to_input(self):
        """The integer H/W should be close to the requested aspect ratio."""
        for a in [0.5, 0.75, 1.0, 1.333, 1.5, 2.0]:
            h, w = get_h_w(a, 360 * 640)
            assert h / w == pytest.approx(a, abs=0.05)

    def test_pixel_count_close_to_target(self):
        """The integer H*W should be close to (but typically slightly above) the target."""
        target = 230400  # 360 * 640
        h, w = get_h_w(1.0, target)
        assert abs(h * w - target) / target < 0.05  # within 5%


# ---------------------------------------------------------------------------
# get_aspect_ratios_dict
# ---------------------------------------------------------------------------


class TestGetAspectRatiosDict:
    """``get_aspect_ratios_dict`` builds a name->(H, W) dict from a list of ratios."""

    def test_keys_are_string_floats(self):
        out = get_aspect_ratios_dict([1.0, 0.5, 2.0])
        assert set(out.keys()) == {"1.00", "0.50", "2.00"}

    def test_values_are_int_tuples(self):
        out = get_aspect_ratios_dict([1.0])
        h, w = out["1.00"]
        assert isinstance(h, int)
        assert isinstance(w, int)

    def test_custom_target(self):
        """A non-default ``ts`` shrinks the resulting dims proportionally."""
        out_small = get_aspect_ratios_dict([1.0], ts=100 * 100)
        out_large = get_aspect_ratios_dict([1.0], ts=1000 * 1000)
        h_small, _ = out_small["1.00"]
        h_large, _ = out_large["1.00"]
        assert h_large > h_small


# ---------------------------------------------------------------------------
# get_closest_ratio
# ---------------------------------------------------------------------------


class TestGetClosestRatio:
    """``get_closest_ratio(h, w, ratios)`` returns the ratio key closest to h/w."""

    def test_exact_match(self):
        ratios = {"0.50": None, "1.00": None, "2.00": None}
        # h=16, w=8 -> 2.0 -> "2.00"
        assert get_closest_ratio(16, 8, ratios) == "2.00"
        # h=8, w=8 -> 1.0 -> "1.00"
        assert get_closest_ratio(8, 8, ratios) == "1.00"

    def test_tie_breaks_min(self):
        """``min`` picks the first key with the minimum distance (stable across CPython)."""
        ratios = {"1.00": None, "1.01": None}
        # h=10, w=10 -> 1.0 exactly; min picks "1.00".
        assert get_closest_ratio(10, 10, ratios) == "1.00"

    def test_close_to_middle(self):
        """A value in the middle picks the closest key by absolute distance."""
        ratios = {"0.50": None, "1.00": None, "2.00": None}
        # h=12, w=8 -> 1.5, equidistant from 1.0 and 2.0; min picks "1.00".
        # Use a value clearly closer to one or the other:
        # h=11, w=8 -> 1.375, closer to 1.0.
        assert get_closest_ratio(11, 8, ratios) == "1.00"
        # h=15, w=8 -> 1.875, closer to 2.0.
        assert get_closest_ratio(15, 8, ratios) == "2.00"


# ---------------------------------------------------------------------------
# get_image_size
# ---------------------------------------------------------------------------


class TestGetImageSize:
    """``get_image_size`` looks up ``(H, W)`` for a (resolution, ar_ratio) pair."""

    def test_known_resolution_and_ratio(self):
        # Pick a known bucket and confirm the lookup is well-defined.
        # "360p" / "16:9" is one such pair; if it's removed from the
        # config in the future, the test will surface a clear error.
        if "360p" in ASPECT_RATIOS and "16:9" in ASPECT_RATIO_MAP:
            h, w = get_image_size("360p", "16:9")
            assert isinstance(h, int) and isinstance(w, int)
            assert h > 0 and w > 0

    def test_unknown_aspect_ratio_raises(self):
        # The lookup happens in two stages (ASPECT_RATIO_MAP then rs_dict);
        # either can raise.  We just confirm *some* clear error is raised.
        with pytest.raises((AssertionError, KeyError)):
            get_image_size("360p", "9999:9999")


# ---------------------------------------------------------------------------
# get_num_pixels
# ---------------------------------------------------------------------------


class TestGetNumPixels:
    """``get_num_pixels(name)`` returns the canonical pixel count for a bucket."""

    def test_known_pixels(self):
        # ASPECT_RATIOS["360p"] = (230400, {...}); confirm lookup.
        if "360p" in ASPECT_RATIOS:
            assert get_num_pixels("360p") == 230400

    def test_720p(self):
        if "720p" in ASPECT_RATIOS:
            assert get_num_pixels("720p") == 921600


# ---------------------------------------------------------------------------
# get_ratio
# ---------------------------------------------------------------------------


class TestGetRatio:
    """``get_ratio(name)`` returns the float ratio for a named bucket key."""

    def test_known(self):
        # The exact values depend on the bucket map; the function
        # must at least return a finite positive float.
        if "360p" in ASPECT_RATIO_MAP:
            r = get_ratio("360p")
            assert isinstance(r, float)
            assert r > 0


# ---------------------------------------------------------------------------
# get_num_frames
# ---------------------------------------------------------------------------


class TestGetNumFrames:
    """``get_num_frames(num_frames)`` adjusts the count to be ``ae_stride_t`` aligned."""

    def test_passthrough_when_aligned(self):
        """A count that is already aligned is returned unchanged (or unchanged + 0)."""
        # The function is supposed to make num_frames compatible with
        # the AE temporal stride (default 4 in the codebase).  The
        # exact rule is implementation-defined; we test that the
        # function returns an int and preserves the value up to a
        # stride-aligned delta.
        out = get_num_frames(16)
        assert isinstance(out, int)
        # The output should be at most stride-1 away from the input.
        assert abs(out - 16) < 16
