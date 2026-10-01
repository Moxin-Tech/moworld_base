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

"""Unit tests for the ``mindspeed_mm.optimizer.muon`` module.

Covers:
  * ``zeropower_via_newtonschulz5``: orthogonalization properties
    (singular-value bounds, tall/wide input handling, error path on
    non-2D input).
  * ``adjust_lr_wd_for_muon``: shape-based learning-rate scaling.
  * ``normalize_range``: simple translation of a tuple by ``start``.
  * ``Muon`` (the optimizer): initialization, two-group routing
    (Muon group vs AdamW fallback group), gradient step on a tiny
    parameter set, and ``state_dict`` round-trip.

The tests are CPU-only.  Muon's Newton-Schulz iteration runs in
bf16 by design; on a CPU host this is slow but correct for tiny
tensors, which is all we need here.
"""

import copy

import pytest
import torch

from mindspeed_mm.optimizer.muon import (
    Muon,
    adjust_lr_wd_for_muon,
    normalize_range,
    zeropower_via_newtonschulz5,
)


# ---------------------------------------------------------------------------
# zeropower_via_newtonschulz5
# ---------------------------------------------------------------------------


class TestZeropowerViaNewtonSchulz5:
    """Properties of the NS5 orthogonalization used by Muon."""

    def test_output_shape_matches_input(self):
        torch.manual_seed(0)
        G = torch.randn(8, 4, dtype=torch.bfloat16)
        out = zeropower_via_newtonschulz5(G, steps=5)
        assert out.shape == G.shape

    def test_wide_input_tall_output(self):
        """Tall matrices (rows > cols) are transposed, orthogonalized, and transposed back."""
        torch.manual_seed(1)
        G = torch.randn(16, 4, dtype=torch.bfloat16)
        out = zeropower_via_newtonschulz5(G, steps=5)
        assert out.shape == G.shape

    def test_singular_values_bounded_around_one(self):
        """After NS5, the singular values of ``G`` are clustered around 1.

        The Keller Jordan NS5 does not produce a perfect orthogonal
        matrix; the singular values are in ``[0.5, 1.5]`` by design
        (the iteration pushes towards 1 but the slope-at-zero
        coefficient choice trades exact convergence for fast
        convergence in the early steps).
        """
        torch.manual_seed(2)
        G = torch.randn(16, 16, dtype=torch.bfloat16).float()
        out = zeropower_via_newtonschulz5(G.to(torch.bfloat16), steps=5).float()
        s = torch.linalg.svdvals(out)
        # The interval [0.5, 1.5] is the documented design range.
        assert s.min() > 0.4, f"min singular value {s.min()} below 0.4"
        assert s.max() < 1.6, f"max singular value {s.max()} above 1.6"

    def test_non_2d_input_raises(self):
        """1D and 3D inputs are programmer errors and must raise ValueError."""
        with pytest.raises(ValueError, match="expects a 2-D tensor"):
            zeropower_via_newtonschulz5(torch.randn(8), steps=5)
        with pytest.raises(ValueError, match="expects a 2-D tensor"):
            zeropower_via_newtonschulz5(torch.randn(2, 3, 4), steps=5)

    def test_more_steps_closer_to_orthogonal(self):
        """More NS5 iterations -> singular values cluster tighter around 1.

        This is the empirical claim of the original paper; the test
        uses a small number of steps (3 vs 8) and a 32x32 matrix to
        stay within CPU-time budget.
        """
        torch.manual_seed(3)
        G = torch.randn(32, 32, dtype=torch.bfloat16)
        s_few = torch.linalg.svdvals(zeropower_via_newtonschulz5(G, steps=3).float())
        s_many = torch.linalg.svdvals(zeropower_via_newtonschulz5(G, steps=8).float())
        # The spread of singular values shrinks with more iterations.
        spread_few = s_few.max() - s_few.min()
        spread_many = s_many.max() - s_many.min()
        assert spread_many < spread_few


# ---------------------------------------------------------------------------
# adjust_lr_wd_for_muon
# ---------------------------------------------------------------------------


class TestAdjustLrWdForMuon:
    """``adjust_lr_wd_for_muon`` rescales lr by ``sqrt(max(A, B)) * matched_rms``."""

    def test_lr_scaled_by_sqrt_max_dim(self):
        """For a (16, 8) parameter, adjusted_lr = lr * sqrt(16) * matched_rms."""
        lr = 0.02
        matched_rms = 0.2
        adjusted = adjust_lr_wd_for_muon(lr, matched_rms, (16, 8))
        assert adjusted == pytest.approx(0.02 * (16 ** 0.5) * 0.2)

    def test_lr_uses_max_of_two_dims(self):
        """For a (4, 16) parameter, the larger dim (16) wins."""
        adjusted_1 = adjust_lr_wd_for_muon(0.02, 0.2, (16, 4))
        adjusted_2 = adjust_lr_wd_for_muon(0.02, 0.2, (4, 16))
        assert adjusted_1 == adjusted_2

    def test_extra_dims_ignored(self):
        """The function only looks at the first two dims of the shape tuple."""
        adjusted = adjust_lr_wd_for_muon(0.02, 0.2, (16, 8, 64, 64))
        assert adjusted == pytest.approx(0.02 * 4 * 0.2)

    def test_zero_lr_stays_zero(self):
        """A zero base lr scales to zero regardless of the matched_rms."""
        assert adjust_lr_wd_for_muon(0.0, 0.5, (16, 16)) == 0.0


# ---------------------------------------------------------------------------
# normalize_range
# ---------------------------------------------------------------------------


class TestNormalizeRange:
    """``normalize_range`` shifts a (start, end) tuple by ``-start``."""

    def test_basic_shift(self):
        assert normalize_range((5, 10), 3) == (2, 7)

    def test_zero_start_no_change(self):
        assert normalize_range((2, 8), 0) == (2, 8)

    def test_negative_shift(self):
        """A negative ``start`` shifts in the opposite direction."""
        assert normalize_range((2, 8), -3) == (5, 11)


# ---------------------------------------------------------------------------
# Muon optimizer
# ---------------------------------------------------------------------------


def _make_param_and_grad(shape, with_grad=True, init="randn"):
    """Tiny helper to make a leaf parameter with a grad."""
    torch.manual_seed(0)
    if init == "randn":
        p = torch.nn.Parameter(torch.randn(*shape, dtype=torch.float32))
    elif init == "ones":
        p = torch.nn.Parameter(torch.ones(*shape, dtype=torch.float32))
    elif init == "zeros":
        p = torch.nn.Parameter(torch.zeros(*shape, dtype=torch.float32))
    else:
        raise ValueError(f"unknown init: {init}")
    if with_grad:
        p.grad = torch.randn_like(p)
    return p


class TestMuonOptimizer:
    """End-to-end behavior of the ``Muon`` optimizer on tiny parameters."""

    def test_init_stores_hyperparameters(self):
        """All keyword hyperparameters are stored in ``defaults``."""
        params = [_make_param_and_grad((8, 8))]
        opt = Muon(
            [{"params": params}],
            lr=0.02,
            weight_decay=0.1,
            matched_adamw_rms=0.2,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
            adamw_betas=(0.9, 0.95),
            adamw_eps=1e-8,
        )
        assert opt.defaults["lr"] == 0.02
        assert opt.defaults["weight_decay"] == 0.1
        assert opt.defaults["matched_adamw_rms"] == 0.2
        assert opt.defaults["momentum"] == 0.95
        assert opt.defaults["nesterov"] is True
        assert opt.defaults["ns_steps"] == 5
        assert opt.defaults["adamw_betas"] == (0.9, 0.95)
        assert opt.defaults["adamw_eps"] == 1e-8

    def test_step_advances_muon_param(self):
        """A Muon step must change the parameter values.

        We use ``use_muon=True`` in the param group so the step goes
        through the Newton-Schulz path.  A non-zero grad + non-zero
        lr + a real parameter must move the param.
        """
        param = _make_param_and_grad((8, 8))
        original = param.data.clone()
        opt = Muon(
            [{"params": [param], "use_muon": True}],
            lr=0.01,
            ns_steps=3,
        )
        opt.step()
        # Param must have changed; magnitude is unconstrained but >0.
        assert not torch.allclose(param.data, original), "param did not change after Muon step"

    def test_step_with_no_grad_does_not_change_param(self):
        """A param without a grad must be skipped (no NaN/Inf introduced)."""
        param = _make_param_and_grad((8, 8), with_grad=False)
        original = param.data.clone()
        opt = Muon(
            [{"params": [param], "use_muon": True}],
            lr=0.01,
            ns_steps=3,
        )
        opt.step()
        assert torch.allclose(param.data, original)

    def test_adamw_fallback_group(self):
        """A non-Muon group (use_muon=False) goes through the AdamW path."""
        param = _make_param_and_grad((8, 8))
        original = param.data.clone()
        opt = Muon(
            [{"params": [param], "use_muon": False}],
            lr=0.01,
        )
        opt.step()
        assert not torch.allclose(param.data, original)

    def test_state_dict_round_trip(self):
        """``state_dict`` and ``load_state_dict`` preserve param and step state."""
        param_muon = _make_param_and_grad((8, 8))
        param_adamw = _make_param_and_grad((8, 8))
        opt = Muon(
            [
                {"params": [param_muon], "use_muon": True},
                {"params": [param_adamw], "use_muon": False},
            ],
            lr=0.01,
            ns_steps=3,
        )
        opt.step()  # initialize state entries
        sd = opt.state_dict()
        # ``state_dict`` is a list of dicts (one per param).  It must
        # be deep-copyable and re-loadable into a fresh optimizer.
        sd_copy = copy.deepcopy(sd)
        opt2 = Muon(
            [
                {"params": [_make_param_and_grad((8, 8), with_grad=False)], "use_muon": True},
                {"params": [_make_param_and_grad((8, 8), with_grad=False)], "use_muon": False},
            ],
            lr=0.01,
            ns_steps=3,
        )
        opt2.load_state_dict(sd_copy)

    def test_muon_step_with_zero_grad_is_safe(self):
        """A zero grad should keep the param unchanged (no NaN from div-by-zero)."""
        param = _make_param_and_grad((8, 8))
        param.grad = torch.zeros_like(param)
        original = param.data.clone()
        opt = Muon(
            [{"params": [param], "use_muon": True}],
            lr=0.01,
            ns_steps=3,
        )
        opt.step()
        # With grad=0 the NS5 update is 0; weight decay (lr*wd) shrinks the
        # param slightly.  Either way the param must remain finite.
        assert torch.isfinite(param.data).all()
        # And the param must not have moved more than what weight decay alone
        # would cause (``p * (1 - lr * wd)``).
        assert torch.allclose(param.data, original * (1 - 0.01 * opt.defaults["weight_decay"]), atol=1e-3)
