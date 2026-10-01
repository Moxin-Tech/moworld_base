"""Unit tests for FSDP gradient clipping helper functions and control flow."""

import math
import os
import types

import pytest


def _make_param_with_grad(values):
    torch = pytest.importorskip("torch")
    param = torch.nn.Parameter(torch.zeros(len(values), dtype=torch.float32))
    param.grad = torch.tensor(values, dtype=torch.float32)
    return param


class TestLocalPthSum:
    @pytest.mark.parametrize(
        "grads,p,expected",
        [
            ([[3.0, 4.0]], 2.0, 25.0),
            ([[1.0, 2.0, 2.0]], 2.0, 9.0),
            ([[1.0, -2.0, 3.0]], 1.0, 6.0),
            ([[2.0, -3.0]], 3.0, 35.0),
            ([[0.0, 0.0]], 2.0, 0.0),
            ([[1.5, -2.0]], 2.0, 6.25),
            ([[1.0], [2.0], [3.0]], 2.0, 14.0),
            ([[1.0, 2.0], [3.0, 4.0]], 2.0, 30.0),
            ([[1.0, -2.0], [-3.0, 4.0]], 1.0, 10.0),
            ([[2.0, 2.0], [2.0, 2.0]], 4.0, 64.0),
        ],
    )
    def test_local_pth_sum_matches_manual_sum_of_powers(self, grads, p, expected):
        from mindspeed_mm.fsdp.optimizer.clip_grad_norm import _local_pth_sum

        params = [_make_param_with_grad(values) for values in grads]

        assert _local_pth_sum(params, p).item() == pytest.approx(expected)

    def test_local_pth_sum_ignores_parameters_without_grad(self):
        torch = pytest.importorskip("torch")
        from mindspeed_mm.fsdp.optimizer.clip_grad_norm import _local_pth_sum

        with_grad = _make_param_with_grad([3.0, 4.0])
        without_grad = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))

        assert _local_pth_sum([with_grad, without_grad], 2.0).item() == pytest.approx(25.0)

    def test_local_pth_sum_preserves_current_empty_param_behavior(self, monkeypatch):
        pytest.importorskip("torch")
        import mindspeed_mm.fsdp.optimizer.clip_grad_norm as mod

        monkeypatch.setattr(mod, "get_device_type", lambda: "cpu")

        with pytest.raises(RuntimeError, match="empty"):
            mod._local_pth_sum([], 2.0)


class TestLocalMax:
    @pytest.mark.parametrize(
        "grads,expected",
        [
            ([[3.0, 4.0]], 4.0),
            ([[-3.0, 4.0]], 4.0),
            ([[0.0, 0.0]], 0.0),
            ([[1.0, 2.0], [9.0]], 9.0),
            ([[-1.0, -2.0], [-8.0, 3.0]], 8.0),
            ([[0.5], [0.25], [0.125]], 0.5),
        ],
    )
    def test_local_max_returns_largest_absolute_gradient(self, grads, expected):
        from mindspeed_mm.fsdp.optimizer.clip_grad_norm import _local_max

        params = [_make_param_with_grad(values) for values in grads]

        assert _local_max(params).item() == pytest.approx(expected)

    def test_local_max_ignores_parameters_without_grad(self):
        torch = pytest.importorskip("torch")
        from mindspeed_mm.fsdp.optimizer.clip_grad_norm import _local_max

        with_grad = _make_param_with_grad([1.0, -7.0])
        without_grad = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32))

        assert _local_max([without_grad, with_grad]).item() == pytest.approx(7.0)

    def test_local_max_returns_zero_for_empty_param_list(self, monkeypatch):
        pytest.importorskip("torch")
        import mindspeed_mm.fsdp.optimizer.clip_grad_norm as mod

        monkeypatch.setattr(mod, "get_device_type", lambda: "cpu")

        value = mod._local_max([])

        assert value.item() == pytest.approx(0.0)


class TestFsdp2ReduceGroup:
    def test_fsdp2_reduce_group_uses_sum_for_finite_norms(self, monkeypatch):
        pytest.importorskip("torch")
        import mindspeed_mm.fsdp.optimizer.clip_grad_norm as mod

        calls = []

        def fake_all_reduce(value, op=None, group=None):
            calls.append((value.item(), op, group))
            value.add_(10.0)

        monkeypatch.setattr(mod.dist, "all_reduce", fake_all_reduce)

        group_a = object()
        group_b = object()
        result = mod._fsdp2_reduce_group(
            params=[_make_param_with_grad([3.0, 4.0])],
            norm_type=2.0,
            reduce_groups=[("a", group_a), ("b", group_b), ("none", None)],
        )

        assert result.item() == pytest.approx(45.0)
        assert calls == [
            (25.0, mod.dist.ReduceOp.SUM, group_a),
            (35.0, mod.dist.ReduceOp.SUM, group_b),
        ]

    def test_fsdp2_reduce_group_uses_max_for_inf_norm(self, monkeypatch):
        pytest.importorskip("torch")
        import mindspeed_mm.fsdp.optimizer.clip_grad_norm as mod

        calls = []

        def fake_all_reduce(value, op=None, group=None):
            calls.append((value.item(), op, group))
            value.fill_(max(value.item(), 9.0))

        monkeypatch.setattr(mod.dist, "all_reduce", fake_all_reduce)

        group = object()
        result = mod._fsdp2_reduce_group(
            params=[_make_param_with_grad([3.0, 4.0])],
            norm_type=float("inf"),
            reduce_groups=[("fsdp", group)],
        )

        assert result.item() == pytest.approx(9.0)
        assert calls == [(4.0, mod.dist.ReduceOp.MAX, group)]

    def test_fsdp2_reduce_group_skips_none_groups(self, monkeypatch):
        pytest.importorskip("torch")
        import mindspeed_mm.fsdp.optimizer.clip_grad_norm as mod

        calls = []
        monkeypatch.setattr(mod.dist, "all_reduce", lambda *args, **kwargs: calls.append((args, kwargs)))

        result = mod._fsdp2_reduce_group(
            params=[_make_param_with_grad([1.0, 2.0])],
            norm_type=2.0,
            reduce_groups=[("none", None)],
        )

        assert result.item() == pytest.approx(5.0)
        assert calls == []


class TestClipGradNormControlFlow:
    def test_clip_grad_norm_uses_ep_path_when_model_has_ep_param_groups(self, monkeypatch):
        pytest.importorskip("torch")
        import torch
        import mindspeed_mm.fsdp.optimizer.clip_grad_norm as mod

        model = torch.nn.Linear(2, 2)
        model._ep_param_groups = {"ep": set(), "non_ep": set()}
        sentinel = torch.tensor(12.0)

        monkeypatch.setattr(mod, "ep_fsdp2_clip_grad_norm", lambda *args, **kwargs: sentinel)

        assert mod.clip_grad_norm(model, max_norm=1.0) is sentinel

    def test_clip_grad_norm_compute_only_does_not_modify_gradients(self, monkeypatch):
        torch = pytest.importorskip("torch")
        import mindspeed_mm.fsdp.optimizer.clip_grad_norm as mod

        dummy_ps = types.SimpleNamespace(get_fsdp_group=lambda: None)
        monkeypatch.setattr(mod, "get_parallel_state", lambda: dummy_ps)

        model = torch.nn.Linear(2, 2, bias=False)
        model.weight.grad = torch.full_like(model.weight, 3.0)
        before = model.weight.grad.clone()

        returned = mod.clip_grad_norm(model, max_norm=0.0, norm_type=2.0)

        assert torch.allclose(model.weight.grad, before)
        assert returned.item() == pytest.approx(math.sqrt(float(torch.sum(before.float() ** 2))))

    def test_ep_fsdp2_clip_grad_norm_compute_only_returns_combined_norm(self, monkeypatch):
        torch = pytest.importorskip("torch")
        import mindspeed_mm.fsdp.optimizer.clip_grad_norm as mod

        ep_param = _make_param_with_grad([3.0])
        non_ep_param = _make_param_with_grad([4.0])

        model = types.SimpleNamespace(
            _ep_param_groups={
                "ep": {ep_param},
                "non_ep": {non_ep_param},
            }
        )
        dummy_ps = types.SimpleNamespace(
            get_fsdp_group=lambda: None,
            get_ep_group=lambda: None,
            get_efsdp_group=lambda: None,
            is_ep_enable=lambda: True,
        )
        monkeypatch.setattr(mod, "get_parallel_state", lambda: dummy_ps)

        returned = mod.ep_fsdp2_clip_grad_norm(model, max_norm=0.0, norm_type=2.0)

        assert returned.item() == pytest.approx(5.0)
        assert torch.allclose(ep_param.grad, torch.tensor([3.0]))
        assert torch.allclose(non_ep_param.grad, torch.tensor([4.0]))
