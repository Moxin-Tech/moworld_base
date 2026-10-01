# pylint: skip-file
import os

import pytest


class TestClipGradNorm:
    def test_clip_grad_norm_matches_manual_l2(self, monkeypatch):
        pytest.importorskip("torch")
        import torch
        import types

        # avoid initializing parallelstate/device mesh in ut: stup get_parallel_state()
        import mindspeed_mm.fsdp.optimizer.clip_grad_norm as mod

        dummy_ps = types.SimpleNamespace(get_fsdp_group=lambda: None)
        monkeypatch.setattr(mod, "get_parallel_state", lambda: dummy_ps)

        m = torch.nn.Linear(3, 4, bias=False)
        # deterministic gradients
        for p in m.parameters():
            p.grad = torch.ones_like(p.data) * 2.0

        total = mod.clip_grad_norm(m, max_norm=0.0, norm_type=2.0)
        # manual l2 norm: sqrt(sum(g^2))
        manual = torch.sqrt(torch.sum(torch.stack([torch.sum(p.grad.float() ** 2) for p in m.parameters()])))
        assert torch.allclose(total, manual)

    def test_clip_grad_norm_applies_clipping(self, monkeypatch):
        pytest.importorskip("torch")
        import torch
        import types

        import mindspeed_mm.fsdp.optimizer.clip_grad_norm as mod

        dummy_ps = types.SimpleNamespace(get_fsdp_group=lambda: None)
        monkeypatch.setattr(mod, "get_parallel_state", lambda: dummy_ps)

        m = torch.nn.Linear(2, 2, bias=False)
        for p in m.parameters():
            p.grad = torch.ones_like(p.data) * 10.0

        # returned value is the total norm before clipping (see implementation)
        before = mod.clip_grad_norm(m, max_norm=0.0, norm_type=2.0)
        max_norm = 1.0
        returned = mod.clip_grad_norm(m, max_norm=max_norm, norm_type=2.0)
        assert torch.allclose(before, returned)

        # verify gradients are actually clipped
        total_after = torch.sqrt(torch.sum(torch.stack([torch.sum(p.grad.float() ** 2) for p in m.parameters()])))
        assert total_after <= max_norm + 1e-6
