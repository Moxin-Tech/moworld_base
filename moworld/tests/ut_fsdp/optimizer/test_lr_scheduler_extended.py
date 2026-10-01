"""Unit tests for FSDP learning-rate scheduler curves and multi-scheduler wrapper."""

import math
import os
from collections import OrderedDict

import pytest


def _make_optimizer(lr=1.0):
    torch = pytest.importorskip("torch")
    param = torch.nn.Parameter(torch.tensor([1.0]))
    return torch.optim.SGD([param], lr=lr)


def _scheduler_lrs(scheduler, steps):
    values = [scheduler.get_last_lr()[0]]
    for _ in range(steps):
        scheduler.step()
        values.append(scheduler.get_last_lr()[0])
    return values


class _FakeMultiOptimizer:
    _is_multi_optimizer = True

    def __init__(self, optimizers_dict):
        self.optimizers_dict = optimizers_dict
        self.key_names = list(optimizers_dict.keys())


class _FakeScheduler:
    def __init__(self, lr):
        self.lr = lr
        self.step_calls = 0
        self.loaded_state = None

    def step(self):
        self.step_calls += 1

    def state_dict(self):
        return {"lr": self.lr, "step_calls": self.step_calls}

    def load_state_dict(self, state_dict):
        self.loaded_state = state_dict
        self.lr = state_dict["lr"]
        self.step_calls = state_dict["step_calls"]

    def get_last_lr(self):
        return [self.lr]


class TestConstantScheduleWithWarmup:
    @pytest.mark.parametrize(
        "warmup_steps,lr_start,init_lr,expected",
        [
            (0, 0.0, 1.0, [1.0, 1.0, 1.0, 1.0]),
            (1, 0.0, 1.0, [0.0, 1.0, 1.0, 1.0]),
            (2, 0.0, 1.0, [0.0, 0.5, 1.0, 1.0]),
            (4, 0.0, 1.0, [0.0, 0.25, 0.5, 0.75]),
            (4, 0.2, 1.0, [0.2, 0.4, 0.6, 0.8]),
            (5, 0.1, 0.5, [0.1, 0.18, 0.26, 0.34]),
        ],
    )
    def test_constant_schedule_warmup_values(self, warmup_steps, lr_start, init_lr, expected):
        from mindspeed_mm.fsdp.optimizer.lr_scheduler import get_constant_schedule_with_warmup

        optimizer = _make_optimizer(lr=init_lr)
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            init_lr=init_lr,
            lr_start=lr_start,
        )

        observed = _scheduler_lrs(scheduler, len(expected) - 1)

        assert observed == pytest.approx(expected)

    def test_constant_schedule_returns_zero_when_init_lr_is_zero(self):
        from mindspeed_mm.fsdp.optimizer.lr_scheduler import get_constant_schedule_with_warmup

        optimizer = _make_optimizer(lr=0.0)
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=3,
            init_lr=0.0,
        )

        assert _scheduler_lrs(scheduler, 4) == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0])


class TestLinearScheduleWithWarmup:
    @pytest.mark.parametrize(
        "warmup_steps,total_steps,expected",
        [
            (0, 4, [1.0, 0.75, 0.5, 0.25, 1e-7]),
            (1, 4, [0.0, 1.0, 2 / 3, 1 / 3, 1e-7]),
            (2, 6, [0.0, 0.5, 1.0, 0.75, 0.5, 0.25, 1e-7]),
            (3, 9, [0.0, 1 / 3, 2 / 3, 1.0, 5 / 6, 4 / 6, 3 / 6]),
        ],
    )
    def test_linear_schedule_warmup_then_decays_to_min_lr(self, warmup_steps, total_steps, expected):
        from mindspeed_mm.fsdp.optimizer.lr_scheduler import get_linear_schedule_with_warmup

        optimizer = _make_optimizer(lr=1.0)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            init_lr=1.0,
        )

        observed = _scheduler_lrs(scheduler, len(expected) - 1)

        assert observed == pytest.approx(expected)

    @pytest.mark.parametrize(
        "min_lr,expected_floor",
        [
            (0.0, 0.0),
            (1e-7, 1e-7),
            (0.01, 0.01),
            (0.1, 0.1),
            (0.5, 0.5),
        ],
    )
    def test_linear_schedule_respects_min_lr_floor(self, min_lr, expected_floor):
        from mindspeed_mm.fsdp.optimizer.lr_scheduler import get_linear_schedule_with_warmup

        optimizer = _make_optimizer(lr=1.0)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=2,
            init_lr=1.0,
            min_lr=min_lr,
        )

        lrs = _scheduler_lrs(scheduler, 5)

        assert min(lrs) == pytest.approx(expected_floor)
        assert lrs[-1] == pytest.approx(expected_floor)

    def test_linear_schedule_returns_zero_when_init_lr_is_zero(self):
        from mindspeed_mm.fsdp.optimizer.lr_scheduler import get_linear_schedule_with_warmup

        optimizer = _make_optimizer(lr=0.0)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=2,
            num_training_steps=4,
            init_lr=0.0,
        )

        assert _scheduler_lrs(scheduler, 5) == pytest.approx([0.0] * 6)


class TestCosineScheduleWithWarmup:
    @pytest.mark.parametrize(
        "step,warmup,total,decay_ratio,min_lr,expected",
        [
            (0, 0, 10, 1.0, 0.0, 1.0),
            (5, 0, 10, 1.0, 0.0, 0.5),
            (10, 0, 10, 1.0, 0.0, 0.0),
            (0, 2, 10, 1.0, 0.0, 0.0),
            (1, 2, 10, 1.0, 0.0, 0.5),
            (2, 2, 10, 1.0, 0.0, 1.0),
            (6, 2, 10, 1.0, 0.0, 0.5),
            (10, 2, 10, 1.0, 0.0, 0.0),
            (10, 2, 10, 1.0, 0.1, 0.1),
            (6, 2, 10, 1.0, 0.1, 0.55),
        ],
    )
    def test_cosine_schedule_expected_points(self, step, warmup, total, decay_ratio, min_lr, expected):
        from mindspeed_mm.fsdp.optimizer.lr_scheduler import get_cosine_schedule_with_warmup

        optimizer = _make_optimizer(lr=1.0)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup,
            num_training_steps=total,
            init_lr=1.0,
            lr_decay_ratio=decay_ratio,
            min_lr=min_lr,
        )

        observed = _scheduler_lrs(scheduler, step)[-1]

        assert observed == pytest.approx(expected)

    def test_cosine_schedule_uses_min_lr_after_decay_steps(self):
        from mindspeed_mm.fsdp.optimizer.lr_scheduler import get_cosine_schedule_with_warmup

        optimizer = _make_optimizer(lr=1.0)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=10,
            init_lr=1.0,
            lr_decay_ratio=0.5,
            min_lr=0.25,
        )

        lrs = _scheduler_lrs(scheduler, 8)

        assert lrs[5] == pytest.approx(0.25)
        assert lrs[6] == pytest.approx(0.25)
        assert lrs[8] == pytest.approx(0.25)

    def test_cosine_schedule_returns_min_lr_when_warmup_exceeds_decay_window(self):
        from mindspeed_mm.fsdp.optimizer.lr_scheduler import get_cosine_schedule_with_warmup

        optimizer = _make_optimizer(lr=1.0)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=8,
            num_training_steps=10,
            init_lr=1.0,
            lr_decay_ratio=0.5,
            min_lr=0.2,
        )

        lrs = _scheduler_lrs(scheduler, 10)

        assert lrs[7] == pytest.approx(0.875)
        assert lrs[8] == pytest.approx(0.2)
        assert lrs[10] == pytest.approx(0.2)


class TestBuildLRScheduler:
    @pytest.mark.parametrize(
        "decay_style",
        [
            "constant",
            "linear",
            "cosine",
        ],
    )
    def test_build_lr_scheduler_returns_lambda_lr_for_supported_styles(self, decay_style):
        from torch.optim.lr_scheduler import LambdaLR
        from mindspeed_mm.fsdp.optimizer.lr_scheduler import build_lr_scheduler

        optimizer = _make_optimizer(lr=0.5)

        scheduler = build_lr_scheduler(
            optimizer,
            train_steps=10,
            lr=0.5,
            lr_decay_style=decay_style,
            lr_warmup_ratio=0.2,
            lr_min=0.01,
        )

        assert isinstance(scheduler, LambdaLR)

    def test_build_lr_scheduler_rejects_unknown_decay_style(self):
        from mindspeed_mm.fsdp.optimizer.lr_scheduler import build_lr_scheduler

        optimizer = _make_optimizer(lr=1.0)

        with pytest.raises(ValueError, match="Unknown learning rate decay style"):
            build_lr_scheduler(optimizer, train_steps=10, lr_decay_style="triangle")

    def test_build_lr_scheduler_creates_scheduler_for_each_multi_optimizer_key(self):
        from mindspeed_mm.fsdp.optimizer.lr_scheduler import MultiLRScheduler, build_lr_scheduler

        multi_optimizer = _FakeMultiOptimizer(
            OrderedDict(
                [
                    ("language", _make_optimizer(lr=1.0)),
                    ("vision", _make_optimizer(lr=1.0)),
                    ("audio", _make_optimizer(lr=1.0)),
                ]
            )
        )

        scheduler = build_lr_scheduler(
            multi_optimizer,
            train_steps=8,
            lr=1.0,
            lr_decay_style="constant",
            lr_warmup_ratio=0.25,
        )

        assert isinstance(scheduler, MultiLRScheduler)
        assert list(scheduler.keys()) == ["language", "vision", "audio"]
        assert all(hasattr(inner_scheduler, "step") for inner_scheduler in scheduler.values())


class TestMultiLRScheduler:
    def test_multi_lr_scheduler_step_delegates_to_all_schedulers(self):
        from mindspeed_mm.fsdp.optimizer.lr_scheduler import MultiLRScheduler

        scheduler = MultiLRScheduler(
            {
                "a": _FakeScheduler(0.1),
                "b": _FakeScheduler(0.2),
                "c": _FakeScheduler(0.3),
            }
        )

        scheduler.step()
        scheduler.step()

        assert scheduler["a"].step_calls == 2
        assert scheduler["b"].step_calls == 2
        assert scheduler["c"].step_calls == 2

    def test_multi_lr_scheduler_state_dict_is_keyed_by_scheduler_name(self):
        from mindspeed_mm.fsdp.optimizer.lr_scheduler import MultiLRScheduler

        scheduler = MultiLRScheduler(
            {
                "language": _FakeScheduler(0.1),
                "vision": _FakeScheduler(0.2),
            }
        )
        scheduler.step()

        assert scheduler.state_dict() == {
            "language": {"lr": 0.1, "step_calls": 1},
            "vision": {"lr": 0.2, "step_calls": 1},
        }

    def test_multi_lr_scheduler_load_state_dict_ignores_missing_keys(self):
        from mindspeed_mm.fsdp.optimizer.lr_scheduler import MultiLRScheduler

        scheduler = MultiLRScheduler(
            {
                "language": _FakeScheduler(0.1),
                "vision": _FakeScheduler(0.2),
            }
        )

        scheduler.load_state_dict({"language": {"lr": 0.5, "step_calls": 9}})

        assert scheduler["language"].lr == pytest.approx(0.5)
        assert scheduler["language"].step_calls == 9
        assert scheduler["vision"].lr == pytest.approx(0.2)
        assert scheduler["vision"].loaded_state is None

    def test_multi_lr_scheduler_get_last_lr_returns_first_scheduler_lr(self):
        from mindspeed_mm.fsdp.optimizer.lr_scheduler import MultiLRScheduler

        scheduler = MultiLRScheduler(
            OrderedDict(
                [
                    ("first", _FakeScheduler(0.123)),
                    ("second", _FakeScheduler(0.456)),
                ]
            )
        )

        assert scheduler.get_last_lr() == [0.123]

    def test_multi_lr_scheduler_get_last_lr_returns_zero_for_empty_mapping(self):
        from mindspeed_mm.fsdp.optimizer.lr_scheduler import MultiLRScheduler

        scheduler = MultiLRScheduler()

        assert scheduler.get_last_lr() == [0.0]
