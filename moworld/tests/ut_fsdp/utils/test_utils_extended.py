"""Unit tests for FSDP utility registries, dtype helpers, and dataclass helpers."""

import os
from dataclasses import dataclass, field, is_dataclass

import pytest


class TestDtype:
    @pytest.mark.parametrize(
        "name,torch_attr",
        [
            ("fp16", "float16"),
            ("bf16", "bfloat16"),
            ("fp32", "float32"),
            ("fp64", "float64"),
            ("int8", "int8"),
            ("int16", "int16"),
            ("int32", "int32"),
            ("int64", "int64"),
        ],
    )
    def test_get_dtype_maps_supported_names(self, name, torch_attr):
        torch = pytest.importorskip("torch")
        from mindspeed_mm.fsdp.utils.dtype import get_dtype

        assert get_dtype(name) is getattr(torch, torch_attr)

    @pytest.mark.parametrize(
        "name",
        [
            "",
            "float16",
            "float32",
            "FP16",
            "bf 16",
            "uint8",
            "bool",
            None,
        ],
    )
    def test_get_dtype_rejects_unsupported_names(self, name):
        pytest.importorskip("torch")
        from mindspeed_mm.fsdp.utils.dtype import get_dtype

        with pytest.raises(ValueError, match="Unsupported dtype"):
            get_dtype(name)


class TestRegister:
    def test_register_and_get_returns_original_object(self):
        pytest.importorskip("torch")
        from mindspeed_mm.fsdp.utils.register import Register

        registry = Register()

        @registry.register("model")
        class Model:
            pass

        assert registry.get("model") is Model

    def test_register_decorator_returns_registered_object(self):
        pytest.importorskip("torch")
        from mindspeed_mm.fsdp.utils.register import Register

        registry = Register()

        def fn():
            return "ok"

        decorated = registry.register("fn")(fn)

        assert decorated is fn
        assert registry.get("fn") is fn

    def test_register_rejects_duplicate_id(self):
        pytest.importorskip("torch")
        from mindspeed_mm.fsdp.utils.register import Register

        registry = Register()

        @registry.register("duplicate")
        class First:
            pass

        with pytest.raises(KeyError, match="already registered"):

            @registry.register("duplicate")
            class Second:
                pass

    @pytest.mark.parametrize(
        "missing_id",
        [
            "missing",
            "",
            None,
            123,
        ],
    )
    def test_register_get_rejects_missing_id(self, missing_id):
        pytest.importorskip("torch")
        from mindspeed_mm.fsdp.utils.register import Register

        registry = Register()

        with pytest.raises(KeyError, match="not registered"):
            registry.get(missing_id)


class TestParamsUtils:
    def test_create_nested_dataclass_builds_nested_defaults(self):
        pytest.importorskip("torch")
        from mindspeed_mm.fsdp.params.utils import create_nested_dataclass

        Config = create_nested_dataclass(
            "Config",
            {
                "model": {
                    "hidden_size": 4096,
                    "dropout": 0.1,
                },
                "enabled": True,
                "names": ["q_proj", "v_proj"],
                "tags": {"vision", "language"},
            },
        )

        cfg = Config()

        assert is_dataclass(Config)
        assert cfg.model.hidden_size == 4096
        assert cfg.model.dropout == pytest.approx(0.1)
        assert cfg.enabled is True
        assert cfg.names == []
        assert cfg.tags == set()

    def test_create_nested_dataclass_uses_independent_mutable_defaults(self):
        pytest.importorskip("torch")
        from mindspeed_mm.fsdp.params.utils import create_nested_dataclass

        Config = create_nested_dataclass(
            "Config",
            {
                "items": [],
                "labels": set(),
                "nested": {"values": []},
            },
        )

        left = Config()
        right = Config()

        left.items.append("left")
        left.labels.add("tag")
        left.nested.values.append(1)

        assert right.items == []
        assert right.labels == set()
        assert right.nested.values == []

    def test_allow_extra_fields_adds_unknown_scalar_attributes(self, monkeypatch):
        pytest.importorskip("torch")
        import mindspeed_mm.fsdp.params.utils as params_utils

        monkeypatch.setattr(params_utils, "print_rank", lambda *args, **kwargs: None)

        @params_utils.allow_extra_fields
        @dataclass
        class Config:
            name: str = "default"

        cfg = Config(name="custom", lr=0.01, enabled=True)

        assert cfg.name == "custom"
        assert cfg.lr == pytest.approx(0.01)
        assert cfg.enabled is True
        assert cfg._extra_fields == {"lr": 0.01, "enabled": True}

    def test_allow_extra_fields_adds_unknown_nested_dict_as_dataclass(self, monkeypatch):
        pytest.importorskip("torch")
        import mindspeed_mm.fsdp.params.utils as params_utils

        monkeypatch.setattr(params_utils, "print_rank", lambda *args, **kwargs: None)

        @params_utils.allow_extra_fields
        @dataclass
        class Config:
            name: str = "default"

        cfg = Config(
            name="custom",
            optimizer={
                "lr": 0.001,
                "betas": {
                    "beta1": 0.9,
                    "beta2": 0.95,
                },
            },
        )

        assert cfg.optimizer.lr == pytest.approx(0.001)
        assert cfg.optimizer.betas.beta1 == pytest.approx(0.9)
        assert cfg.optimizer.betas.beta2 == pytest.approx(0.95)
        assert cfg._extra_fields == {
            "optimizer": {
                "lr": 0.001,
                "betas": {
                    "beta1": 0.9,
                    "beta2": 0.95,
                },
            }
        }

    def test_instantiate_dataclass_recursively_instantiates_nested_dataclasses(self, monkeypatch):
        pytest.importorskip("torch")
        import mindspeed_mm.fsdp.params.utils as params_utils

        monkeypatch.setattr(params_utils, "print_rank", lambda *args, **kwargs: None)

        @params_utils.allow_extra_fields
        @dataclass
        class Inner:
            width: int = 1
            depth: int = 2

        @params_utils.allow_extra_fields
        @dataclass
        class Outer:
            inner: Inner = field(default_factory=Inner)
            name: str = "outer"

        cfg = params_utils.instantiate_dataclass(
            Outer,
            {
                "name": "configured",
                "inner": {
                    "width": 16,
                    "depth": 32,
                },
                "new_field": "accepted",
            },
        )

        assert isinstance(cfg, Outer)
        assert isinstance(cfg.inner, Inner)
        assert cfg.name == "configured"
        assert cfg.inner.width == 16
        assert cfg.inner.depth == 32
        assert cfg.new_field == "accepted"

    def test_instantiate_dataclass_returns_input_for_non_dataclass_type(self):
        pytest.importorskip("torch")
        from mindspeed_mm.fsdp.params.utils import instantiate_dataclass

        data = {"a": 1}

        assert instantiate_dataclass(dict, data) is data

    def test_instantiate_dataclass_wraps_type_hint_failures(self, monkeypatch):
        pytest.importorskip("torch")
        import mindspeed_mm.fsdp.params.utils as params_utils

        @dataclass
        class Config:
            value: "MissingType"

        with pytest.raises(RuntimeError, match="Failed to get type hints"):
            params_utils.instantiate_dataclass(Config, {"value": 1})


class TestRuntimeUtils:
    def test_configure_hsdp_gradient_sync_sets_both_flags(self):
        pytest.importorskip("torch")
        from mindspeed_mm.fsdp.utils.utils import configure_hsdp_gradient_sync

        class Model:
            def __init__(self):
                self.last_backward_values = []
                self.requires_all_reduce_values = []

            def set_is_last_backward(self, value):
                self.last_backward_values.append(value)

            def set_requires_all_reduce(self, value):
                self.requires_all_reduce_values.append(value)

        model = Model()

        configure_hsdp_gradient_sync(model, True)
        configure_hsdp_gradient_sync(model, False)

        assert model.last_backward_values == [True, False]
        assert model.requires_all_reduce_values == [True, False]

    def test_singleton_metaclass_reuses_instance(self):
        pytest.importorskip("torch")
        from mindspeed_mm.fsdp.utils.decorators import Singleton

        class Example(metaclass=Singleton):
            def __init__(self, value):
                self.value = value

        try:
            first = Example("first")
            second = Example("second")

            assert first is second
            assert second.value == "first"
        finally:
            Singleton._instances.pop(Example, None)
