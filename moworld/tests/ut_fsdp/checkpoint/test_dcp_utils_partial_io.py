# pylint: skip-file
import os
from dataclasses import dataclass

import pytest


@dataclass
class _FakePlan:
    storage_data: object = None


class _Waitable:
    def __init__(self, value):
        self._value = value
        self.wait_called = 0

    def wait(self):
        self.wait_called += 1

    def value(self):
        return self._value


class _FakeWriter:
    def __init__(self):
        self.setup_called = 0
        self.prepared_local = 0
        self.prepared_global = 0
        self.written = 0
        self.finished = 0

    def storage_meta(self):
        return {"m": 1}

    def set_up_storage_writer(self, is_coordinator: bool):
        self.setup_called += 1

    def prepare_local_plan(self, plan):
        self.prepared_local += 1
        return plan

    def prepare_global_plan(self, plans):
        self.prepared_global += 1
        return plans

    def write_data(self, plan, planner):
        self.written += 1
        return _Waitable(["ok"])

    def finish(self, metadata, results):
        self.finished += 1


class _FakeReader:
    def __init__(self):
        self.setup_called = 0
        self.prepared_local = 0
        self.prepared_global = 0
        self.read_called = 0
        self._metadata = None

    def read_metadata(self):
        return self._metadata

    def set_up_storage_reader(self, metadata, is_coordinator: bool):
        self.setup_called += 1

    def prepare_local_plan(self, plan):
        self.prepared_local += 1
        return plan

    def prepare_global_plan(self, plans):
        self.prepared_global += 1
        return plans

    def read_data(self, plan, planner):
        self.read_called += 1
        return _Waitable(None)


def test_partial_save_uses_storage_meta_signature_and_prefix(monkeypatch):
    pytest.importorskip("torch")

    import mindspeed_mm.fsdp.checkpoint.dcp_utils as u

    class PlannerWithStorageMeta:
        def __init__(self):
            self.setup_kwargs = None

        def set_up_planner(self, *, state_dict, storage_meta, is_coordinator: bool):
            self.setup_kwargs = (state_dict, storage_meta, is_coordinator)

        def create_local_plan(self):
            return _FakePlan(storage_data=None)

        def create_global_plan(self, all_local_plans):
            return all_local_plans, object()

        def finish_plan(self, plan):
            return plan

    writer = _FakeWriter()
    planner = PlannerWithStorageMeta()
    meta, writes = u.partial_save_dcp_state_dict({"w", 1}, writer, planner=planner, part_idx=2)

    assert meta is not None
    assert writes == ["ok"]
    assert planner.setup_kwargs[1] == {"m": 1}
    assert writer.setup_called == 1
    assert writer.prepared_local == 1
    assert writer.prepared_global == 1
    assert writer.written == 1


def test_partial_save_legacy_signature_emits_warning(monkeypatch):
    pytest.importorskip("torch")
    import warnings

    import mindspeed_mm.fsdp.checkpoint.dcp_utils as u

    class LegacyPlanner:
        def __init__(self):
            self.setup_args = None

        # legacy signature without storage_meta
        def set_up_planner(self, state_dict, is_coordinator: bool):
            self.setup_args = (state_dict, is_coordinator)

        def create_local_plan(self):
            return _FakePlan(storage_data=None)

        def create_global_plan(self, all_local_plans):
            return all_local_plans, object()

        def finish_plan(self, plan):
            return plan

    writer = _FakeWriter()
    planner = LegacyPlanner()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        u.partial_save_dcp_state_dict({"w", 1}, writer, planner=planner)
        assert any("SavePlanner.set_up_planner" in str(x.message) for x in w)
    assert planner.setup_args == ({"w", 1}, True)


def test_partial_load_waits_and_populates_state_dict(monkeypatch):
    pytest.importorskip("torch")

    import mindspeed_mm.fsdp.checkpoint.dcp_utils as u
    from torch.distributed.checkpoint.metadata import Metadata

    # Minimal metadata: empty, because we only validate contron flow + wait
    md = Metadata(state_dict_metadata={}, storage_data={}, planner_data={})
    reader = _FakeReader()

    class Planner:
        def __init__(self):
            self.setup = 0

        def set_up_planner(self, state_dict, metadata, is_coordinator: bool):
            self.setup += 1
            # populate a sentinel to prove the same dict instance is used e2e
            state_dict["sentinel"] = 42

        def create_local_plan(self):
            return _FakePlan()

        def create_global_plan(self, plans):
            return plans

        def finish_plan(self, plan):
            return plan

    st = u.partial_load_dcp_state_dict(metadata=md, storage_reader=reader, planner=Planner())
    assert st["sentinel"] == 42
    assert reader.setup_called == 1
    assert reader.read_called == 1
