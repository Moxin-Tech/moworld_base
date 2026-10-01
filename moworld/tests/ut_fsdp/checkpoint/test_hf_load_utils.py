# pylint: skip-file
"""Unit tests for mindspeed_mm.fsdp.checkpoint.hf_load_utils.

Single-process tests cover every primitive and end-to-end ``load_hf_weights``
on a plain (non-FSDP) model. The multi-rank test for
``rank0_load_and_broadcast_hf_weights`` follows the project convention used by
``test_parallel_state_multi_rank.py``: ``mp.spawn`` real NPU processes with the
``hccl`` backend, auto-skipped when fewer than 2 NPUs are available.
"""

import json
import logging
import os
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

os.environ.setdefault("NON_MEGATRON", "true")


# ===========================================================================
# Helpers
# ===========================================================================
def _make_single_safetensors(tmpdir: str, weights: dict) -> str:
    path = os.path.join(tmpdir, "model.safetensors")
    save_file(weights, path)
    return path


def _make_sharded_safetensors(tmpdir: str, shards: dict) -> None:
    weight_map = {}
    total_size = 0
    for shard_name, tensors in shards.items():
        save_file(tensors, os.path.join(tmpdir, shard_name))
        for k, v in tensors.items():
            weight_map[k] = shard_name
            total_size += v.numel() * v.element_size()
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(tmpdir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f)


# ===========================================================================
# Pure-function primitives (read / write / log / retie)
# ===========================================================================
class TestPrimitives:
    @pytest.mark.parametrize("name,expected_local", [
        ("level1.level2.weight", "weight"),   # nested
        ("weight", "weight"),                 # root
        ("layers.1.weight", "weight"),        # ModuleList index
    ])
    def test_resolve_leaf_walks_to_owner(self, name, expected_local):
        from mindspeed_mm.fsdp.checkpoint.hf_load_utils import _resolve_leaf

        model = nn.Module()
        model.level1 = nn.Module()
        model.level1.level2 = nn.Linear(4, 4)
        model.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        model.weight = nn.Parameter(torch.zeros(4))

        leaf, local = _resolve_leaf(model, name)
        assert local == expected_local
        assert local in leaf._parameters

    def test_resolve_leaf_missing_path_raises(self):
        from mindspeed_mm.fsdp.checkpoint.hf_load_utils import _resolve_leaf

        with pytest.raises(ValueError, match="Cannot resolve"):
            _resolve_leaf(nn.Linear(4, 4), "no.such.path.weight")

    @pytest.mark.parametrize("mapping,key,expected", [
        (None,                          "x.weight",  "x.weight"),  # no attribute
        ({},                            "x.weight",  "x.weight"),  # empty
        ({r"^foo": "bar"},              "x.weight",  "x.weight"),  # no match
        ({r"^model": "language_model"}, "model.x",   "language_model.x"),
        ({r"^foo": "^bar(group)"},      "foo.x",     "bar.x"),     # strips ^ and ()
    ])
    def test_convert_weight_key(self, mapping, key, expected):
        from mindspeed_mm.fsdp.checkpoint.hf_load_utils import convert_weight_key

        model = nn.Module()
        if mapping is not None:
            model._checkpoint_conversion_mapping = mapping
        assert convert_weight_key(key, model) == expected

    def test_write_full_tensor_parameter_and_buffer(self):
        from mindspeed_mm.fsdp.checkpoint.hf_load_utils import write_full_tensor

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 8)
                self.register_buffer("buf", torch.zeros(4))

        m = M()
        new_w = torch.randn(8, 4)
        new_buf = torch.randn(4)
        write_full_tensor(m, "linear.weight", new_w)
        write_full_tensor(m, "buf", new_buf)
        assert torch.allclose(m.linear.weight, new_w)
        assert torch.allclose(m.buf, new_buf)

    def test_write_full_tensor_casts_dtype(self):
        from mindspeed_mm.fsdp.checkpoint.hf_load_utils import write_full_tensor

        m = nn.Linear(4, 8)
        src = torch.randn(8, 4, dtype=torch.bfloat16)
        write_full_tensor(m, "weight", src)
        assert m.weight.dtype == torch.float32
        assert torch.allclose(m.weight, src.float(), atol=1e-2)

    def test_write_full_tensor_unknown_name_raises(self):
        from mindspeed_mm.fsdp.checkpoint.hf_load_utils import write_full_tensor

        with pytest.raises(ValueError, match="neither a parameter nor a buffer"):
            write_full_tensor(nn.Linear(4, 4), "nope", torch.randn(4, 4))


# ===========================================================================
# File location / detection
# ===========================================================================
class TestLocateAndDetect:
    @pytest.mark.parametrize("setup,expected", [
        ("none",        False),  # path is None
        ("nonexistent", False),
        ("empty",       False),
        ("single",      True),   # has model.safetensors
        ("index",       True),   # has model.safetensors.index.json
        ("dcp",         False),  # DCP tracker only
    ])
    def test_looks_like_hf_weight_dir(self, setup, expected, tmp_path):
        from mindspeed_mm.fsdp.checkpoint.hf_load_utils import looks_like_hf_weight_dir

        if setup == "none":
            assert looks_like_hf_weight_dir(None) is False
            return
        if setup == "nonexistent":
            assert looks_like_hf_weight_dir(str(tmp_path / "nope")) is False
            return
        if setup == "single":
            _make_single_safetensors(str(tmp_path), {"w": torch.zeros(2)})
        elif setup == "index":
            (tmp_path / "model.safetensors.index.json").write_text("{}")
        elif setup == "dcp":
            (tmp_path / "latest_checkpointed_iteration.txt").write_text("release")
        # else: empty
        assert looks_like_hf_weight_dir(str(tmp_path)) is expected

    def test_locate_single_file(self):
        from mindspeed_mm.fsdp.checkpoint.hf_load_utils import locate_hf_weight_files

        with tempfile.TemporaryDirectory() as td:
            _make_single_safetensors(td, {"w": torch.randn(2, 3)})
            streams = locate_hf_weight_files(td)
        assert len(streams) == 1
        assert streams[0].filepath.endswith("model.safetensors")

    def test_locate_sharded_sorted(self):
        from mindspeed_mm.fsdp.checkpoint.hf_load_utils import locate_hf_weight_files

        with tempfile.TemporaryDirectory() as td:
            _make_sharded_safetensors(td, {
                "model-00002-of-00002.safetensors": {"b": torch.randn(2, 3)},
                "model-00001-of-00002.safetensors": {"a": torch.randn(2, 3)},
            })
            streams = locate_hf_weight_files(td)
        assert [os.path.basename(s.filepath) for s in streams] == [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ]

    def test_locate_empty_dir_raises(self):
        from mindspeed_mm.fsdp.checkpoint.hf_load_utils import locate_hf_weight_files

        with tempfile.TemporaryDirectory() as td, \
             pytest.raises(ValueError, match="No HF safetensors"):
            locate_hf_weight_files(td)

    def test_locate_index_references_missing_shard(self):
        from mindspeed_mm.fsdp.checkpoint.hf_load_utils import locate_hf_weight_files

        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "model.safetensors.index.json"), "w") as f:
                json.dump({"metadata": {}, "weight_map": {"a": "missing.safetensors"}}, f)
            with pytest.raises(FileNotFoundError):
                locate_hf_weight_files(td)

    def test_hf_weight_file_stream_iterates_all(self):
        from mindspeed_mm.fsdp.checkpoint.hf_load_utils import HFWeightFileStream

        weights = {"a": torch.randn(2, 3), "b": torch.randn(4)}
        with tempfile.TemporaryDirectory() as td:
            path = _make_single_safetensors(td, weights)
            collected = dict(HFWeightFileStream(path))
        assert set(collected.keys()) == set(weights.keys())
        for k in weights:
            assert torch.allclose(collected[k], weights[k])


# ===========================================================================
# _log_unexpected_keys (single-rank only; rank-gating is exercised by the
# multi-rank end-to-end test below)
# ===========================================================================
class TestLogUnexpectedKeys:
    def test_empty_does_not_log(self, caplog):
        from mindspeed_mm.fsdp.checkpoint.hf_load_utils import _log_unexpected_keys

        with caplog.at_level(logging.INFO):
            _log_unexpected_keys(set())
        assert len(caplog.records) == 0

    def test_logs_count_and_samples_without_truncation(self, caplog):
        from mindspeed_mm.fsdp.checkpoint.hf_load_utils import _log_unexpected_keys

        with caplog.at_level(logging.INFO):
            _log_unexpected_keys({"k_a", "k_b", "k_c"})
        msg = caplog.records[0].getMessage()
        assert "3 key(s) not present" in msg
        assert "k_a" in msg and "k_b" in msg and "k_c" in msg
        assert "showing 5 of" not in msg

    def test_truncates_above_five(self, caplog):
        from mindspeed_mm.fsdp.checkpoint.hf_load_utils import _log_unexpected_keys

        keys = {f"k_{i:02d}" for i in range(12)}
        with caplog.at_level(logging.INFO):
            _log_unexpected_keys(keys)
        msg = caplog.records[0].getMessage()
        assert "12 key(s)" in msg
        assert "showing 5 of 12" in msg


# ===========================================================================
# _retie_embeddings
# ===========================================================================
def _make_tieable_model(tie: bool):
    class TieableModel(nn.Module):
        def __init__(self, tie):
            super().__init__()
            self.embed = nn.Embedding(10, 4)
            self.lm_head = nn.Linear(4, 10, bias=False)
            self.config = SimpleNamespace(tie_word_embeddings=tie)

        def get_input_embeddings(self):
            return self.embed

        def get_output_embeddings(self):
            return self.lm_head

    return TieableModel(tie)


class TestRetieEmbeddings:
    def test_tie_true_restores_object_sharing(self):
        from mindspeed_mm.fsdp.checkpoint.hf_load_utils import _retie_embeddings

        m = _make_tieable_model(tie=True)
        assert m.embed.weight is not m.lm_head.weight
        _retie_embeddings(m)
        assert m.embed.weight is m.lm_head.weight
        m.embed.weight.data.fill_(0.0)
        assert torch.all(m.lm_head.weight == 0)

    def test_tie_false_is_noop(self):
        from mindspeed_mm.fsdp.checkpoint.hf_load_utils import _retie_embeddings

        m = _make_tieable_model(tie=False)
        _retie_embeddings(m)
        assert m.embed.weight is not m.lm_head.weight

    def test_model_without_config_is_skipped(self):
        from mindspeed_mm.fsdp.checkpoint.hf_load_utils import _retie_embeddings

        _retie_embeddings(nn.Linear(4, 4))  # must not raise

    def test_nested_text_config_and_rule(self):
        """Outer tie=True but inner text_config tie=False -> no tie (AND rule)."""
        from mindspeed_mm.fsdp.checkpoint.hf_load_utils import _retie_embeddings

        class Cfg:
            tie_word_embeddings = True

            def get_text_config(self, decoder=False):
                return SimpleNamespace(tie_word_embeddings=False)

        m = _make_tieable_model(tie=True)
        m.config = Cfg()
        _retie_embeddings(m)
        assert m.embed.weight is not m.lm_head.weight


# ===========================================================================
# load_hf_weights end-to-end (single-rank, plain Module, no FSDP/DTensor)
# ===========================================================================
class TestLoadHfWeightsE2E:
    def _make_model(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(10, 4)
                self.layer = nn.Linear(4, 8)
                self.lm_head = nn.Linear(8, 10, bias=False)

        return TinyModel()

    def _matching_weights(self, model):
        return {n: torch.randn_like(p) for n, p in model.named_parameters()}

    def test_full_load_round_trip(self):
        from mindspeed_mm.fsdp.checkpoint.hf_load_utils import load_hf_weights

        model, target = self._make_model(), None
        target = self._matching_weights(model)
        with tempfile.TemporaryDirectory() as td:
            _make_single_safetensors(td, target)
            load_hf_weights(model, td)
        loaded = dict(model.named_parameters())
        for n, expected in target.items():
            assert torch.allclose(loaded[n], expected), n

    def test_sharded_load_round_trip(self):
        from mindspeed_mm.fsdp.checkpoint.hf_load_utils import load_hf_weights

        model = self._make_model()
        target = self._matching_weights(model)
        keys = sorted(target.keys())
        with tempfile.TemporaryDirectory() as td:
            _make_sharded_safetensors(td, {
                "model-00001-of-00002.safetensors": {k: target[k] for k in keys[:2]},
                "model-00002-of-00002.safetensors": {k: target[k] for k in keys[2:]},
            })
            load_hf_weights(model, td)
        loaded = dict(model.named_parameters())
        for n, expected in target.items():
            assert torch.allclose(loaded[n], expected), n

    def test_missing_key_warns(self, caplog):
        from mindspeed_mm.fsdp.checkpoint.hf_load_utils import load_hf_weights

        model = self._make_model()
        target = self._matching_weights(model)
        target.pop("lm_head.weight")
        with tempfile.TemporaryDirectory() as td:
            _make_single_safetensors(td, target)
            with caplog.at_level(logging.WARNING):
                load_hf_weights(model, td)
        assert any("lm_head.weight" in r.getMessage() and "absent" in r.getMessage()
                   for r in caplog.records)

    def test_load_strict_raises_on_missing(self):
        from mindspeed_mm.fsdp.checkpoint.hf_load_utils import load_hf_weights

        model = self._make_model()
        target = self._matching_weights(model)
        target.pop("lm_head.weight")
        with tempfile.TemporaryDirectory() as td:
            _make_single_safetensors(td, target)
            with pytest.raises(RuntimeError, match="load_strict"):
                load_hf_weights(model, td, load_strict=True)

    def test_unexpected_key_summary_logged(self, caplog):
        from mindspeed_mm.fsdp.checkpoint.hf_load_utils import load_hf_weights

        model = self._make_model()
        target = self._matching_weights(model)
        target["stranger.weight"] = torch.randn(3, 3)
        with tempfile.TemporaryDirectory() as td:
            _make_single_safetensors(td, target)
            with caplog.at_level(logging.INFO):
                load_hf_weights(model, td)
        msgs = [r.getMessage() for r in caplog.records]
        assert any("1 key(s) not present" in m and "stranger.weight" in m for m in msgs)


# ===========================================================================
# LoRA support: inject_adapter_in_model puts the base weight under .base_layer
# and adds lora_A/lora_B; enable_lora translates the HF bare key to base_layer.
# ===========================================================================
class _FakeLoraLinear(nn.Module):
    """Mimics a PEFT LoraLayer: original Linear under .base_layer + lora_A/lora_B."""

    def __init__(self, in_f, out_f, r=2):
        super().__init__()
        self.base_layer = nn.Linear(in_f, out_f, bias=False)
        self.lora_A = nn.Linear(in_f, r, bias=False)
        self.lora_B = nn.Linear(r, out_f, bias=False)


class TestLora:
    def test_base_key_map(self):
        from mindspeed_mm.fsdp.checkpoint.hf_load_utils import _lora_base_key_map

        names = {"l0.q.base_layer.weight", "l0.q.lora_A.default.weight", "norm.weight"}
        assert _lora_base_key_map(names) == {"l0.q.weight": "l0.q.base_layer.weight"}
        assert _lora_base_key_map({"a.weight", "b.bias"}) == {}  # non-LoRA -> empty

    def test_enable_lora_loads_base_and_keeps_adapter(self, caplog):
        from mindspeed_mm.fsdp.checkpoint.hf_load_utils import load_hf_weights

        model = nn.Module()
        model.q_proj = _FakeLoraLinear(4, 8)  # LoRA target
        model.norm = nn.LayerNorm(4)          # non-target, plain
        target = {
            "q_proj.weight": torch.randn(8, 4),
            "norm.weight": torch.randn(4),
            "norm.bias": torch.randn(4),
        }
        lora_a_before = model.q_proj.lora_A.weight.detach().clone()
        with tempfile.TemporaryDirectory() as td:
            _make_single_safetensors(td, target)
            with caplog.at_level(logging.WARNING):
                load_hf_weights(model, td, enable_lora=True)
        assert torch.allclose(model.q_proj.base_layer.weight, target["q_proj.weight"])  # bare->base_layer
        assert torch.allclose(model.norm.weight, target["norm.weight"])                 # plain loaded
        assert torch.allclose(model.q_proj.lora_A.weight, lora_a_before)                # adapter untouched
        assert not any("absent" in r.getMessage() for r in caplog.records)             # lora not false-missing


# ===========================================================================
# rank0_load_and_broadcast_hf_weights (multi-rank)
#
# Follows tests/ut/fsdp/distributed/test_parallel_state_multi_rank.py: spawn real
# NPU worker processes with the hccl backend; auto-skip when < 2 NPUs.
# ===========================================================================
def _rank0_broadcast_worker(rank: int, world_size: int, init_file: str, hf_dir: str):
    import torch
    import torch.distributed as dist
    import torch.nn as nn

    from mindspeed_mm.fsdp.checkpoint.hf_load_utils import (
        rank0_load_and_broadcast_hf_weights,
    )

    if hasattr(torch, "npu"):
        torch.npu.set_device(rank)
    dist.init_process_group(
        backend="hccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(8, 4)
                self.layer = nn.Linear(4, 4)
                self.head = nn.Linear(4, 8, bias=False)

        model = TinyModel().npu()
        rank0_load_and_broadcast_hf_weights(model, hf_dir)

        # Every rank should now hold the file's weights.
        from safetensors import safe_open

        with safe_open(
            os.path.join(hf_dir, "model.safetensors"), framework="pt", device="cpu"
        ) as f:
            for k in f.keys():
                expected = f.get_tensor(k)
                got = dict(model.named_parameters())[k].detach().cpu()
                assert torch.allclose(got, expected), f"rank {rank}: mismatch {k}"
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class TestRank0LoadAndBroadcast:
    def test_end_to_end_multi_rank(self):
        import torch.multiprocessing as mp

        if not hasattr(torch, "npu") or torch.npu.device_count() < 2:
            pytest.skip("需要至少 2 张 NPU 才能运行该多卡用例")

        world_size = 2
        with tempfile.TemporaryDirectory() as hf_dir:
            _make_single_safetensors(hf_dir, {
                "embed.weight": torch.randn(8, 4),
                "layer.weight": torch.randn(4, 4),
                "layer.bias":   torch.randn(4),
                "head.weight":  torch.randn(8, 4),
            })
            with tempfile.NamedTemporaryFile(delete=False) as f:
                init_file = f.name
            try:
                mp.spawn(
                    _rank0_broadcast_worker,
                    args=(world_size, init_file, hf_dir),
                    nprocs=world_size,
                    join=True,
                )
            finally:
                try:
                    os.remove(init_file)
                except OSError:
                    pass
