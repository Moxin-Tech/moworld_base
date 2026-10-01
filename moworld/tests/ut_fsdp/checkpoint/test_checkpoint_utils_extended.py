"""Unit tests for FSDP checkpoint path, metadata, and key migration helpers."""

import os
import tempfile

import pytest


class TestCheckpointNameAndTracker:
    @pytest.mark.parametrize(
        "iteration,expected_suffix",
        [
            (1, "iter_0000001"),
            (7, "iter_0000007"),
            (10, "iter_0000010"),
            (42, "iter_0000042"),
            (99, "iter_0000099"),
            (100, "iter_0000100"),
            (321, "iter_0000321"),
            (999, "iter_0000999"),
            (1000, "iter_0001000"),
            (2024, "iter_0002024"),
            (9999, "iter_0009999"),
            (10000, "iter_0010000"),
            (123456, "iter_0123456"),
            (999999, "iter_0999999"),
            (1000000, "iter_1000000"),
        ],
    )
    def test_get_checkpoint_name_zero_pads_iterations(self, iteration, expected_suffix):
        from mindspeed_mm.fsdp.checkpoint.utils import get_checkpoint_name

        assert get_checkpoint_name("/tmp/checkpoints", iteration).endswith(expected_suffix)

    @pytest.mark.parametrize(
        "root",
        [
            "/tmp/checkpoints",
            "/tmp/checkpoints/",
            "relative/checkpoints",
            ".",
            "",
        ],
    )
    def test_get_checkpoint_tracker_filename_uses_latest_checkpoint_file(self, root):
        from mindspeed_mm.fsdp.checkpoint.utils import get_checkpoint_tracker_filename

        tracker = get_checkpoint_tracker_filename(root)
        assert tracker == os.path.join(root, "latest_checkpointed_iteration.txt")

    def test_get_checkpoint_name_release_ignores_iteration(self):
        from mindspeed_mm.fsdp.checkpoint.utils import get_checkpoint_name

        assert get_checkpoint_name("/tmp/checkpoints", 1, release=True) == os.path.join("/tmp/checkpoints", "release")
        assert get_checkpoint_name("/tmp/checkpoints", 999999, release=True) == os.path.join(
            "/tmp/checkpoints", "release"
        )

    @pytest.mark.parametrize(
        "metadata,expected_iteration,expected_release",
        [
            ("1", 1, False),
            ("7", 7, False),
            ("10\n", 10, False),
            (" 42 ", 42, False),
            ("\t99\n", 99, False),
            ("123456", 123456, False),
            ("release", 0, True),
            (" release\n", 0, True),
        ],
    )
    def test_read_metadata_accepts_iteration_or_release(self, metadata, expected_iteration, expected_release):
        from mindspeed_mm.fsdp.checkpoint.utils import read_metadata

        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = os.path.join(temp_dir, "latest.txt")
            with open(tracker, "w", encoding="utf-8") as handle:
                handle.write(metadata)

            iteration, release = read_metadata(tracker)

        assert iteration == expected_iteration
        assert release is expected_release

    @pytest.mark.parametrize(
        "metadata",
        [
            "",
            "0",
            "-1",
            "latest",
            "Release",
            "release-candidate",
            "1.5",
            "iter_0000001",
            "None",
            "null",
        ],
    )
    def test_read_metadata_rejects_invalid_non_release_values(self, metadata):
        from mindspeed_mm.fsdp.checkpoint.utils import read_metadata

        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = os.path.join(temp_dir, "latest.txt")
            with open(tracker, "w", encoding="utf-8") as handle:
                handle.write(metadata)

            if metadata in {"0", "-1"}:
                iteration, release = read_metadata(tracker)
                assert iteration == int(metadata)
                assert release is False
            else:
                with pytest.raises(ValueError, match="Invalid metadata file"):
                    read_metadata(tracker)


class TestBaseLayerKeyMigration:
    @pytest.mark.parametrize(
        "state_dict,expected_mapping,expected_keys",
        [
            (
                {"model.layer.base_layer.weight": 1},
                {"model.layer.base_layer.weight": "model.layer.weight"},
                {"model.layer.weight"},
            ),
            (
                {"model.layer.base_layer.bias": 2},
                {"model.layer.base_layer.bias": "model.layer.bias"},
                {"model.layer.bias"},
            ),
            (
                {"a.base_layer.b.base_layer.c": 3},
                {"a.base_layer.b.base_layer.c": "a.b.c"},
                {"a.b.c"},
            ),
            (
                {"no_base_layer.weight": 4},
                {},
                {"no_base_layer.weight"},
            ),
            (
                {"encoder.0.base_layer.weight": 5, "encoder.1.weight": 6},
                {"encoder.0.base_layer.weight": "encoder.0.weight"},
                {"encoder.0.weight", "encoder.1.weight"},
            ),
            (
                {"lm_head.base_layer.weight": 7, "lm_head.base_layer.bias": 8},
                {
                    "lm_head.base_layer.weight": "lm_head.weight",
                    "lm_head.base_layer.bias": "lm_head.bias",
                },
                {"lm_head.weight", "lm_head.bias"},
            ),
        ],
    )
    def test_remove_base_layer_keys_rewrites_matching_keys_in_place(
        self,
        state_dict,
        expected_mapping,
        expected_keys,
    ):
        from mindspeed_mm.fsdp.checkpoint.utils import remove_base_layer_keys

        original_id = id(state_dict)
        mapping = remove_base_layer_keys(state_dict)

        assert id(state_dict) == original_id
        assert mapping == expected_mapping
        assert set(state_dict.keys()) == expected_keys

    @pytest.mark.parametrize(
        "bad_state_dict",
        [
            None,
            [],
            (),
            "model.layer.base_layer.weight",
            123,
            object(),
        ],
    )
    def test_remove_base_layer_keys_returns_empty_mapping_for_non_dict_inputs(self, bad_state_dict):
        from mindspeed_mm.fsdp.checkpoint.utils import remove_base_layer_keys

        assert remove_base_layer_keys(bad_state_dict) == {}

    def test_remove_base_layer_keys_preserves_values_when_keys_move(self):
        from mindspeed_mm.fsdp.checkpoint.utils import remove_base_layer_keys

        value = object()
        state_dict = {
            "module.base_layer.weight": value,
            "module.other.weight": "kept",
        }

        mapping = remove_base_layer_keys(state_dict)

        assert mapping == {"module.base_layer.weight": "module.weight"}
        assert state_dict["module.weight"] is value
        assert state_dict["module.other.weight"] == "kept"

    def test_restore_base_layer_keys_moves_rewritten_keys_back(self):
        from mindspeed_mm.fsdp.checkpoint.utils import remove_base_layer_keys, restore_base_layer_keys

        state_dict = {
            "model.layers.0.base_layer.weight": "w0",
            "model.layers.0.base_layer.bias": "b0",
            "model.layers.1.weight": "w1",
        }
        mapping = remove_base_layer_keys(state_dict)

        restore_base_layer_keys(state_dict, mapping)

        assert state_dict == {
            "model.layers.0.base_layer.weight": "w0",
            "model.layers.0.base_layer.bias": "b0",
            "model.layers.1.weight": "w1",
        }

    def test_restore_base_layer_keys_is_noop_for_invalid_inputs(self):
        from mindspeed_mm.fsdp.checkpoint.utils import restore_base_layer_keys

        restore_base_layer_keys(None, {"a": "b"})
        restore_base_layer_keys([], {"a": "b"})
        restore_base_layer_keys({"a": 1}, None)
        restore_base_layer_keys({"a": 1}, [])

    def test_restore_base_layer_keys_ignores_mapping_entries_not_present(self):
        from mindspeed_mm.fsdp.checkpoint.utils import restore_base_layer_keys

        state_dict = {"kept.weight": 1}
        restore_base_layer_keys(
            state_dict,
            {
                "missing.base_layer.weight": "missing.weight",
                "other.base_layer.bias": "other.bias",
            },
        )

        assert state_dict == {"kept.weight": 1}

    def test_restore_base_layer_keys_handles_partial_restore(self):
        from mindspeed_mm.fsdp.checkpoint.utils import restore_base_layer_keys

        state_dict = {
            "layer.weight": "restored",
            "layer.bias": "left alone because missing from reverse mapping",
        }
        mapping = {
            "layer.base_layer.weight": "layer.weight",
        }

        restore_base_layer_keys(state_dict, mapping)

        assert state_dict == {
            "layer.base_layer.weight": "restored",
            "layer.bias": "left alone because missing from reverse mapping",
        }

    @pytest.mark.parametrize(
        "keys",
        [
            ["a.base_layer.weight"],
            ["a.base_layer.weight", "a.base_layer.bias"],
            ["a.base_layer.weight", "b.base_layer.weight", "c.weight"],
            ["prefix.base_layer.inner.base_layer.weight", "untouched"],
            ["adapter.base_layer.lora_A.weight", "adapter.base_layer.lora_B.weight"],
        ],
    )
    def test_remove_then_restore_round_trips_key_sets(self, keys):
        from mindspeed_mm.fsdp.checkpoint.utils import remove_base_layer_keys, restore_base_layer_keys

        original = {key: idx for idx, key in enumerate(keys)}
        state_dict = dict(original)

        mapping = remove_base_layer_keys(state_dict)
        restore_base_layer_keys(state_dict, mapping)

        assert state_dict == original
