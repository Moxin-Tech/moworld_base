import tempfile
from pathlib import Path
import pytest


class TestUpdateSafetensorsFiles:
    def test_update_adds_new_keys_to_existing_file(self):
        torch = pytest.importorskip("torch")
        from safetensors.torch import load_file, save_file
        from checkpoint.common.merge_dcp_to_hf import update_safetensors_files

        with tempfile.TemporaryDirectory() as td:
            existing = {"layer.0.weight": torch.randn(2, 3)}
            save_file(existing, Path(td) / "model.safetensors-00001-of-00001.safetensors")

            weight_map = {
                "layer.0.weight": "model.safetensors-00001-of-00001.safetensors",
                "mtp.layers.0.mlp.experts.0.gate_proj.weight": "model.safetensors-00001-of-00001.safetensors",
            }
            new_weights = {"mtp.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(4, 5)}

            update_safetensors_files(Path(td), new_weights, weight_map)

            result = load_file(Path(td) / "model.safetensors-00001-of-00001.safetensors")
            assert "layer.0.weight" in result
            assert "mtp.layers.0.mlp.experts.0.gate_proj.weight" in result
            assert result["mtp.layers.0.mlp.experts.0.gate_proj.weight"].shape == (4, 5)

    def test_update_distributes_keys_across_multiple_files(self):
        torch = pytest.importorskip("torch")
        from safetensors.torch import load_file, save_file
        from checkpoint.common.merge_dcp_to_hf import update_safetensors_files

        with tempfile.TemporaryDirectory() as td:
            file_a = {"layer.0.weight": torch.randn(2, 3)}
            file_b = {"layer.1.weight": torch.randn(3, 4)}
            save_file(file_a, Path(td) / "model.safetensors-00001-of-00002.safetensors")
            save_file(file_b, Path(td) / "model.safetensors-00002-of-00002.safetensors")

            weight_map = {
                "layer.0.weight": "model.safetensors-00001-of-00002.safetensors",
                "layer.1.weight": "model.safetensors-00002-of-00002.safetensors",
            }
            new_weights = {
                "layer.0.weight": torch.randn(4, 5),
                "layer.1.weight": torch.randn(5, 6),
            }

            update_safetensors_files(Path(td), new_weights, weight_map)

            result_a = load_file(Path(td) / "model.safetensors-00001-of-00002.safetensors")
            result_b = load_file(Path(td) / "model.safetensors-00002-of-00002.safetensors")
            assert "layer.0.weight" in result_a
            assert "layer.1.weight" in result_b
            assert "layer.0.weight" not in result_b
            assert "layer.1.weight" not in result_a
            assert result_a["layer.0.weight"].shape == (4, 5)
            assert result_b["layer.1.weight"].shape == (5, 6)

    def test_update_skips_keys_not_in_weight_map(self):
        torch = pytest.importorskip("torch")
        from safetensors.torch import load_file, save_file
        from checkpoint.common.merge_dcp_to_hf import update_safetensors_files

        with tempfile.TemporaryDirectory() as td:
            existing = {"layer.0.weight": torch.randn(2, 3)}
            save_file(existing, Path(td) / "model.safetensors-00001-of-00001.safetensors")

            weight_map = {"layer.0.weight": "model.safetensors-00001-of-00001.safetensors"}
            new_weights = {"unknown.key.weight": torch.randn(4, 5)}

            update_safetensors_files(Path(td), new_weights, weight_map)

            result = load_file(Path(td) / "model.safetensors-00001-of-00001.safetensors")
            assert "unknown.key.weight" not in result
            assert "layer.0.weight" in result
