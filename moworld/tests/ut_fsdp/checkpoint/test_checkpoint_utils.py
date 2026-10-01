# pylint: skip-file
import os
import tempfile

import pytest


class TestCheckpointUtils:
    def test_get_checkpoint_name(self):
        from mindspeed_mm.fsdp.checkpoint.utils import get_checkpoint_name

        assert get_checkpoint_name("/ckpt", 1, release=False).endswith("iter_0000001")
        assert get_checkpoint_name("/ckpt", 123, release=False).endswith("iter_0000123")
        assert get_checkpoint_name("/ckpt", 999, release=False).endswith("iter_0000999")

    def test_read_metadata_iteration_and_release(self):
        from mindspeed_mm.fsdp.checkpoint.utils import read_metadata

        with tempfile.TemporaryDirectory() as td:
            f1 = os.path.join(td, "latest.txt")
            with open(f1, "w", encoding="utf-8") as f:
                f.write("10")
            it, rel = read_metadata(f1)
            assert it == 10
            assert rel is False

            f2 = os.path.join(td, "latest2.txt")
            with open(f2, "w", encoding="utf-8") as f:
                f.write("release")
            it, rel = read_metadata(f2)
            assert it == 0
            assert rel is True

    def test_read_metadata_invalid_raises(self):
        from mindspeed_mm.fsdp.checkpoint.utils import read_metadata

        with tempfile.TemporaryDirectory() as td:
            fn = os.path.join(td, "latest.txt")
            with open(fn, "w", encoding="utf-8") as f:
                f.write("not-a-number")

            with pytest.raises(ValueError):
                read_metadata(fn)
