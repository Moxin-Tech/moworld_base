# pylint: skip-file
import os
import tempfile

import pytest


def _init_pg(rank: int, world_size: int, init_file: str):
    import torch.distributed as dist
    import torch

    if hasattr(torch, "npu"):
        torch.npu.set_device(rank)

    dist.init_process_group(
        backend="hccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size
    )


def _destroy_pg():
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def _worker(rank: int, world_size: int, init_file: str):
    pytest.importorskip("torch")
    import torch
    import torch.distributed as dist

    try:
        _init_pg(rank, world_size, init_file)

        import mindspeed_mm.fsdp.distributed.parallel_state as ps_mod

        # Force NPU device mesh in UT
        ps_mod.get_device_type = lambda: "npu" # type: ignore[assignment]

        # Reset singleton/global state inside spawned worker
        ps_mod._PARALLEL_STATE = None
        from mindspeed_mm.fsdp.utils.decorators import Singleton

        Singleton._instances = {}

        ps = ps_mod.init_parallel_state(
            data_parallel_size=world_size,
            fully_shard_parallel_size=1,
            tensor_parallel_size=1,
            ring_attention_size=1,
            ulysses_parallel_size=1,
            expert_parallel_size=world_size,
            expert_fully_shard_parallel_size=1,
        )

        # device mesh dynamic helpers should exist and match the real world topology
        assert ps.get_dp_group_size() == world_size
        assert ps.get_dp_rank() == rank
        assert ps.get_fsdp_group_size() == world_size
        assert ps.get_fsdp_rank() == rank

        # flattened/disabled dimensions should be singletons in this config
        assert ps.get_cp_group_size() == 1
        assert ps.is_cp_enable() is False
        assert ps.get_tp_group_size() == 1
        assert ps.is_tp_enable() is False

        # ep is enabled when expert_parallel_size == world_size
        assert ps.is_ep_enable() is True
        assert ps.get_ep_group_size() == world_size

        # get_fsdp_device_mesh should return a valid submesh
        fsdp_mesh = ps.get_fsdp_device_mesh()
        assert hasattr(fsdp_mesh, "mesh")
        # submesh total elements should match world size
        assert int(torch.numel(fsdp_mesh.mesh)) == world_size

        # singleton semantics: constructing again yields the same instance
        ps2 = ps_mod.ParallelState(
            data_parallel_size=world_size,
            fully_shard_parallel_size=1,
            tensor_parallel_size=1,
            ring_attention_size=1,
            ulysses_parallel_size=1,
            expert_parallel_size=world_size,
            expert_fully_shard_parallel_size=1,
        )
        assert ps2 is ps

        # sanity: groups exist and are usable
        assert dist.get_world_size(ps.get_fsdp_group()) == world_size
        dist.barrier(ps.get_fsdp_group())
    finally:
        _destroy_pg()


def test_parallel_state_multi_rank():
    """Hard UT: validate real device mesh group sizes/ranks with gloo multi-proc"""

    pytest.importorskip("torch")
    import torch
    import torch.multiprocessing as mp

    if not hasattr(torch, "npu") or torch.npu.device_count() < 2:
        pytest.skip("需要至少2张卡才能运行该分布式用例")

    world_size = 2
    with tempfile.NamedTemporaryFile(delete=False) as f:
        init_file = f.name
    try:
        mp.spawn(_worker, args=(world_size, init_file), nprocs=world_size, join=True)
    finally:
        try:
            os.remove(init_file)
        except OSError:
            pass
