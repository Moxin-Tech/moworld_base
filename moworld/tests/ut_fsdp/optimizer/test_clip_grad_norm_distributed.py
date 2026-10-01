# pylint: skip-file
import os
import tempfile

import pytest


def _reset_parallel_state_for_worker():
    """Reset global/singleton state inside spawned worker process.

    pytest-xdist/torch.multiprocessing may execute multiple tests per worker.
    Resetting avoids leaking a previously-created device mesh/parallelstate
    """

    import mindspeed_mm.fsdp.distributed.parallel_state as ps_mod

    # force ut device mesh in UT
    ps_mod.get_device_type = lambda: "npu"

    ps_mod._PARALLEL_STATE = None
    from mindspeed_mm.fsdp.utils.decorators import Singleton

    Singleton._instances = {}


def _init_pg(rank: int, world_size: int, init_file: str):
    import torch.distributed as dist
    import torch

    # bind each process to a dedicated npu
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


def _worker_l2_clip(rank: int, world_size: int, init_file: str):
    pytest.importorskip("torch")
    import torch
    import torch.distributed as dist

    try:
        _init_pg(rank, world_size, init_file)
        _reset_parallel_state_for_worker()

        import mindspeed_mm.fsdp.distributed.parallel_state as ps_mod

        ps_mod.init_parallel_state(
            data_parallel_size=world_size,
            fully_shard_parallel_size=1,
            tensor_parallel_size=1,
            ring_attention_size=1,
            ulysses_parallel_size=1,
            expert_parallel_size=1,
            expert_fully_shard_parallel_size=1
        )

        import mindspeed_mm.fsdp.optimizer.clip_grad_norm as mod

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.zeros(1))
                self.b = torch.nn.Parameter(torch.zeros(1))

        m = M()
        m = m.to(torch.device("npu", rank))
        #make ranks contribute different local norms
        if rank == 0:
            m.w.grad = torch.tensor([3.0], device=torch.device("npu", rank))
            m.b.grad = torch.tensor([4.0], device=torch.device("npu", rank))
        else:
            m.w.grad = torch.tensor([0.0], device=torch.device("npu", rank))
            m.b.grad = torch.tensor([0.0], device=torch.device("npu", rank))

        # global l2 norm: sqrt(3^2 + 4^2) == 5
        returned = mod.clip_grad_norm(m, max_norm=1.0, norm_type=2.0)
        expected = torch.tensor(5.0, device=torch.device("npu", rank))

        gathered = [torch.zeros_like(returned) for _ in range(world_size)]
        dist.all_gather(gathered, returned)
        assert all(torch.allclose(x, expected, atol=1e-6) for x in gathered)

        # verify gradients are clipped with a single global coefficient on every rank
        # coef = 1 / 5
        if rank == 0:
            assert torch.allclose(m.w.grad, torch.tensor([0.6], device=torch.device("npu", rank)), atol=1e-6)
            assert torch.allclose(m.b.grad, torch.tensor([0.8], device=torch.device("npu", rank)), atol=1e-6)
        else:
            assert torch.allclose(m.w.grad, torch.tensor([0.0], device=torch.device("npu", rank)), atol=1e-6)
            assert torch.allclose(m.b.grad, torch.tensor([0.0], device=torch.device("npu", rank)), atol=1e-6)
    finally:
        _destroy_pg()


def _worker_inf_norm(rank: int, world_size: int, init_file: str):
    pytest.importorskip("torch")
    import torch
    import torch.distributed as dist

    try:
        _init_pg(rank, world_size, init_file)
        _reset_parallel_state_for_worker()

        import mindspeed_mm.fsdp.distributed.parallel_state as ps_mod

        ps_mod.init_parallel_state(
            data_parallel_size=world_size,
            fully_shard_parallel_size=1,
            tensor_parallel_size=1,
            ring_attention_size=1,
            ulysses_parallel_size=1,
            expert_parallel_size=1,
            expert_fully_shard_parallel_size=1
        )

        import mindspeed_mm.fsdp.optimizer.clip_grad_norm as mod

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.zeros(4))

        m = M()
        m = m.to(torch.device("npu", rank))
        if rank == 0:
            m.p.grad = torch.full((4,), 2.0, device=torch.device("npu", rank))
        else:
            m.p.grad = torch.full((4,), 5.0, device=torch.device("npu", rank))

        before = m.p.grad.clone()
        returned = mod.clip_grad_norm(m, max_norm=0.0, norm_type=float("inf"))

        gathered = [torch.zeros_like(returned) for _ in range(world_size)]
        dist.all_gather(gathered, returned)
        assert all(torch.allclose(x, torch.tensor(5.0, device=torch.device("npu", rank)), atol=1e-6) for x in gathered)
        # max_norm = 0 means compute-only, grads must remain unchanged
        assert torch.allclose(m.p.grad, before)
    finally:
        _destroy_pg()


def _worker_ep_path(rank: int, world_size: int, init_file: str):
    pytest.importorskip("torch")
    import torch
    import torch.distributed as dist

    try:
        _init_pg(rank, world_size, init_file)
        _reset_parallel_state_for_worker()

        import mindspeed_mm.fsdp.distributed.parallel_state as ps_mod

        ps = ps_mod.init_parallel_state(
            data_parallel_size=world_size,
            fully_shard_parallel_size=1,
            tensor_parallel_size=1,
            ring_attention_size=1,
            ulysses_parallel_size=1,
            expert_parallel_size=world_size,
            expert_fully_shard_parallel_size=1,
        )
        assert ps.is_ep_enable() is True

        import mindspeed_mm.fsdp.optimizer.clip_grad_norm as mod

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ep = torch.nn.Parameter(torch.zeros(1))
                self.non_ep = torch.nn.Parameter(torch.zeros(1))
                # Trigger EP-aware code path
                self._ep_param_groups = {"ep": {self.ep}, "non_ep": {self.non_ep}}

        m = M()
        m = m.to(torch.device("npu", rank))
        if rank == 0:
            m.ep.grad = torch.tensor([3.0], device=torch.device("npu", rank))
            m.non_ep.grad = torch.tensor([4.0], device=torch.device("npu", rank))
        else:
            m.ep.grad = torch.tensor([0.0], device=torch.device("npu", rank))
            m.non_ep.grad = torch.tensor([0.0], device=torch.device("npu", rank))

        returned = mod.clip_grad_norm(m, max_norm=1.0, norm_type=2.0)
        gathered = [torch.zeros_like(returned) for _ in range(world_size)]
        dist.all_gather(gathered, returned)
        assert all(torch.allclose(x, torch.tensor(5.0, device=torch.device("npu", rank)), atol=1e-6) for x in gathered)

        if rank == 0:
            assert torch.allclose(m.ep.grad, torch.tensor([0.6], device=torch.device("npu", rank)), atol=1e-6)
            assert torch.allclose(m.non_ep.grad, torch.tensor([0.8], device=torch.device("npu", rank)), atol=1e-6)
        else:
            assert torch.allclose(m.ep.grad, torch.tensor([0.0], device=torch.device("npu", rank)), atol=1e-6)
            assert torch.allclose(m.non_ep.grad, torch.tensor([0.0], device=torch.device("npu", rank)), atol=1e-6)
    finally:
        _destroy_pg()


@pytest.mark.parametrize(
    "worker",
    [_worker_l2_clip, _worker_inf_norm, _worker_ep_path],
)
def test_clip_grad_norm_multi_process(worker):
    """ verify real gloo all_reduce paths across 2 ranks

    This test convers:
    - non-EP path (FSDP group SUM reduction)
    - inf-norm path (MAX reduction)
    - EP-aware path (efsdp then ep reductions; shared clip coefficient)
    """

    pytest.importorskip("torch")
    import torch
    import torch.multiprocessing as mp

    if not hasattr(torch, "npu") or torch.npu.device_count() < 2:
        pytest.skip("需要至少2卡NPU才能运行该用例")

    world_size = 2
    with tempfile.NamedTemporaryFile(delete=False) as f:
        init_file = f.name
    try:
        mp.spawn(worker, args=(world_size, init_file), nprocs=world_size, join=True)
    finally:
        try:
            os.remove(init_file)
        except OSError:
            pass
