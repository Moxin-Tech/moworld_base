import os
import pytest
import tempfile
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
from mindspeed_mm.fsdp.distributed.parallel_state import init_parallel_state
from tests.ut_fsdp.utils.utils import judge_expression


def _init_pg(rank: int, world_size: int, init_file: str):
    if hasattr(torch, "npu"):
        torch.npu.set_device(rank)

    dist.init_process_group(backend="hccl", init_method=f"file://{init_file}", rank=rank, world_size=world_size)


def _destroy_pg():
    if dist.is_initialized():
        dist.destroy_process_group()


class TestFlashAttention:
    @staticmethod
    def _test_flash_attention_forward(module, q, k, v, attn_mask, common_kwargs):
        from mindspeed_mm.fsdp.ops.flash_attn.flash_attn_refactor import flash_attention_forward

        with torch.no_grad():
            output, _ = flash_attention_forward(
                module=module, query=q, key=k, value=v, attention_mask=attn_mask, **common_kwargs
            )
        return output

    def _test_npu_flash_attention(self):
        dtype = torch.bfloat16
        device = "npu"
        torch.manual_seed(42)

        # B S N D, cu_seq_lens, is_causal
        TEST_CASES = [
            (1, 4096, 32, 128, None, True),
            (1, 4096, 32, 128, [0, 1024, 2048, 4096], True),
            (1, 4096, 32, 128, None, False),
            (1, 4096, 32, 128, [0, 1024, 2048, 4096], False),
        ]

        for batch_size, seq_len, num_heads, head_dim, cu_seq_lens, is_causal in TEST_CASES:

            shape = (batch_size, seq_len, num_heads, head_dim)

            q = torch.randn(shape, device=device, dtype=dtype).requires_grad_(False)
            k = torch.randn(shape, device=device, dtype=dtype).requires_grad_(False)
            v = torch.randn(shape, device=device, dtype=dtype).requires_grad_(False)

            is_packing = cu_seq_lens is not None
            attention_mask = (
                torch.triu(torch.ones([2048, 2048], device=device), diagonal=1).bool() if is_causal else None
            )
            target_output = torch_npu.npu_fusion_attention(
                q.squeeze(0) if is_packing else q,
                k.squeeze(0) if is_packing else k,
                v.squeeze(0) if is_packing else v,
                num_heads,
                "TND" if is_packing else "BSND",
                padding_mask=None,
                atten_mask=attention_mask,
                actual_seq_qlen=cu_seq_lens,
                actual_seq_kvlen=cu_seq_lens,
                scale=1.0 / (head_dim**0.5),
                keep_prob=1,
                inner_precise=0,
                sparse_mode=3 if is_causal else 0,
            )[0]

            if is_packing:
                target_output = target_output.unsqueeze(0)

            class DummyModule:
                def __init__(self):
                    self.config = type(
                        "obj",
                        (object,),
                        {
                            "_attn_implementation": "flash_attention_2",
                        },
                    )()
                    self.is_causal = is_causal
                    self.layer_idx = 0

            module = DummyModule()
            common_kwargs = {
                "dropout": 0.0,
                "scaling": 1.0 / (head_dim**0.5),
                "cu_seq_lens_q": cu_seq_lens,
                "cu_seq_lens_k": cu_seq_lens,
                "input_layout": "1TND" if is_packing else "BSND",
                "total_seq_len": seq_len,
            }

            output = self._test_flash_attention_forward(
                module=module, q=q, k=k, v=v, attn_mask=None, common_kwargs=common_kwargs
            )

            judge_expression(torch.allclose(target_output, output, rtol=1e-4, atol=1e-5))

            # BNSD or 1NTD case
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()

            common_kwargs["input_layout"] = "1NTD" if is_packing else "BNSD"
            output = self._test_flash_attention_forward(
                module=module, q=q, k=k, v=v, attn_mask=None, common_kwargs=common_kwargs
            )

            judge_expression(torch.allclose(target_output, output, rtol=1e-4, atol=1e-5))

    def _test_cp(self, ulysses_parallel_size=1, ring_parallel_size=1):
        from mindspeed_mm.fsdp.distributed.context_parallel.communication import (
            split_forward_gather_backward_with_cp,
            packed_data_split_forward_gather_backward_with_cp,
        )
        from mindspeed_mm.fsdp.distributed.context_parallel.utils import cal_split_sizes_multi

        dtype = torch.bfloat16
        device = "npu"
        torch.manual_seed(42)

        cp_size = ulysses_parallel_size * ring_parallel_size
        init_parallel_state(
            fully_shard_parallel_size=cp_size,
            ulysses_parallel_size=ulysses_parallel_size,
            ring_attention_size=ring_parallel_size,
        )

        # B S N D
        if ring_parallel_size > 1:
            TEST_CASES = [
                (1, 4096, 32, 128, None, True),
                (1, 4096, 32, 128, None, False),
                (1, 4096, 32, 128, [0, 1024, 2048, 4096], False),
            ]
        else:
            TEST_CASES = [
                (1, 4096, 32, 128, None, True),
                (1, 4096, 32, 128, [0, 1024, 2048, 4096], True),
                (1, 4096, 32, 128, None, False),
                (1, 4096, 32, 128, [0, 1024, 2048, 4096], False),
            ]

        for batch_size, seq_len, num_heads, head_dim, cu_seq_lens, is_causal in TEST_CASES:

            shape = (batch_size, seq_len, num_heads, head_dim)

            q = torch.randn(shape, device=device, dtype=dtype).requires_grad_(False)
            k = torch.randn(shape, device=device, dtype=dtype).requires_grad_(False)
            v = torch.randn(shape, device=device, dtype=dtype).requires_grad_(False)

            # broadcast, 确保cp域内的结果一致
            dist.broadcast(q, src=0)
            dist.broadcast(k, src=0)
            dist.broadcast(v, src=0)

            is_packing = cu_seq_lens is not None
            attention_mask = (
                torch.triu(torch.ones([2048, 2048], device=device), diagonal=1).bool() if is_causal else None
            )
            target_output = torch_npu.npu_fusion_attention(
                q.squeeze(0) if is_packing else q,
                k.squeeze(0) if is_packing else k,
                v.squeeze(0) if is_packing else v,
                num_heads,
                "TND" if is_packing else "BSND",
                padding_mask=None,
                atten_mask=attention_mask,
                actual_seq_qlen=cu_seq_lens,
                actual_seq_kvlen=cu_seq_lens,
                scale=1.0 / (head_dim**0.5),
                keep_prob=1,
                inner_precise=0,
                sparse_mode=3 if is_causal else 0,
            )[0]

            if is_packing:
                target_output = target_output.unsqueeze(0)
                split_seq_lens = [
                    cu_seq_len2 - cu_seq_len1 for cu_seq_len1, cu_seq_len2 in zip(cu_seq_lens[:-1], cu_seq_lens[1:])
                ]
                local_q = packed_data_split_forward_gather_backward_with_cp(q, dim=1, seq_lens=split_seq_lens)
                local_k = packed_data_split_forward_gather_backward_with_cp(k, dim=1, seq_lens=split_seq_lens)
                local_v = packed_data_split_forward_gather_backward_with_cp(v, dim=1, seq_lens=split_seq_lens)
            else:
                local_q = split_forward_gather_backward_with_cp(q, dim=1)
                local_k = split_forward_gather_backward_with_cp(k, dim=1)
                local_v = split_forward_gather_backward_with_cp(v, dim=1)

            class DummyModule:
                def __init__(self):
                    self.config = type(
                        "obj",
                        (object,),
                        {
                            "_attn_implementation": "flash_attention_2",
                        },
                    )()
                    self.is_causal = is_causal
                    self.layer_idx = 0

            module = DummyModule()
            common_kwargs = {
                "dropout": 0.0,
                "scaling": 1.0 / (head_dim**0.5),
                "cu_seq_lens_q": cu_seq_lens,
                "cu_seq_lens_k": cu_seq_lens,
                "input_layout": "1TND" if is_packing else "BSND",
                "total_seq_len": seq_len,
            }

            if is_packing:
                common_kwargs["seq_split_lens"] = cal_split_sizes_multi(split_seq_lens, ring_parallel_size)

            local_output = self._test_flash_attention_forward(
                module=module, q=local_q, k=local_k, v=local_v, attn_mask=None, common_kwargs=common_kwargs
            )

            if is_packing:
                local_target_output = packed_data_split_forward_gather_backward_with_cp(
                    target_output, dim=1, seq_lens=split_seq_lens
                )
            else:
                local_target_output = split_forward_gather_backward_with_cp(target_output, dim=1)

            judge_expression(torch.allclose(local_target_output, local_output, rtol=1e-2, atol=1e-3))

            # 验证ring_in_bnsd
            if not is_packing and ring_parallel_size > 1:
                common_kwargs["ring_in_bnsd"] = True
                local_output = self._test_flash_attention_forward(
                    module=module, q=local_q, k=local_k, v=local_v, attn_mask=None, common_kwargs=common_kwargs
                )

                judge_expression(torch.allclose(local_target_output, local_output, rtol=1e-2, atol=1e-3))
                common_kwargs.pop("ring_in_bnsd")

            # 1NTD和BNSD输入
            local_q = local_q.transpose(1, 2).contiguous()
            local_k = local_k.transpose(1, 2).contiguous()
            local_v = local_v.transpose(1, 2).contiguous()

            common_kwargs["input_layout"] = "1NTD" if is_packing else "BNSD"
            local_output = self._test_flash_attention_forward(
                module=module, q=local_q, k=local_k, v=local_v, attn_mask=None, common_kwargs=common_kwargs
            )

            judge_expression(torch.allclose(local_target_output, local_output, rtol=1e-2, atol=1e-3))

            # 验证ring_in_bnsd
            if not is_packing and ring_parallel_size > 1:
                common_kwargs["ring_in_bnsd"] = True
                local_output = self._test_flash_attention_forward(
                    module=module, q=local_q, k=local_k, v=local_v, attn_mask=None, common_kwargs=common_kwargs
                )

                judge_expression(torch.allclose(local_target_output, local_output, rtol=1e-2, atol=1e-3))
                common_kwargs.pop("ring_in_bnsd")

    def _init_env_and_test(self, rank, world_size, init_file, test_func, test_kwargs=None):
        _init_pg(rank, world_size, init_file)
        test_func(**test_kwargs)
        _destroy_pg()

    def test_npu_flash_attention(self):
        world_size = 1
        with tempfile.NamedTemporaryFile(delete=False) as f:
            init_file = f.name

        try:
            mp.spawn(
                self._init_env_and_test,
                args=(world_size, init_file, self._test_npu_flash_attention, {}),
                nprocs=world_size,
                join=True,
            )
        finally:
            if os.path.exists(init_file):
                os.remove(init_file)

    @pytest.mark.skipif(torch.npu.device_count() < 2, reason="Requires at least 2 devices to test ulysses cp")
    def test_ulysses_cp(self):
        world_size = 2
        with tempfile.NamedTemporaryFile(delete=False) as f:
            init_file = f.name

        try:
            mp.spawn(
                self._init_env_and_test,
                args=(world_size, init_file, self._test_cp, {"ulysses_parallel_size": 2}),
                nprocs=world_size,
                join=True,
            )

        finally:
            if os.path.exists(init_file):
                os.remove(init_file)

    @pytest.mark.skipif(torch.npu.device_count() < 2, reason="Requires at least 2 devices to test ring cp")
    def test_ring_cp(self):
        world_size = 2
        with tempfile.NamedTemporaryFile(delete=False) as f:
            init_file = f.name

        try:
            mp.spawn(
                self._init_env_and_test,
                args=(world_size, init_file, self._test_cp, {"ring_parallel_size": 2}),
                nprocs=world_size,
                join=True,
            )

        finally:
            if os.path.exists(init_file):
                os.remove(init_file)

    @pytest.mark.skipif(torch.npu.device_count() < 4, reason="Requires at least 2 devices to test hybrid cp")
    def test_hybrid_cp(self):
        world_size = 4
        with tempfile.NamedTemporaryFile(delete=False) as f:
            init_file = f.name

        try:
            mp.spawn(
                self._init_env_and_test,
                args=(world_size, init_file, self._test_cp, {"ring_parallel_size": 2, "ulysses_parallel_size": 2}),
                nprocs=world_size,
                join=True,
            )

        finally:
            if os.path.exists(init_file):
                os.remove(init_file)
