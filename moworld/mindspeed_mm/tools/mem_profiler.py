import os
import time
import sys

import torch
import torch_npu
from megatron.core import mpu


def _record(stacks: str = "all", max_entries: int = sys.maxsize):
    torch.npu.memory._record_memory_history(stacks=stacks, max_entries=max_entries)


def _dump(dump_path: str = "./memory_snapshot", dump_ranks=None):
    rank_id = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if dump_ranks is None or rank_id in dump_ranks:
        str_time = time.strftime("%Y-%m-%d-%H-%M")
        file_path = os.path.join(dump_path, f"snapshot_{str_time}_{rank_id}.pickle")
        torch.npu.memory._dump_snapshot(file_path)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if rank_id == 0:
        print(f"memory snapshot dump to {dump_path}")


def _stop():
    torch.npu.memory._record_memory_history(enabled=None)


class MemoryProfiler:
    def __init__(self):
        self.enable = False
        self.start_step = None
        self.end_step = None
        self.save_path = None
        self.dump_ranks = None
        self.stacks = None
        self.max_entries = None
        self.current_step = 0
        self.mem_info = False

    def reset(self, config=None):
        if config is not None:
            self.enable = config.enable
            self.mem_info = config.mem_info
            if self.enable:
                rank_id = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                self.start_step = config.start_step
                self.end_step = config.end_step
                self.save_path = config.save_path
                self.current_step = 0
                self.dump_ranks = config.dump_ranks
                self.stacks = config.stacks
                self.max_entries = sys.maxsize if config.max_entries is None else config.max_entries
                if rank_id == 0 and not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)
                self.step()
        else:
            self.enable = False

    def step(self):
        if self.enable:
            if self.start_step == self.current_step:
                _record(self.stacks, self.max_entries)
            if self.end_step == self.current_step:
                _dump(self.save_path, self.dump_ranks)
                _stop()
                self.enable = False
        if self.mem_info:
            max_memory_reserved = torch.npu.max_memory_reserved()
            max_memory_allocated = torch.npu.max_memory_allocated()
            print(f"\nstep: {self.current_step} \
                    global_rank: {torch.distributed.get_rank()}  \
                    pp_rank: {mpu.get_pipeline_model_parallel_rank()} \
                    tp_rank: {mpu.get_tensor_model_parallel_rank()} \
                    dp_rank: {mpu.get_data_parallel_rank()} \
                    max_memory_reserved: {max_memory_reserved} \
                    max_memory_allocated: {max_memory_allocated}\n",
                  end="",
                  flush=True)
            torch.npu.reset_peak_memory_stats()
        self.current_step += 1

    def stop(self):
        if self.enable:
            if self.start_step is not None and self.current_step > self.start_step:
                _dump(self.save_path, self.dump_ranks)
                _stop()
            self.enable = False


memory_profiler = MemoryProfiler()
