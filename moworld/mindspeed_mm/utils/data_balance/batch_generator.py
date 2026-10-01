import threading
import torch


def move_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {k: move_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_cpu(v) for v in obj)
    else:
        return obj


class PrefetchMicroBatchIterator:
    def __init__(self, mbs):
        self.mbs = mbs
        self.n = len(mbs)
        self.idx = 0
        self.cpu_cache = [None] * self.n
        self.cpu_cache[0] = mbs[0]

        if self.n > 1:
            self.prefetch_thread = threading.Thread(target=self._prefetch_to_cpu, daemon=True)
            self.prefetch_thread.start()
        else:
            self.prefetch_thread = None

    def _prefetch_to_cpu(self):
        if torch.npu.is_available():
            torch.npu.set_device(torch.npu.current_device())
        for i in range(1, self.n):
            self.cpu_cache[i] = move_to_cpu(self.mbs[i])

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.n:
            self.close()
            raise StopIteration

        if self.idx == 0:
            data = self.cpu_cache[0]
        else:
            if self.idx == 1 and self.prefetch_thread is not None and self.prefetch_thread.is_alive():
                self.prefetch_thread.join()
            data = self.cpu_cache[self.idx]

        self.idx += 1
        return data

    def close(self):
        if self.prefetch_thread is not None and self.prefetch_thread.is_alive():
            self.prefetch_thread.join()
        self.cpu_cache = None
        self.mbs = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
