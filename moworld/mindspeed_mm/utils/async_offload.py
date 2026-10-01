import torch
import torch_npu
from torch.autograd.graph import saved_tensors_hooks


def base_check_fn(tensor) -> bool:
    """
    Basic check to determine if a tensor is eligible for offloading.
    - Skip Parameters and their views.
    - Skip empty storage tensors.
    """
    if isinstance(tensor._base, torch.nn.parameter.Parameter) or isinstance(tensor, torch.nn.parameter.Parameter):
        return False
    if tensor.storage().size() <= 0:
        return False
    return True


class GetCnt:
    """Tracks tensor count per block for unique key generation and prefetching."""
    def __init__(self):
        self._block_idx = -1
        self._block_tensor_nums = {}  # offload tensors per block {block_id: tensor_idx}

    def get_cnt(self, block_idx):
        after_block = False
        if block_idx > self._block_idx:
            self._block_tensor_nums[block_idx] = 1
            if block_idx != 0:
                after_block = True
            self._block_idx = block_idx
        elif block_idx == self._block_idx:
            self._block_tensor_nums[block_idx] += 1
        else:
            # one step end
            self._block_idx = block_idx
            self._block_tensor_nums = {block_idx: 1}

        offload_tensor_key = "{}_{}".format(self._block_idx, self._block_tensor_nums[self._block_idx] - 1)
        return offload_tensor_key, after_block

    def get_prefetch_keys(self, block_idx, tensor_idx):
        prefetch_block_idx = max((idx for idx in self._block_tensor_nums.keys() if idx < block_idx), default=None)

        if prefetch_block_idx is None:
            return []

        prefetch_block_tensor_nums = self._block_tensor_nums[prefetch_block_idx]
        block_tensor_nums = self._block_tensor_nums[block_idx]
        start = tensor_idx * prefetch_block_tensor_nums // block_tensor_nums
        end = (tensor_idx + 1) * prefetch_block_tensor_nums // block_tensor_nums
        prefetch_idxs = list(range(start, end))
        return ["{}_{}".format(block_idx - 1, prefetch_idx) for prefetch_idx in prefetch_idxs]


class SwapTensor:
    """
    Wrapper to manage device<->host tensor transfers.
    """
    def __init__(self, tensor, key):
        self.tensor = tensor
        self.size = tensor.size()
        self.storage_size = tensor.storage().size()
        self.tensor_cpu = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True, device='cpu')

        self.is_slice_tensor = tensor.storage().size() != tensor.numel()
        self.stat = "device"
        self.key = key

        self.d2h_event = torch_npu.npu.Event()
        self.h2d_event = torch_npu.npu.Event()

    # device to host
    def launch_d2h(self, stream):
        if self.stat != "device":
            return

        forward_event = torch_npu.npu.Event()
        forward_event.record()
        with torch.no_grad():
            with torch_npu.npu.stream(stream):
                stream.wait_event(forward_event)
                if self.is_slice_tensor:
                    self.tensor_cpu.copy_(self.tensor, non_blocking=True)
                else:
                    self.tensor_cpu.storage().copy_(self.tensor.storage(), non_blocking=True)
                self.d2h_event.record()
                self.stat = "host"

    # synchronize d2h and resize 0
    def wait_d2h_finished(self):
        if self.stat != "host":
            return
        torch_npu.npu.current_stream().wait_event(self.d2h_event)
        self.tensor.storage().resize_(0)
        self.stat = "host"

    # resize storage_size and host to device
    def launch_h2d(self, h2d_stream):
        if self.stat != "host":
            return
        backward_event = torch_npu.npu.Event()
        backward_event.record()

        with torch.no_grad():
            with torch_npu.npu.stream(h2d_stream):
                h2d_stream.wait_event(backward_event)
                self.tensor.storage().resize_(self.storage_size)
                if self.is_slice_tensor:
                    self.tensor.copy_(self.tensor_cpu, non_blocking=True)
                else:
                    self.tensor.storage().copy_(self.tensor_cpu.storage(), non_blocking=True)
                self.h2d_event.record()
                self.stat = "device"


class SingletonMeta(type):
    """
    single meta class.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance

        return cls._instances[cls]


class OffloadItem:
    """
    class for offload item
    """

    def __init__(self, act=None, ref_cnt=0, event=None):
        self.act = act
        self.ref_cnt = ref_cnt
        self.event = event


class OffloadManager(metaclass=SingletonMeta):
    """
    Global manager for offloaded tensors with reference counting and prefetch support.
    """

    def __init__(self, check=False):
        self.items = {}
        self.check = check
        self.npu_item = []
        self.getcnt = GetCnt()

    def get_cnt(self, block_idx):
        return self.getcnt.get_cnt(block_idx)

    def assert_exist(self, key):
        if key not in self.items:
            raise RuntimeError(f"Key {key} does not exist in items")

    def exist(self, key):
        return key in self.items

    def assert_not_exist(self, key):
        if key not in self.items:
            raise RuntimeError(f"Key {key} already exist in items")

    # insert or increment reference count for an offloaded tensor.
    def put(self, key, act, event=None):
        if key in self.items:
            self.items[key].act = act
            self.items[key].ref_cnt += 1
            self.items[key].event = event
        else:
            self.items[key] = OffloadItem(act, 1, event)

    def put_npu_tensor(self, act):
        self.npu_item.append(act)

    # Wait for Device-to-Host (D2H) transfer to complete for all tensors whose keys start with the given prefix.
    def del_npu_tensor(self, prefile_key):
        for key in self.items.keys():
            if key.startswith(prefile_key):
                self.items[key].act.wait_d2h_finished()

    # Retrieve tensor and decrement ref count; auto-remove when zero.
    def get(self, key):
        self.assert_exist(key)
        item = self.items[key]

        act = item.act
        if item.event is not None:
            item.get_event().wait()

        item.ref_cnt -= 1
        if item.ref_cnt == 0:
            self.clear(key)
        return act

    # Prefetch tensors needed for the current computation by loading them from host to device (H2D).
    def prefetch_get(self, block_idx, tensor_idx, h2d_stream, d2h_stream):
        prefetch_keys = self.getcnt.get_prefetch_keys(block_idx, tensor_idx)
        for prefetch_key in prefetch_keys:
            if self.exist(prefetch_key):
                prefetch_swap_tensor = self.get(prefetch_key)
                d2h_stream.wait_stream(h2d_stream)
                prefetch_swap_tensor.launch_h2d(h2d_stream)

    def clear(self, key=None):
        if key is None:
            self.items.clear()
        else:
            self.assert_exist(key)
            self.items.pop(key)


class async_save_on_cpu(saved_tensors_hooks):
    """
    A context manager that handles automatic tensor transfers:
    performs device-to-host (D2H) transfer during the forward pass,
    and host-to-device (H2D) transfer during the backward pass.
    """
    def __init__(
            self,
            h2d_stream,
            d2h_stream,
            block_idx,
            depth,
            custom_check_fn=None,
            prefetch=True
        ) -> None:

        def _pack_to_cpu(tensor):
            # skip ineligible tensors
            if not base_check_fn(tensor):
                return tensor

            if (custom_check_fn is not None) and (not custom_check_fn(tensor)):
                return tensor

            key, after_block = OffloadManager().get_cnt(block_idx)

            if after_block:
                OffloadManager().del_npu_tensor("{}_".format(block_idx - 1))

            if block_idx == depth - 1:
                return tensor

            swap_tensor = SwapTensor(tensor, key)

            # Only offload if not in last block (to avoid unnecessary transfer before backward)
            if block_idx < depth - 1:
                swap_tensor.launch_d2h(d2h_stream)

            OffloadManager().put(key, swap_tensor)
            return swap_tensor

        def _unpack_from_cpu(swap_tensor) -> torch.Tensor:
            if isinstance(swap_tensor, torch.Tensor):
                return swap_tensor

            swap_tensor.launch_h2d(h2d_stream)

            # make sure d2h copy is done before into backward
            torch_npu.npu.current_stream().wait_event(swap_tensor.h2d_event)

            if prefetch:
                block_idx, tensor_idx = swap_tensor.key.split("_")
                OffloadManager().prefetch_get(int(block_idx), int(tensor_idx), h2d_stream, d2h_stream)
            return swap_tensor.tensor

        super().__init__(_pack_to_cpu, _unpack_from_cpu)
