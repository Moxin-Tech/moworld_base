from enum import Enum
from typing import Optional, Tuple, Dict
import torch

from mindspeed_mm.fsdp.utils.device import create_stream, create_event, get_current_stream, switch_to_specified_stream
from mindspeed_mm.fsdp.utils.decorators import Singleton


class DeviceState(Enum):
    HOST = "host"
    DEVICE = "device"


class MultiDeviceTensor:
    def __init__(self, key: str, tensor: torch.Tensor, prefetch_keys: Tuple[str] = tuple()):
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.device = tensor.device

        self.key = str(key)  # Unique identifier
        self.state = DeviceState.DEVICE  # State
        self.storage_size = None
        self.device_tensor = None  # Weak reference to memory
        self.host_tensor = torch.empty(self.shape, dtype=self.dtype, pin_memory=True, device='cpu')
        self.d2h_event = None
        self.h2d_event = None
        self.prefetch_keys = prefetch_keys  # Keys of tensors to prefetch. If not set here, they can also be set in restore_grad. prefetch_keys set in restore_grad have higher priority.

    def set_grad_tensor(self, tensor: torch.Tensor):
        self.device_tensor = tensor
        self.storage_size = tensor.storage().size()

    def get_state(self) -> DeviceState:
        return self.state

    def get_d2h_event(self) -> Optional[torch.cuda.Event]:
        return self.d2h_event

    def get_h2d_event(self) -> Optional[torch.cuda.Event]:
        return self.h2d_event

    def launch_d2h(self, d2h_stream: torch.cuda.Stream):
        """Initiate asynchronous transfer from device to host."""
        if self.state != DeviceState.DEVICE:
            return

        device_tensor = self.device_tensor

        # Record main compute stream event
        main_stream_event = create_event()
        main_stream_event.record()

        self.d2h_event = create_event()

        with switch_to_specified_stream(d2h_stream):
            # Adding an event here prevents the copy from starting too early, which could overlap with main compute stream memory and cause excessive memory usage.
            d2h_stream.wait_event(main_stream_event)
            if device_tensor.is_contiguous():
                # Underlying storage copy, faster for contiguous memory scenarios
                self.host_tensor.storage().copy_(device_tensor.storage(), non_blocking=True)
            else:
                self.host_tensor.copy_(device_tensor, non_blocking=True)
            self.d2h_event.record(d2h_stream)
        get_current_stream().wait_event(self.d2h_event)
        device_tensor.untyped_storage().resize_(0)

        self.state = DeviceState.HOST

    def launch_h2d(self, h2d_stream: torch.cuda.Stream):
        """Initiate asynchronous transfer from host to device."""
        if self.state != DeviceState.HOST:
            return

        device_tensor = self.device_tensor
        device_tensor.storage().resize_(self.storage_size)

        # Record main compute stream event
        main_compute_event = create_event()
        main_compute_event.record()

        # Create H2D completion event
        self.h2d_event = create_event()

        with switch_to_specified_stream(h2d_stream):
            h2d_stream.wait_event(main_compute_event)
            if device_tensor.is_contiguous():
                # Underlying storage copy, faster for contiguous memory scenarios
                self.device_tensor.storage().copy_(self.host_tensor.storage(), non_blocking=True)
            else:
                self.device_tensor.copy_(self.host_tensor, non_blocking=True)
            self.h2d_event.record(h2d_stream)

        self.state = DeviceState.DEVICE


class GradOffloadManager(metaclass=Singleton):
    def __init__(self):
        self.gradient_storage: Dict[str, MultiDeviceTensor] = {}
        self.d2h_stream: Optional[torch.cuda.Stream] = create_stream()
        self.h2d_stream: Optional[torch.cuda.Stream] = self.d2h_stream

    def register_grad(self, key: str, input_tensor: torch.Tensor, prefetch_keys: Tuple[str] = tuple()) -> str:
        self.gradient_storage[key] = MultiDeviceTensor(key, input_tensor, prefetch_keys)

    def record_grad(self, key: str, grad_tensor: torch.Tensor):
        if key not in self.gradient_storage:
            return
        self.gradient_storage[key].set_grad_tensor(grad_tensor)

    def offload_grad(self, key: str):
        if key not in self.gradient_storage:
            return

        self.gradient_storage[key].launch_d2h(self.d2h_stream)

    def restore_grad(self, key: str, prefetch_keys: Tuple[str] = None) -> Optional[torch.Tensor]:
        if key not in self.gradient_storage:
            return None

        multi_tensor = self.gradient_storage[key]

        # Asynchronously initiate H2D transfer
        multi_tensor.launch_h2d(self.h2d_stream)

        tensor_prefetch_keys = tuple()
        if prefetch_keys is not None:
            tensor_prefetch_keys = prefetch_keys
        elif multi_tensor.prefetch_keys is not None:
            tensor_prefetch_keys = multi_tensor.prefetch_keys

        for prefetch_key in tensor_prefetch_keys:
            if prefetch_key not in self.gradient_storage:
                continue
            self.gradient_storage[prefetch_key].launch_h2d(self.h2d_stream)
        get_current_stream().wait_event(multi_tensor.get_h2d_event())

        # Return the device tensor
        return multi_tensor.device_tensor

    def clear(self):
        self.gradient_storage.clear()


grad_offload_manager = GradOffloadManager()


class GradientOffload(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, key: Optional[str] = None, prefetch_keys: Tuple[str] = None):
        ctx.key = key
        ctx.prefetch_keys = prefetch_keys
        ctx.x_device = x.device
        ctx.x_dtype = x.dtype
        ctx.x_shape = x.shape
        grad_offload_manager.register_grad(key, x, prefetch_keys)

        return x

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.key is not None:
            grad_offload_manager.record_grad(ctx.key, grad_output)
            grad_offload_manager.offload_grad(ctx.key)

        # Create a broadcasted view using expand_as. Views created by expand do not allocate new memory.
        return torch.tensor(0.0, device=ctx.x_device, dtype=ctx.x_dtype).expand(ctx.x_shape), None, None


class GradientRestore(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, key: Optional[str] = None, prefetch_keys: Tuple[str] = None):
        """Restore gradient from storage
        prefetch_keys set here have higher priority than those set at the GradientOffload location.
        """

        ctx.key = key
        ctx.prefetch_keys = prefetch_keys
        return x

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.key is not None:
            restored_grad = grad_offload_manager.restore_grad(ctx.key, ctx.prefetch_keys)
            return restored_grad, None, None

        return grad_output, None, None


def clear_offload_grad():
    grad_offload_manager.clear()
