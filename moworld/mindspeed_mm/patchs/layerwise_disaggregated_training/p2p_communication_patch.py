# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from typing import List, Optional, Tuple, Union

import torch

from megatron import core
from megatron.core import ModelParallelConfig
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_next_rank,
    get_pipeline_model_parallel_prev_rank,
)
from megatron.core.pipeline_parallel.p2p_communication import (
    _batched_p2p_ops,
    _p2p_ops,
)

from mindspeed_mm.patchs.layerwise_disaggregated_training.parallel_state_patch import (
    is_vtp_stage_rank0,
    get_vtp_size_list,
    get_vtp_stage_ranks,
    get_vtp_my_stage_idx,
)

# Types
Shape = Union[List[int], torch.Size]


def _communicate_impl(
    *,
    tensor_send_next: Optional[torch.Tensor],
    tensor_send_prev: Optional[torch.Tensor],
    recv_prev: bool,
    recv_next: bool,
    tensor_shape: Shape,
    config: ModelParallelConfig,
    wait_on_reqs: bool = True,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Communicate tensors between stages. Used as helper method in other
    communication methods that are used in megatron/schedules.py.

    Args:
        tensor_send_next (torch.Tensor, optional):
            Tensor to send to next rank (no tensor sent if None)

        tensor_send_prev (torch.Tensor, optional):
            Tensor to send to prev rank (no tensor sent if None)

        recv_prev (boolean, required):
            whether tensor should be received from previous rank.

        recv_next (boolean, required):
            whether tensor should be received from next rank.

        tensor_shape (List[int] or torch.Size, required):
            shape of tensor to receive (this method assumes that all
            tensors sent and received in a single function call are
            the same shape).

        wait_on_reqs (boolean, optional, default=False):
            For non-batched p2p communication, wait on each request
            before returning.

    Returns:
        tuple containing

        - tensor_recv_prev: torch.Tensor if recv_prev is True, None otherwise.
        - tensor_recv_next: torch.Tensor if recv_next is True, None otherwise.

    """

    tensor_recv_prev_func = None
    tensor_recv_next_func = None

    recv_prev_shape = tensor_shape
    recv_next_shape = tensor_shape

    def create_tensor_recv_prev():
        return torch.empty(
            recv_prev_shape,
            requires_grad=True,
            device=torch.cuda.current_device(),
            dtype=config.pipeline_dtype,
        )

    def create_tensor_recv_next():
        return torch.empty(
            recv_next_shape,
            requires_grad=True,
            device=torch.cuda.current_device(),
            dtype=config.pipeline_dtype,
        )

    if recv_prev:
        if config.pipeline_dtype is None:
            raise RuntimeError("pipeline_dtype must be provided if recv_prev is True")
        if tensor_shape is None:
            raise RuntimeError(
                "tensor_shape must be specified if recv_prev is True. "
                "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
            )
        tensor_recv_prev_func = create_tensor_recv_prev

    if recv_next:
        if config.pipeline_dtype is None:
            raise RuntimeError("dtype must be provided if recv_next is True")
        if tensor_shape is None:
            raise RuntimeError(
                "tensor_shape must be specified if recv_next is True. "
                "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
            )
        tensor_recv_next_func = create_tensor_recv_next

    # Send tensors in both the forward and backward directions as appropriate.
    if config.use_ring_exchange_p2p:

        def _ring_exchange_wrapper(**kwargs):
            torch.distributed.ring_exchange(**kwargs)
            return []

        p2p_func = _ring_exchange_wrapper
    elif config.batch_p2p_comm:
        if not wait_on_reqs:
            raise ValueError()
        p2p_func = _batched_p2p_ops
    else:
        p2p_func = _p2p_ops

    # Each rank can now be part of several different pipeline parallel groups
    # (specifically, this can occur when encoder tensor parallelism != decoder
    # tensor parallelism, and hence a rank in the encoder is going to feed
    # several different decoder ranks. We therefore have to receive or send tensors
    # from several groups. For convenience, I wrap everything into lists.
    if "group" in kwargs.keys():
        if kwargs["group"] is not None:
            pp_group = kwargs["group"]
        else:
            pp_group = get_pipeline_model_parallel_group()
    else:
        pp_group = get_pipeline_model_parallel_group()

    next_rank = kwargs.get("next_rank", None)
    prev_rank = kwargs.get("prev_rank", None)
    if next_rank is None:
        next_rank = get_pipeline_model_parallel_next_rank()
    if prev_rank is None:
        prev_rank = get_pipeline_model_parallel_prev_rank()

    if not isinstance(pp_group, list):
        pp_group = [pp_group]
    if not isinstance(next_rank, list):
        next_rank = [next_rank]
    if not isinstance(prev_rank, list):
        prev_rank = [prev_rank]

    if config.use_ring_exchange_p2p or config.batch_p2p_comm:
        reqs = []
    else:
        reqs = {}
    tensor_recv_prev_list = []
    tensor_recv_next_list = []

    for group, nr, pr in zip(pp_group, next_rank, prev_rank):
        if tensor_recv_prev_func is not None:
            tensor_recv_prev = tensor_recv_prev_func()
            tensor_recv_prev_list.append(tensor_recv_prev)
        else:
            tensor_recv_prev = None

        if tensor_recv_next_func is not None:
            tensor_recv_next = tensor_recv_next_func()
            tensor_recv_next_list.append(tensor_recv_next)
        else:
            tensor_recv_next = None

        p2p_reqs = p2p_func(
            tensor_send_prev=tensor_send_prev,
            tensor_recv_prev=tensor_recv_prev,
            tensor_send_next=tensor_send_next,
            tensor_recv_next=tensor_recv_next,
            group=group,
            prev_pipeline_rank=pr,
            next_pipeline_rank=nr,
        )
        if isinstance(p2p_reqs, list):
            reqs.extend(p2p_reqs)
        else:
            reqs.update(p2p_reqs)

    if wait_on_reqs and len(reqs) > 0:
        for req in reqs if isinstance(reqs, list) else reqs.values():
            req.wait()
        reqs = None

    # Synchronization condition check
    need_sync = (
        (config.batch_p2p_comm and config.batch_p2p_sync)
        or len(tensor_recv_prev_list) > 1
        or len(tensor_recv_next_list) > 1
    )
    if need_sync:
        # To protect against race condition when using batch_isend_irecv().
        # User should assert that we have a modern enough PyTorch to not need this
        torch.cuda.synchronize()

    def _handle_tensor_list(x):
        """This basically handles all the cases that we expect to see. Either the list None,
        or it's a singleton (the usual cases, since most ranks only belong to one pipeline group),
        or everything returned is None, or everything returned is not None, and it has to be summed
        together."""
        if len(x) == 0:
            return None
        if len(x) == 1:
            return x[0]
        if all(xx is None for xx in x):
            return None
        # When the encoder's TP size differs from the decoder's TP size
        # (with the constraint `encoder_tp_size <= decoder_tp_size`), each encoder TP rank
        # may receive multiple gradients from corresponding decoder TP ranks.
        # For example, if `ETP=1` and `DTP=2`, then encoder rank 0 will receive gradients
        # from decoder ranks 1 and 2. These received gradients must be averaged.
        return torch.stack(x, dim=0).mean(dim=0, dtype=torch.float32).to(x[0].dtype)

    tensor_recv_prev = _handle_tensor_list(tensor_recv_prev_list)
    tensor_recv_next = _handle_tensor_list(tensor_recv_next_list)

    return tensor_recv_prev, tensor_recv_next, reqs


def recv_forward_with_reqs(
    tensor_shape: Shape,
    config: ModelParallelConfig,
    is_end_stage: bool = False,
    **kwargs,
):
    """Receive tensor from previous rank in pipeline during forward pass.

    This function receives the input tensor from the previous pipeline stage.
    It returns both the received tensor and the communication requests, allowing
    for asynchronous communication handling.

    Args:
        tensor_shape (Shape): Shape of the tensor to receive. Typically
            (seq_length, micro_batch_size, hidden_size).
        config (ModelParallelConfig): Model parallel configuration containing
            settings like pipeline_dtype, timers, and layerwise_disaggregated_training.
        is_end_stage (bool, optional): Whether this is the end stage in
            layerwise disaggregated training mode. Defaults to False.
        **kwargs: Additional arguments passed to _communicate_impl, including:
            - group (optional): Custom pipeline parallel group for communication.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - input_tensor: The received tensor from previous rank, or None if
              this is the first pipeline stage.
            - reqs: Communication request objects for asynchronous operations,
              or None if no communication occurred.

    Note:
        - If this is the first pipeline stage (ignore_virtual=True) and not
          the end stage, returns (None, None) as no tensor needs to be received.
        - Communication requests can be used to wait for completion if needed.
        - Timers are used to track communication performance if configured.
    """

    if core.parallel_state.is_pipeline_first_stage(ignore_virtual=True) and not is_end_stage:
        input_tensor = None
        reqs = None
    else:
        if config.timers is not None:
            config.timers('forward-recv', log_level=2).start()
        input_tensor, _, reqs = _communicate_impl(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=True,
            recv_next=False,
            tensor_shape=tensor_shape,
            config=config,
            **kwargs,
        )
        if config.timers is not None:
            config.timers('forward-recv').stop()
    return input_tensor, reqs


def recv_backward_with_reqs(
    tensor_shape: Shape,
    config: ModelParallelConfig,
    is_end_stage: bool = False,
    **kwargs,
):
    """Receive gradient tensor from next rank in pipeline during backward pass.

    This function receives the output gradient tensor from the next pipeline stage.
    It returns both the received gradient tensor and the communication requests,
    allowing for asynchronous communication handling.

    Args:
        tensor_shape (Shape): Shape of the gradient tensor to receive. Typically
            (seq_length, micro_batch_size, hidden_size).
        config (ModelParallelConfig): Model parallel configuration containing
            settings like pipeline_dtype, timers, and layerwise_disaggregated_training.
        is_end_stage (bool, optional): Whether this is the end stage in
            layerwise disaggregated training mode. Defaults to False.
        **kwargs: Additional arguments passed to _communicate_impl, including:
            - group (optional): Custom pipeline parallel group for communication.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - output_tensor_grad: The received gradient tensor from next rank,
              or None if no gradient needs to be received.
            - reqs: Communication request objects for asynchronous operations,
              or None if no communication occurred.

    Note:
        - In standard pipeline training, the last stage does not receive gradients.
        - In layerwise disaggregated training mode, the first stage at the end
          stage does not receive gradients.
        - Communication requests can be used to wait for completion if needed.
        - Timers are used to track communication performance if configured.
    """
    output_tensor_grad = None
    reqs = None
    if not config.layerwise_disaggregated_training and core.parallel_state.is_pipeline_last_stage(ignore_virtual=True):
        pass
    elif config.layerwise_disaggregated_training and core.parallel_state.is_pipeline_first_stage(ignore_virtual=True) and is_end_stage:
        pass
    else:
        if config.timers is not None:
            config.timers('backward-recv', log_level=2).start()
        _, output_tensor_grad, reqs = _communicate_impl(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            tensor_shape=tensor_shape,
            config=config,
            **kwargs,
        )
        if config.timers is not None:
            config.timers('backward-recv').stop()
    return output_tensor_grad, reqs


def send_forward(
    output_tensor: torch.Tensor,
    config: ModelParallelConfig,
    is_end_stage: bool = False,
    **kwargs,
) -> None:
    """Send tensor to next rank in pipeline (forward send).

    See _communicate_impl for argument details.
    """
    if not config.layerwise_disaggregated_training and core.parallel_state.is_pipeline_last_stage(ignore_virtual=True):
        pass
    elif config.layerwise_disaggregated_training and core.parallel_state.is_pipeline_first_stage(ignore_virtual=True) and is_end_stage:
        pass
    else:
        if config.timers is not None:
            config.timers('forward-send', log_level=2).start()
        _communicate_impl(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=False,
            tensor_shape=None,
            config=config,
            **kwargs,
        )
        if config.timers is not None:
            config.timers('forward-send').stop()


def send_backward(
    input_tensor_grad: torch.Tensor,
    config: ModelParallelConfig,
    is_end_stage: bool = False,
    **kwargs,
) -> None:
    """Send tensor to previous rank in pipeline (backward send).

    See _communicate_impl for argument details.
    """
    if not core.parallel_state.is_pipeline_first_stage(ignore_virtual=True) or is_end_stage:
        if config.timers is not None:
            config.timers('backward-send', log_level=2).start()
        _communicate_impl(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=False,
            recv_next=False,
            tensor_shape=None,
            config=config,
            **kwargs,
        )
        if config.timers is not None:
            config.timers('backward-send').stop()


class _VTPRecvWork:
    """Wraps irecv work + deferred broadcast into a single waitable object.
    When wait() is called:
        1. rank0 waits for irecv to complete
        2. All ranks participate in broadcast to distribute data within the stage
    """

    def __init__(self, irecv_work, tensor, broadcast_src, intra_group, dst_size):
        self._irecv_work = irecv_work
        self._tensor = tensor
        self._broadcast_src = broadcast_src
        self._intra_group = intra_group
        self._dst_size = dst_size

    def wait(self):
        if self._irecv_work is not None:
            # Handle both single work and list of works
            works = self._irecv_work if isinstance(self._irecv_work, list) else [self._irecv_work]
            for work in works:
                if work is not None:
                    work.wait()
        if self._dst_size > 1 and self._intra_group is not None:
            torch.distributed.broadcast(
                self._tensor,
                src=self._broadcast_src,
                group=self._intra_group,
            )


def vtp_recv_forward(tensor_shape, config, async_op=False, **kwargs):
    """VTP forward recv: rank0 receives from prev stage's rank0, then broadcasts.

    Wraparound (first←last) is handled by get_pipeline_model_parallel_prev_rank().
    """
    stage_idx = get_vtp_my_stage_idx()
    stage_ranks = get_vtp_stage_ranks()
    vtp_size_list = get_vtp_size_list()
    dst_size = vtp_size_list[stage_idx]
    intra_group = get_tensor_model_parallel_group()
    broadcast_src = stage_ranks[stage_idx][0]

    group = kwargs.get("group", None)
    if group is not None:
        pp_group = group
    else:
        pp_group = get_pipeline_model_parallel_group()

    # vitural dp scenario, get nr and pr from kwargs
    next_rank = kwargs.get("next_rank", None)
    prev_rank = kwargs.get("prev_rank", None)
    if next_rank is None:
        next_rank = get_pipeline_model_parallel_next_rank()
    if prev_rank is None:
        prev_rank = get_pipeline_model_parallel_prev_rank()

    if not isinstance(pp_group, list):
        pp_group = [pp_group]
    if not isinstance(next_rank, list):
        next_rank = [next_rank]
    if not isinstance(prev_rank, list):
        prev_rank = [prev_rank]

    tensor = torch.empty(
        tensor_shape,
        requires_grad=True,
        device=torch.cuda.current_device(),
        dtype=config.pipeline_dtype,
    )

    if async_op:
        irecv_works = []
        if is_vtp_stage_rank0():
            for g, sr in zip(pp_group, prev_rank):
                work = torch.distributed.irecv(tensor, src=sr, group=g)
                irecv_works.append(work)

        vtp_work = _VTPRecvWork(
            irecv_works, tensor, broadcast_src, intra_group, dst_size
        )

        return tensor, {"recv_prev": vtp_work}
    else:
        if is_vtp_stage_rank0():
            for g, sr in zip(pp_group, prev_rank):
                work = torch.distributed.irecv(tensor, src=sr, group=g)
                work.wait()

        if dst_size > 1 and intra_group is not None:
            torch.distributed.broadcast(
                tensor, src=broadcast_src, group=intra_group
            )
        return tensor


def vtp_recv_backward(tensor_shape, config, async_op=False, **kwargs):
    """VTP backward recv: rank0 receives gradient from next stage's rank0, then broadcasts.
    Wraparound (last←first) is handled by get_pipeline_model_parallel_next_rank().
    """
    stage_idx = get_vtp_my_stage_idx()
    stage_ranks = get_vtp_stage_ranks()
    vtp_size_list = get_vtp_size_list()
    dst_size = vtp_size_list[stage_idx]
    intra_group = get_tensor_model_parallel_group()
    broadcast_src = stage_ranks[stage_idx][0]

    group = kwargs.get("group", None)
    if group is not None:
        pp_group = group
    else:
        pp_group = get_pipeline_model_parallel_group()

    # vitural dp scenario, get nr and pr from kwargs
    next_rank = kwargs.get("next_rank", None)
    prev_rank = kwargs.get("prev_rank", None)
    if next_rank is None:
        next_rank = get_pipeline_model_parallel_next_rank()
    if prev_rank is None:
        prev_rank = get_pipeline_model_parallel_prev_rank()

    if not isinstance(pp_group, list):
        pp_group = [pp_group]
    if not isinstance(next_rank, list):
        next_rank = [next_rank]
    if not isinstance(prev_rank, list):
        prev_rank = [prev_rank]

    tensor = torch.empty(
        tensor_shape,
        requires_grad=True,
        device=torch.cuda.current_device(),
        dtype=config.pipeline_dtype,
    )

    if async_op:
        irecv_works = []
        if is_vtp_stage_rank0():
            for g, sr in zip(pp_group, next_rank):
                work = torch.distributed.irecv(tensor, src=sr, group=g)
                irecv_works.append(work)
        vtp_work = _VTPRecvWork(
            irecv_works, tensor, broadcast_src, intra_group, dst_size
        )

        return tensor, {"recv_next": vtp_work}
    else:
        if is_vtp_stage_rank0():
            for g, sr in zip(pp_group, next_rank):
                work = torch.distributed.irecv(tensor, src=sr, group=g)
                work.wait()

        if dst_size > 1 and intra_group is not None:
            torch.distributed.broadcast(
                tensor, src=broadcast_src, group=intra_group
            )
        return tensor


def vtp_send_forward(tensor, **kwargs):
    """VTP forward send: rank0 async-sends to next stage's rank0.

    Wraparound (last→first) is handled by get_pipeline_model_parallel_next_rank()
    since VTP sets _PIPELINE_GLOBAL_RANKS = rank0_list.

    Returns the isend work handle (or None for non-rank0 ranks).
    """
    if is_vtp_stage_rank0():
        group = kwargs.get("group", None)
        if group is not None:
            pp_group = group
        else:
            pp_group = get_pipeline_model_parallel_group()

        # vitural dp scenario, get nr and pr from kwargs
        next_rank = kwargs.get("next_rank", None)
        prev_rank = kwargs.get("prev_rank", None)
        if next_rank is None:
            next_rank = get_pipeline_model_parallel_next_rank()
        if prev_rank is None:
            prev_rank = get_pipeline_model_parallel_prev_rank()

        if not isinstance(pp_group, list):
            pp_group = [pp_group]
        if not isinstance(next_rank, list):
            next_rank = [next_rank]
        if not isinstance(prev_rank, list):
            prev_rank = [prev_rank]

        handles = []
        for g, dr in zip(pp_group, next_rank):
            h = torch.distributed.isend(tensor, dst=dr, group=g)
            handles.append(h)
        return handles

    return None


def vtp_send_backward(tensor, **kwargs):
    """VTP backward send: rank0 async-sends gradient to prev stage's rank0.

    Wraparound (first→last) is handled by get_pipeline_model_parallel_prev_rank().

    Returns the isend work handle (or None for non-rank0 ranks).
    """
    if is_vtp_stage_rank0():
        group = kwargs.get("group", None)
        if group is not None:
            pp_group = group
        else:
            pp_group = get_pipeline_model_parallel_group()

        # vitural dp scenario, get nr and pr from kwargs
        next_rank = kwargs.get("next_rank", None)
        prev_rank = kwargs.get("prev_rank", None)
        if next_rank is None:
            next_rank = get_pipeline_model_parallel_next_rank()
        if prev_rank is None:
            prev_rank = get_pipeline_model_parallel_prev_rank()

        if not isinstance(pp_group, list):
            pp_group = [pp_group]
        if not isinstance(next_rank, list):
            next_rank = [next_rank]
        if not isinstance(prev_rank, list):
            prev_rank = [prev_rank]

        handles = []
        for g, dr in zip(pp_group, prev_rank):
            h = torch.distributed.isend(tensor, dst=dr, group=g)
            handles.append(h)
        return handles

    return None
