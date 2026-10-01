# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup
from megatron.core import parallel_state


def finish_grad_sync(self):
    """
    Finishes grad sync (all-reduce or reduce-scatter) communication operations
    for all model gradients.

    When overlap_grad_reduce is set to True, waits for asynchronous communication
    calls to complete. When overlap_grad_reduce is set to False, calls synchronous
    communication ops.
    """
    for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
        finish_grad_sync_ldt(bucket_group)


def finish_grad_sync_ldt(bucket_group: _ParamAndGradBucketGroup):
    """
    Finishes grad sync (all-reduce or reduce-scatter) communication operations
    for all buckets in the bucket group.

    When ddp_config.overlap_grad_reduce is set to True, waits for asynchronous
    communication call to complete. When ddp_config.overlap_grad_reduce is set to False,
    makes synchronous call.
    :param bucket_group: The bucket group to finish grad sync.
    """
    # add: layerwise disaggregated training
    # vitural dp scenario, edge side skip grad sync
    if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
        return

    # If overlap_grad_reduce is False, start (and finish) synchronous communication call here.
    bucket_group.param_gather_dispatched = False
    if not bucket_group.ddp_config.overlap_grad_reduce:
        bucket_group.start_grad_sync()
        return
    # When using partial DP DistOpt, we don't need to sync as we launch comms on a separate
    # communication stream
    if bucket_group.ddp_config.num_distributed_optimizer_instances > 1:
        torch.cuda.default_stream().wait_stream(bucket_group.communication_stream)
        return

    if bucket_group.grad_reduce_handle is None:
        raise AssertionError(
            f'Communication call has not been issued for this bucket '
            f'({len(bucket_group.params_with_grad)}/{len(bucket_group.params)} params have grad available)'
        )
    for handle in bucket_group.grad_reduce_handle:
        handle.wait()
    bucket_group.grad_reduce_handle = None


def register_grad_ready(self, param: torch.nn.Parameter):
    """
    Registers grads for the passed-in param to be "ready" for grad sync.

    When the number of microbatches is greater than 1, we only want to register
    grads as ready when processing the last microbatch and ddp_config.overlap_grad_reduce
    is True.
    """
    # add: layerwise disaggregated training
    # vitural dp scenario, edge side skip grad sync
    if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
        return

    if not self.ddp_config.overlap_grad_reduce:
        raise AssertionError('register_grad_ready() should only be called when overlap_grad_reduce is True')
    if self.is_last_microbatch:
        if param not in self.param_to_bucket:
            raise AssertionError('Param is not in the bucket group')
        if param in self.params_with_grad:
            raise AssertionError('Cannot set grad twice')

        self.params_with_grad.add(param)
        # If all params in bucket group have grads available, issue communication call.
        if len(self.params_with_grad) == len(self.params):
            self.start_grad_sync()
