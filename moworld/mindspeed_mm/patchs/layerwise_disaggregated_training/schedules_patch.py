# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import contextlib
from typing import List, Iterator, Union, Dict, Any

import torch

from megatron.core import mpu, parallel_state
from megatron.core.enums import ModelType
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_next_rank,
    get_pipeline_model_parallel_prev_rank,
    get_pipeline_model_parallel_group,
    get_tensor_model_parallel_group
)
from megatron.core.pipeline_parallel.schedules import (
    backward_step,
    check_first_val_step,
    clear_embedding_activation_buffer,
    deallocate_output_tensor,
    finish_embedding_wgrad_compute,
    forward_backward_no_pipelining,
    get_tensor_shapes,
    set_current_microbatch,
)
from megatron.core.transformer.cuda_graphs import create_cudagraphs
from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler
from megatron.core.utils import (
    get_attr_wrapped_model,
    get_model_config,
    get_model_type,
    get_model_xattn,
)
from megatron.training import get_args
from megatron.training.utils import average_losses_across_data_parallel_group

from mindspeed_mm.patchs.layerwise_disaggregated_training import p2p_communication_patch
from mindspeed_mm.patchs.layerwise_disaggregated_training.parallel_state_patch import (
    get_pipeline_model_parallel_group_alternate,
    get_pipeline_model_parallel_group_last_to_first,
    get_pipeline_model_parallel_group_first_to_last,
    get_vdp_size,
    is_vtp_enabled,
    is_vdp_enabled,
    is_vtp_stage_rank0,
    get_vtp_stage_ranks,
    get_vtp_my_stage_idx,
    get_vtp_size_list
)
from mindspeed_mm.utils.utils import compute_token_level_loss

stream_ping = None
stream_pang = None
stream_last_to_first = None
stream_first_to_last = None
default_stream = None


def move_to_device(batch: Dict[str, Any], float_dtype: str):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            dtype = float_dtype if torch.is_floating_point(v) else None
            batch[k] = v.to(device=torch.cuda.current_device(), dtype=dtype)
        elif isinstance(v, list) and all(isinstance(t, torch.Tensor) for t in v):
            batch[k] = [t.to(device=torch.cuda.current_device(),
                             dtype=float_dtype if torch.is_floating_point(t) else None)
                        for t in v]


def get_batch(data_iterator, is_vit_last_stage=False):
    """Generate a batch."""
    if data_iterator is not None:
        batch = next(data_iterator)
    else:
        raise ValueError("Data iterator is None. Unable to retrieve batch.")
    move_to_device(batch, get_args().params_dtype)
    has_video = 'pixel_values_videos' in batch and 'video_grid_thw' in batch
    if has_video:
        batch['pixel_values'] = batch.pop('pixel_values_videos')
        batch['image_grid_thw'] = batch.pop('video_grid_thw')
    if (mpu.is_pipeline_first_stage() or is_vit_last_stage) and get_args().encoder_dp_balance:
        batch['pixel_values'], batch['tranfer'] = EncoderBalanceComm.apply(
            batch['pixel_values'],
            mpu.get_data_parallel_group())
    else:
        batch['tranfer'] = None
    return batch


def get_tps(output_tensor):
    """Get the tokens per sample"""
    B, S, _ = output_tensor.shape
    dp_size = torch.distributed.get_world_size(group=mpu.get_data_parallel_group())
    cp_size = torch.distributed.get_world_size(group=mpu.get_context_parallel_group())
    tokens_per_sample = torch.tensor(S, device=output_tensor.device) / dp_size * cp_size
    torch.distributed.all_reduce(tokens_per_sample, group=mpu.get_data_parallel_group())
    return tokens_per_sample


def loss_func(output_tensor):
    """Loss function."""
    args = get_args()
    loss_dict = output_tensor['loss_dict']

    loss_dir = {}
    if args.log_tps:
        tokens_per_sample = get_tps(output_tensor['logits'])
        loss_dir["tokens per sample"] = tokens_per_sample

    if args.calculate_per_token_loss:
        loss, local_num_tokens, reporting_loss = compute_token_level_loss(loss_dict)
        loss_dir["loss"] = (reporting_loss[0], reporting_loss[1])
        return (
            loss[0].clone(),
            local_num_tokens,
            loss_dir
        )

    loss = loss_dict['loss']
    averaged_loss = average_losses_across_data_parallel_group([loss])
    loss_dir["loss"] = averaged_loss[0]
    loss = loss.unsqueeze(0).clone()
    return loss / mpu.get_context_parallel_world_size(), loss_dir


def get_forward_backward_func():
    """Retrieves the appropriate forward_backward function given the
    configuration of parallel_state.

    Returns a function that will perform all of the forward and
    backward passes of the model given the pipeline model parallel
    world size and virtual pipeline model parallel world size in the
    global parallel_state.

    Note that if using sequence parallelism, the sequence length component of
    the tensor shape is updated to original_sequence_length /
    tensor_model_parallel_world_size.

    The function returned takes the following arguments:

    forward_step_func (required): A function that takes a data
        iterator and a model as its arguments and return the model's
        forward output and the loss function. The loss function should
        take one torch.Tensor and return a torch.Tensor of loss and a
        dictionary of string -> torch.Tensor.

        A third argument, checkpoint_activations_microbatch, indicates
        that the activations for this microbatch should be
        checkpointed. A None value for this argument indicates that
        the default from the configuration should be used. This is
        used when the
        num_microbatches_with_partial_activation_checkpoints is used.

        For example:

        def loss_func(loss_mask, output_tensor):
            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

            # Reduce loss for logging.
            averaged_loss = average_losses_across_data_parallel_group([loss])

            return loss, {'lm loss': averaged_loss[0]}

        def forward_step(data_iterator, model):
            data, loss_mask = next(data_iterator)
            output = model(data)
            return output, partial(loss_func, loss_mask)


        forward_backward_func(forward_step_func=forward_step, ...)


    data_iterator (required): an iterator over the data, will be
        passed as is to forward_step_func. Expected to be a list of
        iterators in the case of interleaved pipeline parallelism.

    model (required): the actual model. Expected to be a list of modules in the case of interleaved
        pipeline parallelism. Must be a (potentially wrapped) megatron.core.models.MegatronModule.

    num_microbatches (int, required):
        The number of microbatches to go through

    seq_length (int, required): Sequence length of the current global batch. If this is a dual-stack
        transformer, this is the encoder's sequence length. This is ignored if variable_seq_lengths
        in the config is True. Otherwise, each microbatch in the current global batch size must use
        this sequence length.

    micro_batch_size (int, required): The number of sequences in a microbatch.

    decoder_seq_length (int, optional): The sequence length for the decoder in a dual-stack
        transformer. This is ignored for a single-stack transformer.

    forward_only (optional, default = False): Perform only the forward step

    collect_non_loss_data (optional, bool, default=False): TODO

    first_val_step (bool, optional): Is the first step of the validation phase. Used by
        Transformer Engine modules to only update their fp8 weights only on the first validation
        step.

    """
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    if pipeline_model_parallel_size > 1:
        forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        forward_backward_func = forward_backward_no_pipelining

    return forward_backward_func


def forward_step_impl(data_iterator, model, batch=None):
    """Forward step."""
    is_vit_last_stage = False
    if model.module.module.add_image_encoder:
        is_vit_last_stage = model.module.module.image_encoder.post_process

    if batch is None:
        output_tensor = model(**get_batch(data_iterator, is_vit_last_stage))
    elif parallel_state.is_pipeline_first_stage(ignore_virtual=True):
        output_tensor = model(**batch)
    else:
        output_tensor = model(
            input_ids=batch['input_ids'],
            pixel_values=batch['pixel_values'],
            attention_mask=batch['attention_mask'],
            image_grid_thw=batch['image_grid_thw'],
        )
    return output_tensor, loss_func


def forward_step(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensor,
    forward_data_store,
    config,
    collect_non_loss_data=False,
    checkpoint_activations_microbatch=None,
    is_first_microbatch=False,
    current_microbatch=None,
    encoder_decoder_xattn=False,
    is_end_stage=False,
    batch=None,
):
    """Forward step for passed-in model.

    If it is the first stage, the input tensor is obtained from the data_iterator.
    Otherwise, the passed-in input_tensor is used.

    Args:
        forward_step_func (callable):
            The forward step function for the model that takes the
            data iterator as the first argument, and model as the second.
            This user's forward step is expected to output a tuple of two elements:

                1. The output object from the forward step. This output object needs to be a
                    tensor or some kind of collection of tensors. The only hard requirement
                    for this object is that it needs to be acceptible as input into the second
                    function.
                2. A function to reduce (optionally) the output from the forward step. This
                    could be a reduction over the loss from the model, it could be a function that
                    grabs the output from the model and reformats, it could be a function that just
                    passes through the model output. This function must have one of the following
                    patterns, and depending on the pattern different things happen internally:

                        a. A tuple of reduced loss and some other data. Note that in this case
                            the first argument is divided by the number of global microbatches,
                            assuming it is a loss, so that the loss is stable as a function of
                            the number of devices the step is split across.
                        b. A triple of reduced loss, number of tokens, and some other data. This
                            is similar to case (a), but the loss is further averaged across the
                            number of tokens in the batch. If the user is not already averaging
                            across the number of tokens, this pattern is useful to use.
                        c. Any arbitrary data the user wants (eg a dictionary of tensors, a list
                            of tensors, etc in the case of inference). To trigger case 3 you need
                            to specify `collect_non_loss_data=True` and you may also want to
                            specify `forward_only=True` in the call to the parent forward_backward
                            function.
        data_iterator (iterator):
            The data iterator.
        model (nn.Module):
            The model to perform the forward step on.
        num_microbatches (int):
            The number of microbatches.
        input_tensor (Tensor or list[Tensor]):
            The input tensor(s) for the forward step.
        forward_data_store (list):
            The list to store the forward data. If you go down path 2.a or
            2.b for the return of your forward reduction function then this will store only the
            final dimension of the output, for example the metadata output by the loss function.
            If you go down the path of 2.c then this will store the entire output of the forward
            reduction function applied to the model output.
        config (object):
            The configuration object.
        collect_non_loss_data (bool, optional):
            Whether to collect non-loss data. Defaults to False.
            This is the path to use if you want to collect arbitrary output from the model forward,
            such as with inference use cases. Defaults to False.
        checkpoint_activations_microbatch (int, optional):
            The microbatch to checkpoint activations.
            Defaults to None.
        is_first_microbatch (bool, optional):
            Whether it is the first microbatch. Defaults to False.
        current_microbatch (int, optional):
            The current microbatch. Defaults to None.

    Returns:
        Tensor or list[Tensor]: The output object(s) from the forward step.
        Tensor: The number of tokens.
    """
    if config.timers is not None:
        config.timers('forward-compute', log_level=2).start()

    if is_first_microbatch and hasattr(model, 'set_is_first_microbatch'):
        model.set_is_first_microbatch()
    if current_microbatch is not None:
        set_current_microbatch(model, current_microbatch)

    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True

    set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
    set_input_tensor(input_tensor)

    if config.enable_autocast:
        context_manager = torch.autocast("cuda", dtype=config.autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()
    with context_manager:
        if checkpoint_activations_microbatch is None:
            output_tensor, loss_function = forward_step_func(data_iterator, model, batch)
        else:
            output_tensor, loss_function = forward_step_func(
                data_iterator, model, checkpoint_activations_microbatch
            )

    num_tokens = torch.tensor(0, dtype=torch.int)
    # U-shaped split scenario, the first and last layers deploy on pp first stage,
    if parallel_state.is_pipeline_first_stage(ignore_virtual=True) and is_end_stage:
        if not collect_non_loss_data:
            outputs = loss_function(output_tensor)
            if len(outputs) == 3:
                output_tensor, num_tokens, loss_reduced = outputs
                if not config.calculate_per_token_loss:
                    output_tensor /= num_tokens
                    output_tensor *= parallel_state.get_context_parallel_world_size()
                    output_tensor /= num_microbatches
            else:
                # preserve legacy loss averaging behavior (ie, over the number of microbatches)
                if not len(outputs) == 2:
                    raise ValueError()
                output_tensor, loss_reduced = outputs
                output_tensor *= parallel_state.get_context_parallel_world_size()
                output_tensor /= num_microbatches
            forward_data_store.append(loss_reduced)
        else:
            data = loss_function(output_tensor, non_loss_data=True)
            forward_data_store.append(data)

    if config.timers is not None:
        config.timers('forward-compute').stop()

    # Set the loss scale for the auxiliary loss of the MoE layer.
    # Since we use a trick to do backward on the auxiliary loss, we need to set the scale
    # explicitly.
    if hasattr(config, 'num_moe_experts') and config.num_moe_experts is not None:
        # Calculate the loss scale based on the grad_scale_func if available, else default to 1.
        loss_scale = (
            config.grad_scale_func(torch.ones(1, device=output_tensor.device))
            if config.grad_scale_func is not None
            else torch.ones(1, device=output_tensor.device)
        )
        # Set the loss scale
        if config.calculate_per_token_loss:
            MoEAuxLossAutoScaler.set_loss_scale(loss_scale)
        else:
            MoEAuxLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)

    # Set the loss scale for Multi-Token Prediction (MTP) loss.
    if hasattr(config, 'mtp_num_layers') and config.mtp_num_layers is not None:
        # Calculate the loss scale based on the grad_scale_func if available, else default to 1.
        loss_scale = (
            config.grad_scale_func(torch.ones(1, device=output_tensor.device))
            if config.grad_scale_func is not None
            else torch.ones(1, device=output_tensor.device)
        )
        # Set the loss scale
        if config.calculate_per_token_loss:
            MTPLossAutoScaler.set_loss_scale(loss_scale)
        else:
            MTPLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)

    # If T5 model and in decoder stack, then send encoder_hidden_state
    # downstream as well.
    model_type = get_model_type(model)
    if (
        model_type == ModelType.encoder_and_decoder
        and encoder_decoder_xattn
        and parallel_state.is_inside_decoder()
    ):
        return [output_tensor, input_tensor[-1]], num_tokens

    if unwrap_output_tensor:
        return output_tensor, num_tokens
    return [output_tensor], num_tokens


def recv_forward_with_reqs(tensor_shapes, config, is_end_stage: bool = False, **kwargs):
    """Wrapper for p2p_communication_patch.recv_forward used with non-interleaving schedule."""
    input_tensors = []
    reps_list = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            input_tensors.append(None)
        else:
            input_tensor, reqs = p2p_communication_patch.recv_forward_with_reqs(
                tensor_shape, config, is_end_stage, **kwargs
            )
            input_tensors.append(input_tensor)
            reps_list.append(reqs)
    return input_tensors, reps_list


def recv_backward_with_reqs(tensor_shapes, config, is_end_stage=False, **kwargs):
    """Wrapper for p2p_communication_patch.recv_backward used with non-interleaving schedule."""
    output_tensor_grads = []
    reps_list = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            output_tensor_grads.append(None)
        else:
            output_tensor_grad, reqs = p2p_communication_patch.recv_backward_with_reqs(
                tensor_shape, config, is_end_stage, **kwargs
            )
            output_tensor_grads.append(output_tensor_grad)
            reps_list.append(reqs)
    return output_tensor_grads, reps_list


def send_forward(
    output_tensors, tensor_shapes, config, is_end_stage: bool = False, **kwargs
):
    """Wrapper for p2p_communication_patch.send_forward used with non-interleaving schedule."""
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    for output_tensor, tensor_shape in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication_patch.send_forward(output_tensor, config, is_end_stage, wait_on_reqs=False, **kwargs)


def send_backward(
    input_tensor_grads, tensor_shapes, config, is_end_stage: bool = False, **kwargs
):
    """Wrapper for p2p_communication_patch.send_backward used with non-interleaving schedule."""
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    for input_tensor_grad, tensor_shape in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication_patch.send_backward(input_tensor_grad, config, is_end_stage, wait_on_reqs=False, **kwargs)


# VTP schedule wrappers
def _vtp_send_forward_wrapper(output_tensors, tensor_shapes, config, is_end_stage=False, **kwargs):
    """VTP-aware forward send: uses rank0 async P2P. Returns isend work handles."""
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    handles = []
    for output_tensor, tensor_shape in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            continue
        h = p2p_communication_patch.vtp_send_forward(output_tensor, **kwargs)
        if h is not None:
            handles.extend(h)
    return handles


def _vtp_recv_forward_wrapper(tensor_shapes, config, async_op=False, is_end_stage=False, **kwargs):
    """VTP-aware forward recv: uses rank0 irecv + deferred broadcast.

    When async_op=True, returns (input_tensors, reqs_list) for overlap with compute.
    When async_op=False, blocks until recv + broadcast complete.
    """
    input_tensors = []
    reqs_list = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            input_tensors.append(None)
        else:
            if async_op:
                tensor, reqs = p2p_communication_patch.vtp_recv_forward(
                    tensor_shape, config, async_op=True, **kwargs
                )
                input_tensors.append(tensor)
                reqs_list.append(reqs)
            else:
                tensor = p2p_communication_patch.vtp_recv_forward(
                    tensor_shape, config, async_op=False, **kwargs
                )
                input_tensors.append(tensor)
    if async_op:
        return input_tensors, reqs_list
    return input_tensors


def _vtp_send_backward_wrapper(input_tensor_grads, tensor_shapes, config, is_end_stage=False, **kwargs):
    """VTP-aware backward send: uses rank0 async P2P. Returns isend work handles."""
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    handles = []
    for input_tensor_grad, tensor_shape in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            continue
        h = p2p_communication_patch.vtp_send_backward(input_tensor_grad, **kwargs)
        if h is not None:
            handles.extend(h)
    return handles


def _vtp_recv_backward_wrapper(tensor_shapes, config, async_op=False, is_end_stage=False, **kwargs):
    """VTP-aware backward recv: uses rank0 irecv + deferred broadcast.

    When async_op=True, returns (output_tensor_grads, reqs_list) for overlap.
    When async_op=False, blocks until recv + broadcast complete.
    """
    output_tensor_grads = []
    reqs_list = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            output_tensor_grads.append(None)
        else:
            if async_op:
                tensor, reqs = p2p_communication_patch.vtp_recv_backward(
                    tensor_shape, config, async_op=True, **kwargs
                )
                output_tensor_grads.append(tensor)
                reqs_list.append(reqs)
            else:
                tensor = p2p_communication_patch.vtp_recv_backward(
                    tensor_shape, config, async_op=False, **kwargs
                )
                output_tensor_grads.append(tensor)
    if async_op:
        return output_tensor_grads, reqs_list
    return output_tensor_grads


def get_all_batchs(mbn, data_iterator, model, config, vit_hidden_size):

    device = f"npu:{torch.cuda.current_device()}"
    data_type = torch.int64
    hidden_size = config.hidden_size

    all_batchs = [[], [], []]
    recv_tensor_shapes = []
    vit_recv_tensor_shapes = []

    def _split_item(item):
        """
        Split item into get_vdp_size() parts with round-robin strategy.
        example: [1,2,3,4,5,6] vdp=3 -> [[1,4],[2,5],[3,6]]
        """
        len_item = item.size(0)
        device = item.device

        # gen all index
        indices = torch.arange(len_item, device=device)
        part_indices = indices % get_vdp_size()

        item_list = []
        for i in range(get_vdp_size()):
            mask = (part_indices == i)
            item_list.append(item[mask])

        return item_list

    def _broadcast(item):
        if item is not None:
            if is_vtp_enabled() and is_vdp_enabled():
                # vitural DP scenario, broadcast each part to each pipeline stage
                if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                    split_item_list = _split_item(item)
                    for i, group in enumerate(parallel_state.get_pipeline_model_parallel_group()):
                        if not is_vtp_stage_rank0():
                            continue
                        torch.distributed.broadcast(split_item_list[i], parallel_state.get_pipeline_model_parallel_first_rank(), group=group)

                elif is_vtp_stage_rank0():
                    # In VTP scenario, the item received by cloud side is already split
                    torch.distributed.broadcast(item, parallel_state.get_pipeline_model_parallel_first_rank(), group=parallel_state.get_pipeline_model_parallel_group())

                # Cloud side needs to broadcast within TP domain
                if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                    vtp_intra_group = get_tensor_model_parallel_group()
                    if vtp_intra_group is not None:
                        stage_ranks = get_vtp_stage_ranks()
                        my_stage = get_vtp_my_stage_idx()
                        tp_group = parallel_state.get_tensor_model_parallel_group()
                        torch.distributed.broadcast(item, stage_ranks[my_stage][0], group=tp_group)

            elif is_vdp_enabled():
                if isinstance(parallel_state.get_pipeline_model_parallel_group(), list):
                    split_item_list = _split_item(item)
                    for i, group in enumerate(parallel_state.get_pipeline_model_parallel_group()):
                        torch.distributed.broadcast(split_item_list[i], parallel_state.get_pipeline_model_parallel_first_rank(),
                        group=group)
                else:
                    torch.distributed.broadcast(item, parallel_state.get_pipeline_model_parallel_first_rank(),
                    group=parallel_state.get_pipeline_model_parallel_group())
            elif is_vtp_enabled():
                # PP broadcast: only rank0 of each stage participates
                if is_vtp_stage_rank0():
                    torch.distributed.broadcast(item, parallel_state.get_pipeline_model_parallel_first_rank(),
                    group=parallel_state.get_pipeline_model_parallel_group())
                # Intra-stage broadcast: rank0 sends to other ranks in the stage
                vtp_intra_group = get_tensor_model_parallel_group()
                if vtp_intra_group is not None:
                    stage_ranks = get_vtp_stage_ranks()
                    my_stage = get_vtp_my_stage_idx()
                    torch.distributed.broadcast(item, stage_ranks[my_stage][0], group=vtp_intra_group)
            else:
                torch.distributed.broadcast(item, parallel_state.get_pipeline_model_parallel_first_rank(),
                    group=parallel_state.get_pipeline_model_parallel_group())

    def get_batch_infos(attention_infos, thws, shapes, i_forward):
        seq_len, mbs = shapes[i_forward][0][0], shapes[i_forward][0][1]
        attention_mask = torch.ones(mbs, seq_len, device=device, dtype=data_type)

        for i, padding_info in enumerate(attention_infos[i_forward]):
            padding_side, padding_num = padding_info[0], padding_info[1]

            if padding_num == 0:
                continue

            if padding_side == 0:
                attention_mask[i, :padding_num] = torch.zeros(padding_num, device=device, dtype=data_type)
            else:
                attention_mask[i, -padding_num:] = torch.zeros(padding_num, device=device, dtype=data_type)

        image_grid_thw = torch.tensor(thws[i_forward], device=device, dtype=data_type)
        return attention_mask, image_grid_thw

    is_vit_last_stage = False
    if model.module.module.add_image_encoder:
        is_vit_last_stage = model.module.module.image_encoder.post_process

    if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
        mbn = mbn * get_vdp_size()

    tensor_shapes = torch.empty(
        mbn,
        5 + 5 * config.micro_batch_size,
        device=device,
        dtype=data_type
    )

    if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
        for i in range(mbn):
            batch = get_batch(data_iterator[0], is_vit_last_stage)
            mbs, seq_len = batch["input_ids"].shape[0], batch["input_ids"].shape[1]
            if batch['pixel_values'] is not None:
                x_pixel_values, y_pixel_values = batch['pixel_values'].shape[0], batch['pixel_values'].shape[1]
            else:
                x_pixel_values = y_pixel_values = 0
            tensor_shapes[i, :5] = torch.tensor([seq_len, mbs, hidden_size, x_pixel_values, y_pixel_values], device=device, dtype=data_type)

            attention_mask = batch["attention_mask"]  # [mbs, seq_len]
            image_grid_thw = batch["image_grid_thw"]  # [mbs, 3]

            padding_side = (attention_mask[:, 0] != 0).long().unsqueeze(1)  # [mbs, 1]
            padding_num = (seq_len - attention_mask.sum(dim=1)).unsqueeze(1)  # [mbs, 1]
            tensor_shapes[i][5:] = torch.cat([padding_side, padding_num, image_grid_thw], dim=1).flatten()  # [mbs * 5, ]

            tensor_shape = [(seq_len, mbs, config.hidden_size)]
            vit_tensor_shape = [(x_pixel_values, 1, vit_hidden_size)]

            all_batchs[0].append(batch)
            all_batchs[1].append(batch)
            all_batchs[2].append(batch)
            recv_tensor_shapes.append(tensor_shape)
            vit_recv_tensor_shapes.append(vit_tensor_shape)

        _broadcast(tensor_shapes)
    else:
        _broadcast(tensor_shapes)

        tensor_shapes_tolist = tensor_shapes.tolist()

        shapes = [[tuple(shape[:3])] for shape in tensor_shapes_tolist]
        recv_tensor_shapes = shapes

        vit_shapes = [[(shape[3], 1, vit_hidden_size)] for shape in tensor_shapes_tolist]
        vit_recv_tensor_shapes = vit_shapes

        pixel_shapes = [(shape[3], shape[4]) for shape in tensor_shapes_tolist]
        attention_infos = [[(shape[5 + 5 * i: 7 + 5 * i]) for i in range(config.micro_batch_size)] for shape in tensor_shapes_tolist]
        thws = [[(shape[7 + 5 * i: 10 + 5 * i]) for i in range(config.micro_batch_size)] for shape in tensor_shapes_tolist]

        for i in range(mbn):
            x_pixel_values, y_pixel_values = pixel_shapes[i][0], pixel_shapes[i][1]
            if x_pixel_values > 0 and y_pixel_values > 0:
                pixel_values = torch.zeros(x_pixel_values, y_pixel_values, device=device, dtype=torch.bfloat16)
            else:
                pixel_values = None
            seq_len, mbs = shapes[i][0][0], shapes[i][0][1]
            input_ids = torch.zeros(mbs, seq_len, device=device, dtype=data_type)
            attention_mask, image_grid_thw = get_batch_infos(attention_infos, thws, shapes, i)
            batch = {
                'input_ids': input_ids,
                'labels': None,
                'pixel_values': pixel_values,
                'attention_mask': attention_mask,
                'image_grid_thw': image_grid_thw,
                'tranfer': None
            }

            all_batchs[0].append(batch)
            all_batchs[1].append(batch)
            all_batchs[2].append(batch)

    return all_batchs, recv_tensor_shapes, vit_recv_tensor_shapes


def forward_backward_pipelining_without_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """
    Run non-interleaved 1F1B schedule, with communication between pipeline
    stages. Returns dictionary with losses if the last stage, empty dict otherwise.
    """

    if not isinstance(model, list):
        raise TypeError("cloud-edge pipeline parallelism expected model chunking")
    if not all(isinstance(chunk, torch.nn.Module) for chunk in model):
        raise TypeError("invalid model chunking")

    if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
        data_iterator = [None]

    config = get_model_config(model[0])
    config.variable_seq_lengths = False
    config.layerwise_disaggregated_training = True
    forward_step_func = forward_step_impl

    # Needed only when gradients are finalized in M-Core
    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = clear_embedding_activation_buffer(config, model)

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if isinstance(no_sync_func, list):
            for func in no_sync_func:
                no_sync_context = func()
                no_sync_context.__enter__()
        else:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Compute number of warmup microbatches.
    num_warmup_microbatches = (
        parallel_state.get_pipeline_model_parallel_world_size()
        - parallel_state.get_pipeline_model_parallel_rank()
    )
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    model_type = get_model_type(model[0])
    encoder_decoder_xattn = get_model_xattn(model[0])

    rank = parallel_state.get_pipeline_model_parallel_rank()
    recv_tensor_shapes = get_tensor_shapes(
        rank=rank - 1,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        encoder_decoder_xattn=encoder_decoder_xattn,
    )
    send_tensor_shapes = get_tensor_shapes(
        rank=rank,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        encoder_decoder_xattn=encoder_decoder_xattn,
    )

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = []
    output_tensors = []
    vit_input_tensors = []
    vit_output_tensors = []
    total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()
    forward_data_store = []

    global default_stream
    if default_stream is None:
        default_stream = torch.cuda.default_stream()

    global stream_ping
    if stream_ping is None:
        stream_ping = torch.cuda.Stream()

    global stream_pang
    if stream_pang is None:
        stream_pang = torch.cuda.Stream()

    global stream_last_to_first
    if stream_last_to_first is None:
        stream_last_to_first = torch.cuda.Stream()

    global stream_first_to_last
    if stream_first_to_last is None:
        stream_first_to_last = torch.cuda.Stream()

    group_ping = get_pipeline_model_parallel_group()
    group_pang = get_pipeline_model_parallel_group_alternate()
    group_last_to_first = get_pipeline_model_parallel_group_last_to_first()
    group_first_to_last = get_pipeline_model_parallel_group_first_to_last()

    # VTP: detect asymmetric boundaries (including U-shape wraparound)
    vtp_active = is_vtp_enabled()
    vtp_need_asymmetric_fwd = False
    vtp_need_asymmetric_bwd = False
    vtp_send_forward_group = None
    vtp_recv_forward_group = None
    vtp_send_backward_group = None
    vtp_recv_backward_group = None

    if vtp_active:
        vtp_size_list = get_vtp_size_list()
        my_stage = get_vtp_my_stage_idx()
        pp_size = parallel_state.get_pipeline_model_parallel_world_size()

        # Check forward/backward asymmetric boundaries with wraparound
        next_stage = (my_stage + 1) % pp_size
        prev_stage = (my_stage - 1) % pp_size
        vtp_need_asymmetric_fwd = vtp_size_list[my_stage] != vtp_size_list[next_stage]
        vtp_need_asymmetric_bwd = vtp_size_list[my_stage] != vtp_size_list[prev_stage]

    if parallel_state.get_pipeline_model_parallel_rank() % 2 == 0:
        receive_forward_stream = receive_backward_stream = stream_ping
        send_forward_stream = send_backward_stream = stream_pang
        receive_forward_group = receive_backward_group = group_ping
        send_forward_group = send_backward_group = group_pang
    else:
        receive_forward_stream = receive_backward_stream = stream_pang
        send_forward_stream = send_backward_stream = stream_ping
        receive_forward_group = receive_backward_group = group_pang
        send_forward_group = send_backward_group = group_ping

    # when pp size is odd, additional communication streams need to be added
    # to the first and last layers
    if parallel_state.get_pipeline_model_parallel_world_size() % 2 == 1:
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            receive_forward_stream = stream_last_to_first
            receive_forward_group = group_last_to_first
            send_backward_stream = stream_first_to_last
            send_backward_group = group_first_to_last
        elif parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            receive_backward_stream = stream_first_to_last
            receive_backward_group = group_first_to_last
            send_forward_stream = stream_last_to_first
            send_forward_group = group_last_to_first

    # VTP reuses the standard PP groups (already rank0-only after VTP init)
    if vtp_need_asymmetric_fwd or vtp_need_asymmetric_bwd:
        vtp_send_forward_group = send_forward_group
        vtp_recv_forward_group = receive_forward_group
        vtp_send_backward_group = send_backward_group
        vtp_recv_backward_group = receive_backward_group

    if not isinstance(receive_forward_group, list):
        receive_forward_group = [receive_forward_group]
    if not isinstance(receive_backward_group, list):
        receive_backward_group = [receive_backward_group]
    if not isinstance(send_forward_group, list):
        send_forward_group = [send_forward_group]
    if not isinstance(send_backward_group, list):
        send_backward_group = [send_backward_group]

    _vtp_pending_sends = []

    def wait_helper(reqs_list):
        is_wait = False
        recv_prev = False
        for reqs in reqs_list:
            if reqs is None:
                continue
            if "recv_prev" in reqs.keys():
                recv_prev = True
            for req in reqs if isinstance(reqs, list) else reqs.values():
                req.wait()
                is_wait = True
        if is_wait:
            if recv_prev:
                default_stream.wait_stream(receive_forward_stream)
            else:
                default_stream.wait_stream(receive_backward_stream)
        reqs_list = []

    def send_forward_with_stream(
        output_tensor, send_tensor_shapes, config, is_end_stage=False, **kwargs
    ):
        with torch.cuda.stream(send_forward_stream):
            send_forward_stream.wait_stream(default_stream)
            if vtp_need_asymmetric_fwd:
                # LDT: first stage + end_stage = end of U-shape forward, no send
                if (parallel_state.is_pipeline_first_stage(ignore_virtual=True)
                        and is_end_stage):
                    return

                # In VDP scenario, vtp_send_forward_group cannot be used directly; instead, kwargs should be passed through which contains the group
                # VDP processes data from different DP domains sequentially via for loop. vtp_send_forward_group directly gets a list containing multiple DP domains. Sending data from one DP domain to multiple DP domains causes issues.
                if is_vdp_enabled():
                    handles = _vtp_send_forward_wrapper(
                        output_tensor, send_tensor_shapes, config,
                        is_end_stage=is_end_stage, **kwargs
                    )
                    _vtp_pending_sends.extend(handles)

                else:
                    handles = _vtp_send_forward_wrapper(
                        output_tensor, send_tensor_shapes, config,
                        group=vtp_send_forward_group,
                        is_end_stage=is_end_stage,
                    )
                    _vtp_pending_sends.extend(handles)
            else:
                send_forward(
                    output_tensor, send_tensor_shapes, config, is_end_stage, **kwargs
                )
            if output_tensor is not None:
                if isinstance(output_tensor, list):
                    for output_tensor_i in output_tensor:
                        if output_tensor_i is not None:
                            output_tensor_i.record_stream(send_forward_stream)
                else:
                    output_tensor.record_stream(send_forward_stream)

    def recv_forward_with_stream(
        recv_tensor_shapes, config, is_end_stage=False, **kwargs
    ):
        with torch.cuda.stream(receive_forward_stream):
            if vtp_need_asymmetric_bwd:
                # First stage doesn't recv in normal forward (only in wraparound)
                if (parallel_state.is_pipeline_first_stage(ignore_virtual=True) and not is_end_stage):
                    default_stream.wait_stream(receive_forward_stream)
                    if kwargs.get("wait_on_reqs", True):
                        return [None]
                    return [None], []

                wait_on_reqs = kwargs.get("wait_on_reqs", True)

                # VTP async path: irecv (non-blocking) + deferred broadcast
                vtp_group = vtp_recv_forward_group

                if is_vdp_enabled():
                    if wait_on_reqs:
                        # Synchronous: recv + broadcast, then sync
                        input_tensor = _vtp_recv_forward_wrapper(
                            recv_tensor_shapes, config,
                            async_op=False, is_end_stage=is_end_stage, **kwargs
                        )
                        for input_tensor_i in input_tensor:
                            if input_tensor_i is not None:
                                input_tensor_i.record_stream(default_stream)
                        default_stream.wait_stream(receive_forward_stream)

                        return input_tensor
                    else:
                        # Async: irecv returns immediately, broadcast deferred to wait_helper
                        input_tensor, reqs_list = _vtp_recv_forward_wrapper(
                            recv_tensor_shapes, config,
                            async_op=True, is_end_stage=is_end_stage, **kwargs
                        )
                        for input_tensor_i in input_tensor:
                            if input_tensor_i is not None:
                                input_tensor_i.record_stream(default_stream)

                        return input_tensor, reqs_list
                else:
                    # VTP async path: irecv (non-blocking) + deferred broadcast
                    vtp_group = vtp_recv_forward_group

                    if wait_on_reqs:
                        # Synchronous: recv + broadcast, then sync
                        input_tensor = _vtp_recv_forward_wrapper(
                            recv_tensor_shapes, config, group=vtp_group,
                            async_op=False, is_end_stage=is_end_stage,
                        )
                        for input_tensor_i in input_tensor:
                            if input_tensor_i is not None:
                                input_tensor_i.record_stream(default_stream)
                        default_stream.wait_stream(receive_forward_stream)
                        return input_tensor
                    else:
                        # Async: irecv returns immediately, broadcast deferred to wait_helper
                        input_tensor, reqs_list = _vtp_recv_forward_wrapper(
                            recv_tensor_shapes, config, group=vtp_group,
                            async_op=True, is_end_stage=is_end_stage,
                        )
                        for input_tensor_i in input_tensor:
                            if input_tensor_i is not None:
                                input_tensor_i.record_stream(default_stream)
                        return input_tensor, reqs_list
            else:
                input_tensor, reqs_list = recv_forward_with_reqs(
                    recv_tensor_shapes, config, is_end_stage, **kwargs
                )
                for input_tensor_i in input_tensor:
                    if input_tensor_i is not None:
                        input_tensor_i.record_stream(default_stream)

        if "wait_on_reqs" in kwargs.keys():
            if kwargs["wait_on_reqs"] is True:
                default_stream.wait_stream(receive_forward_stream)
                return input_tensor
        else:
            default_stream.wait_stream(receive_forward_stream)
            return input_tensor
        return input_tensor, reqs_list

    def send_backward_with_stream(
        input_tensor_grad, recv_tensor_shapes, config, is_end_stage=False, **kwargs
    ):
        with torch.cuda.stream(send_backward_stream):
            send_backward_stream.wait_stream(default_stream)
            if vtp_need_asymmetric_bwd:
                # First stage doesn't send backward in normal backward
                if (parallel_state.is_pipeline_first_stage(ignore_virtual=True)
                        and not is_end_stage):
                    return

                if is_vdp_enabled():
                    handles = _vtp_send_backward_wrapper(
                        input_tensor_grad, recv_tensor_shapes, config,
                        is_end_stage=is_end_stage, **kwargs
                    )
                    _vtp_pending_sends.extend(handles)

                else:
                    handles = _vtp_send_backward_wrapper(
                        input_tensor_grad, recv_tensor_shapes, config,
                        group=vtp_send_backward_group,
                        is_end_stage=is_end_stage,
                    )
                    _vtp_pending_sends.extend(handles)
            else:
                send_backward(
                    input_tensor_grad, recv_tensor_shapes, config, is_end_stage, **kwargs
                )
            if input_tensor_grad is not None:
                if isinstance(input_tensor_grad, list):
                    for input_tensor_grad_i in input_tensor_grad:
                        if input_tensor_grad_i is not None:
                            input_tensor_grad_i.record_stream(send_backward_stream)
                else:
                    input_tensor_grad.record_stream(send_backward_stream)

    def recv_backward_with_stream(
        recv_tensor_shapes, config, is_end_stage=False, **kwargs
    ):
        wait_on_reqs = kwargs.get("wait_on_reqs", True)

        with torch.cuda.stream(receive_backward_stream):
            if vtp_need_asymmetric_fwd:
                # LDT: first stage + end_stage = no backward recv
                if (parallel_state.is_pipeline_first_stage(ignore_virtual=True)
                        and is_end_stage):
                    default_stream.wait_stream(receive_backward_stream)
                    return [None], []
                # VTP async path for backward recv
                vtp_group = vtp_recv_backward_group

                if is_vdp_enabled():
                    if wait_on_reqs:
                        output_tensor_grad = _vtp_recv_backward_wrapper(
                            recv_tensor_shapes, config,
                            async_op=False, is_end_stage=is_end_stage, **kwargs
                        )
                        for output_tensor_grad_i in output_tensor_grad:
                            if output_tensor_grad_i is not None:
                                output_tensor_grad_i.record_stream(default_stream)
                        default_stream.wait_stream(receive_backward_stream)
                        return output_tensor_grad, []
                    else:
                        output_tensor_grad, reqs_list = _vtp_recv_backward_wrapper(
                            recv_tensor_shapes, config,
                            async_op=True, is_end_stage=is_end_stage, **kwargs
                        )
                        for output_tensor_grad_i in output_tensor_grad:
                            if output_tensor_grad_i is not None:
                                output_tensor_grad_i.record_stream(default_stream)
                        return output_tensor_grad, reqs_list
                else:
                    if wait_on_reqs:
                        output_tensor_grad = _vtp_recv_backward_wrapper(
                            recv_tensor_shapes, config, group=vtp_group,
                            async_op=False, is_end_stage=is_end_stage,
                        )
                        for output_tensor_grad_i in output_tensor_grad:
                            if output_tensor_grad_i is not None:
                                output_tensor_grad_i.record_stream(default_stream)
                        default_stream.wait_stream(receive_backward_stream)
                        return output_tensor_grad, []
                    else:
                        output_tensor_grad, reqs_list = _vtp_recv_backward_wrapper(
                            recv_tensor_shapes, config, group=vtp_group,
                            async_op=True, is_end_stage=is_end_stage,
                        )
                        for output_tensor_grad_i in output_tensor_grad:
                            if output_tensor_grad_i is not None:
                                output_tensor_grad_i.record_stream(default_stream)
                        return output_tensor_grad, reqs_list
            else:
                output_tensor_grad, reqs_list = recv_backward_with_reqs(
                    recv_tensor_shapes, config, is_end_stage, **kwargs
                )
                for output_tensor_grad_i in output_tensor_grad:
                    if output_tensor_grad_i is not None:
                        output_tensor_grad_i.record_stream(default_stream)

        if wait_on_reqs:
            default_stream.wait_stream(receive_backward_stream)
            return output_tensor_grad, []

        return output_tensor_grad, reqs_list

    # 读取vit的隐藏层维度
    if hasattr(model[0], 'module'):
        float16_wrapper = model[0].module
    else:
        float16_wrapper = model[0]

    if hasattr(float16_wrapper, 'module'):
        ldt_vlm_model = float16_wrapper.module
    else:
        ldt_vlm_model = float16_wrapper

    vit_hidden_size = 0
    if hasattr(ldt_vlm_model, 'image_encoder'):
        try:
            mlp_linear = ldt_vlm_model.image_encoder.encoder.blocks.layers[0].mlp.linear_fc1
            vit_hidden_size = mlp_linear.input_size
        except Exception as e:
            raise AssertionError(f"Failed to read hidden size from VIT: {str(e)}")

    # 提前获得本轮迭代的所有数据信息
    all_batchs, recv_forward_tensor_shapes, vit_recv_fwd_tensor_shapes = get_all_batchs(
        num_microbatches, data_iterator, model[0], config, vit_hidden_size)

    recv_backward_tensor_shapes = recv_forward_tensor_shapes.copy()
    vit_recv_bwd_tensor_shapes = vit_recv_fwd_tensor_shapes.copy()

    pp_group = get_pipeline_model_parallel_group()
    if not isinstance(pp_group, list):
        pp_group = [pp_group]

    # virtual dp scenario, get next_rank and prev_rank
    next_rank = get_pipeline_model_parallel_next_rank()
    if not isinstance(next_rank, list):
        next_rank = [next_rank]
    prev_rank = get_pipeline_model_parallel_prev_rank()
    if not isinstance(prev_rank, list):
        prev_rank = [prev_rank]

    num_vit_warmup = min(parallel_state.get_pipeline_model_parallel_world_size(), num_microbatches)
    num_forward_end_backward_start = int(
        (4 * parallel_state.get_pipeline_model_parallel_world_size() + 1) / 6 + .00001
    )

    input_tensor_tmp = None
    vit_input_tensor_tmp = None
    vdp_input_tensor_tmp = None
    input_tensor_queue = []
    vit_input_tensor_queue = []
    vdp_input_tensor_queue = []
    reqs_list = []
    vit_reqs_list = []
    vdp_reqs_list = []
    reqs_queue = []
    vit_reqs_queue = []
    vdp_reqs_queue = []

    mbn = num_microbatches * get_vdp_size() if parallel_state.is_pipeline_first_stage(ignore_virtual=True) else num_microbatches
    vit_fwd_num = vit_bwd_num = mbn
    llm_fwd_num = llm_bwd_num = mbn
    last_stage_febs_num = mbn

    group_iter = [i_group for i_group in range(len(pp_group))]

    def set_vpp_rank(vpp_rank):
        parallel_state.set_virtual_pipeline_model_parallel_rank(vpp_rank)

    # Run VIT warmup forward passes.
    set_vpp_rank(0)
    for i in range(num_vit_warmup):
        last_iteration = i == (num_vit_warmup - 1)

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                i % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        # 首轮不掩盖，直接recv vit fwd
        if i == 0:
            for rfg, nr, pr in zip(receive_forward_group, next_rank, prev_rank):
                if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                    recv_tensor_shapes = vit_recv_fwd_tensor_shapes.pop(0)
                input_tensor = recv_forward_with_stream(
                    recv_tensor_shapes,
                    config,
                    group=rfg,
                    next_rank=nr,
                    prev_rank=pr,
                )
                input_tensor_queue.append(input_tensor)
                reqs_queue.append([])

        for i_group, sfg, nr, pr in zip(group_iter, send_forward_group, next_rank, prev_rank):
            # 传算掩盖：对上一轮recv vit fwd进行wait
            reqs_list = reqs_queue.pop(0)
            wait_helper(reqs_list)

            # 传算掩盖：提前recv下一轮vit fwd
            if i_group == 0 and not last_iteration:
                for future_rfg, future_nr, future_pr in zip(receive_forward_group, next_rank, prev_rank):
                    if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                        # 首PP不需要接收，在队列中用None占位
                        input_tensor_queue.append([None])
                        reqs_queue.append([])
                    else:
                        recv_tensor_shapes = vit_recv_fwd_tensor_shapes.pop(0)
                        input_tensor_tmp, reqs_list = recv_forward_with_stream(
                            recv_tensor_shapes,
                            config,
                            group=future_rfg,
                            next_rank=future_nr,
                            prev_rank=future_pr,
                            wait_on_reqs=False,
                        )
                        input_tensor_queue.append(input_tensor_tmp)
                        reqs_queue.append(reqs_list)

            # 传算掩盖：最后一轮提前接收warmup阶段的第一轮llm fwd
            set_vpp_rank(1)
            if i_group == 0 and last_iteration and num_warmup_microbatches > 0:
                for future_rfg, future_nr, future_pr in zip(receive_forward_group, next_rank, prev_rank):
                    if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                        recv_tensor_shapes = vit_recv_fwd_tensor_shapes.pop(0)
                    else:
                        recv_tensor_shapes = recv_forward_tensor_shapes.pop(0)
                    input_tensor_tmp, reqs_list = recv_forward_with_stream(
                        recv_tensor_shapes,
                        config,
                        group=future_rfg,
                        next_rank=future_nr,
                        prev_rank=future_pr,
                        is_end_stage=True,
                        wait_on_reqs=False,
                    )
                    input_tensor_queue.append(input_tensor_tmp)
                    reqs_queue.append(reqs_list)

            set_vpp_rank(0)

            # vit fwd
            this_iterator = None
            this_model = model[0]
            input_tensor = input_tensor_queue.pop(0)
            output_tensor, num_tokens = forward_step(
                forward_step_func,
                this_iterator,
                this_model,
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                collect_non_loss_data,
                checkpoint_activations_microbatch,
                check_first_val_step(first_val_step, forward_only, i == 0),
                current_microbatch=i,
                encoder_decoder_xattn=encoder_decoder_xattn,
                batch=all_batchs[0].pop(0)
            )
            vit_fwd_num -= 1
            total_num_tokens += num_tokens

            # vit send fwd
            send_forward_with_stream(
                output_tensor,
                send_tensor_shapes,
                config,
                group=sfg,
                next_rank=nr,
                prev_rank=pr,
            )

            if not forward_only:
                vit_input_tensors.append(input_tensor)
                vit_output_tensors.append(output_tensor)
                deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        set_vpp_rank(1)
        last_iteration = i == (num_warmup_microbatches - 1)

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                i % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        for i_group, sfg, nr, pr in zip(group_iter, send_forward_group, next_rank, prev_rank):

            # 传算掩盖：阻塞式接收llm fwd
            reqs_list = reqs_queue.pop(0)
            wait_helper(reqs_list)

            # 传算掩盖：第num_forward_end_backward_start轮llm fwd之后需要跟一个vit fwd，这里在llm fwd之前提前接收vit fwd
            if (
                i_group == 0
                and i >= num_forward_end_backward_start - 1
                and vit_fwd_num > 0
            ):
                set_vpp_rank(0)
                for future_rfg, future_nr, future_pr in zip(receive_forward_group, next_rank, prev_rank):
                    if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                        # 首PP不需要接收，在队列中用None占位
                        vit_input_tensor_queue.append([None])
                        vit_reqs_queue.append([])
                    else:
                        recv_tensor_shapes = vit_recv_fwd_tensor_shapes.pop(0)
                        vit_input_tensor_tmp, vit_reqs_list = recv_forward_with_stream(
                            recv_tensor_shapes,
                            config,
                            group=future_rfg,
                            next_rank=future_nr,
                            prev_rank=future_pr,
                            wait_on_reqs=False,
                        )
                        vit_input_tensor_queue.append(vit_input_tensor_tmp)
                        vit_reqs_queue.append(vit_reqs_list)
                set_vpp_rank(1)

            # 传算掩盖：提前接收下一轮llm fwd (如果不是首PP，最后一轮还需要提前接收2f2b阶段的第一轮llm fwd)
            if (
                i_group == 0
                and llm_fwd_num > 1
                and (not last_iteration or (last_iteration and not parallel_state.is_pipeline_first_stage(ignore_virtual=True)))
            ):
                for future_rfg, future_nr, future_pr in zip(receive_forward_group, next_rank, prev_rank):
                    if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                        recv_tensor_shapes = vit_recv_fwd_tensor_shapes.pop(0)
                    else:
                        recv_tensor_shapes = recv_forward_tensor_shapes.pop(0)
                    input_tensor_tmp, reqs_list = recv_forward_with_stream(
                        recv_tensor_shapes,
                        config,
                        group=future_rfg,
                        next_rank=future_nr,
                        prev_rank=future_pr,
                        is_end_stage=True,
                        wait_on_reqs=False,
                    )
                    input_tensor_queue.append(input_tensor_tmp)
                    reqs_queue.append(reqs_list)

            # 传算掩盖：首PP提前接收febs阶段的第一轮llm fwd (尾层)
            if (
                i_group == 0
                and last_iteration
                and parallel_state.is_pipeline_first_stage(ignore_virtual=True)
                and num_forward_end_backward_start > 0
            ):
                set_vpp_rank(2)
                for future_rfg, future_nr, future_pr in zip(receive_forward_group, next_rank, prev_rank):
                    recv_tensor_shapes = recv_forward_tensor_shapes.pop(0)
                    vdp_input_tensor_tmp, vdp_reqs_list = recv_forward_with_stream(
                        recv_tensor_shapes,
                        config,
                        group=future_rfg,
                        next_rank=future_nr,
                        prev_rank=future_pr,
                        is_end_stage=True,
                        wait_on_reqs=False
                    )
                    vdp_input_tensor_queue.append(vdp_input_tensor_tmp)
                    vdp_reqs_queue.append(vdp_reqs_list)
                set_vpp_rank(1)

            # llm fwd
            this_iterator = None
            this_model = model[1]
            input_tensor = input_tensor_queue.pop(0)
            output_tensor, num_tokens = forward_step(
                forward_step_func,
                this_iterator,
                this_model,
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                collect_non_loss_data,
                checkpoint_activations_microbatch,
                check_first_val_step(first_val_step, forward_only, i == 0),
                current_microbatch=i,
                encoder_decoder_xattn=encoder_decoder_xattn,
                batch=all_batchs[1].pop(0)
            )
            llm_fwd_num -= 1
            total_num_tokens += num_tokens
            # llm send fwd
            send_forward_with_stream(
                output_tensor,
                send_tensor_shapes,
                config,
                group=sfg,
                next_rank=nr,
                prev_rank=pr,
            )

            if not forward_only:
                input_tensors.append(input_tensor)
                output_tensors.append(output_tensor)
                deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

        # 第num_forward_end_backward_start轮llm fwd之后需要跟一个vit fwd
        if i >= num_forward_end_backward_start - 1 and vit_fwd_num > 0:
            for sfg, nr, pr in zip(send_forward_group, next_rank, prev_rank):
                set_vpp_rank(0)
                # 传算掩盖：阻塞式接收vit fwd
                reqs_list = vit_reqs_queue.pop(0)
                wait_helper(vit_reqs_list)

                # vit fwd
                this_iterator = None
                this_model = model[0]
                input_tensor = vit_input_tensor_queue.pop(0)
                output_tensor, num_tokens = forward_step(
                    forward_step_func,
                    this_iterator,
                    this_model,
                    num_microbatches,
                    input_tensor,
                    forward_data_store,
                    config,
                    collect_non_loss_data,
                    checkpoint_activations_microbatch,
                    check_first_val_step(first_val_step, forward_only, i == 0),
                    current_microbatch=i,
                    encoder_decoder_xattn=encoder_decoder_xattn,
                    batch=all_batchs[0].pop(0)
                )
                vit_fwd_num -= 1
                total_num_tokens += num_tokens

                # vit send fwd
                send_forward_with_stream(
                    output_tensor,
                    send_tensor_shapes,
                    config,
                    group=sfg,
                    next_rank=nr,
                    prev_rank=pr,
                )

                if not forward_only:
                    vit_input_tensors.append(input_tensor)
                    vit_output_tensors.append(output_tensor)
                    deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

    # Run forward-end-backward-start at end stage for PP0
    set_vpp_rank(2)
    if parallel_state.is_pipeline_first_stage(ignore_virtual=True):

        for i in range(num_forward_end_backward_start):
            if last_stage_febs_num == 0:
                break

            last_iteration = i == (num_forward_end_backward_start - 1)

            for i_group, sbg, nr, pr in zip(group_iter, send_backward_group, next_rank, prev_rank):

                # 传算掩盖：阻塞式接收llm fwd (尾层)
                vdp_reqs_list = vdp_reqs_queue.pop(0)
                wait_helper(vdp_reqs_list)

                # 传算掩盖：首PP提前接收下一轮llm fwd (尾层)
                if i_group == 0 and not last_iteration:
                    for future_rfg, future_nr, future_pr in zip(receive_forward_group, next_rank, prev_rank):
                        recv_tensor_shapes = recv_forward_tensor_shapes.pop(0)
                        vdp_input_tensor_tmp, vdp_reqs_list = recv_forward_with_stream(
                            recv_tensor_shapes,
                            config,
                            group=future_rfg,
                            next_rank=future_nr,
                            prev_rank=future_pr,
                            is_end_stage=True,
                            wait_on_reqs=False,
                        )
                        vdp_input_tensor_queue.append(vdp_input_tensor_tmp)
                        vdp_reqs_queue.append(vdp_reqs_list)

                # llm fwd (尾层)
                this_iterator = None
                this_model = model[2]
                input_tensor_end = vdp_input_tensor_queue.pop(0)
                output_tensor_end, num_tokens = forward_step(
                    forward_step_func,
                    this_iterator,
                    this_model,
                    num_microbatches,
                    input_tensor_end,
                    forward_data_store,
                    config,
                    collect_non_loss_data,
                    checkpoint_activations_microbatch,
                    check_first_val_step(first_val_step, forward_only, i == 0),
                    current_microbatch=i,
                    encoder_decoder_xattn=encoder_decoder_xattn,
                    is_end_stage=True,
                    batch=all_batchs[2].pop(0)
                )
                last_stage_febs_num -= 1
                total_num_tokens += num_tokens

                if not forward_only:
                    output_tensor_grad_end = [None] * len(recv_tensor_shapes)

                    deallocate_output_tensor(output_tensor_end[0], config.deallocate_pipeline_outputs)
                    # llm bwd (尾层)
                    input_tensor_grad_end = backward_step(
                        input_tensor_end, output_tensor_end, output_tensor_grad_end, model_type, config
                    )

                    if last_iteration:
                        input_tensor_end = None
                    # send llm bwd (尾层)
                    send_backward_with_stream(
                        input_tensor_grad_end,
                        send_tensor_shapes,
                        config,
                        group=sbg,
                        next_rank=nr,
                        prev_rank=pr,
                        is_end_stage=True,
                    )

    # Run n-F-n-B in steady state
    output_tensor_grad_tmp = None
    vit_output_tensor_grad_tmp = None
    output_tensor_grad_queue = []
    vit_output_tensor_grad_queue = []

    for i in range(num_microbatches):
        last_iteration = i == num_microbatches - 1
        if i == 0:
            # 首PP这里用非掩盖recv llm fwd，不影响性能
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True) and llm_fwd_num > 0:
                set_vpp_rank(1)
                for rfg, nr, pr in zip(receive_forward_group, next_rank, prev_rank):
                    recv_tensor_shapes = vit_recv_fwd_tensor_shapes.pop(0)
                    input_tensor_tmp = recv_forward_with_stream(
                        recv_tensor_shapes,
                        config,
                        group=rfg,
                        next_rank=nr,
                        prev_rank=pr,
                        is_end_stage=True
                    )
                    input_tensor_queue.append(input_tensor_tmp)
                    reqs_queue.append([])

            # 传算掩盖：提前接收vit fwd (中间层)
            if vit_fwd_num > 0 and i + num_warmup_microbatches >= num_forward_end_backward_start - 1:
                set_vpp_rank(0)
                for rfg, nr, pr in zip(receive_forward_group, next_rank, prev_rank):
                    if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                        # 首PP不需要接收vit fwd，在队列中用None占位
                        vit_input_tensor_queue.append([None])
                        vit_reqs_queue.append([])
                    else:
                        recv_tensor_shapes = vit_recv_fwd_tensor_shapes.pop(0)
                        vit_input_tensor_tmp, vit_reqs_list = recv_forward_with_stream(
                            recv_tensor_shapes,
                            config,
                            group=rfg,
                            next_rank=nr,
                            prev_rank=pr,
                            wait_on_reqs=False
                        )
                        vit_input_tensor_queue.append(vit_input_tensor_tmp)
                        vit_reqs_queue.append(vit_reqs_list)

            # 传算掩盖：首PP提前接收llm fwd (尾层)
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True) and last_stage_febs_num > 0:
                set_vpp_rank(2)
                for rfg, nr, pr in zip(receive_forward_group, next_rank, prev_rank):
                    recv_tensor_shapes = recv_forward_tensor_shapes.pop(0)
                    vdp_input_tensor_tmp, vdp_reqs_list = recv_forward_with_stream(
                        recv_tensor_shapes,
                        config,
                        group=rfg,
                        next_rank=nr,
                        prev_rank=pr,
                        is_end_stage=True,
                        wait_on_reqs=False
                    )
                    vdp_input_tensor_queue.append(vdp_input_tensor_tmp)
                    vdp_reqs_queue.append(vdp_reqs_list)

        # llm fwd
        if llm_fwd_num > 0:
            set_vpp_rank(1)
            for sfg, nr, pr in zip(send_forward_group, next_rank, prev_rank):
                # 传算掩盖：阻塞式接收llm fwd
                reqs_list = reqs_queue.pop(0)
                wait_helper(reqs_list)

                # llm fwd
                this_iterator = None
                this_model = model[1]
                input_tensor = input_tensor_queue.pop(0)
                output_tensor, num_tokens = forward_step(
                    forward_step_func,
                    this_iterator,
                    this_model,
                    num_microbatches,
                    input_tensor,
                    forward_data_store,
                    config,
                    collect_non_loss_data,
                    checkpoint_activations_microbatch,
                    check_first_val_step(
                        first_val_step, forward_only, (llm_fwd_num == mbn)
                        ),
                    current_microbatch=i + num_warmup_microbatches,
                    encoder_decoder_xattn=encoder_decoder_xattn,
                    batch=all_batchs[1].pop(0)
                )
                llm_fwd_num -= 1
                total_num_tokens += num_tokens

                # llm send fwd
                send_forward_with_stream(
                    output_tensor,
                    send_tensor_shapes,
                    config,
                    group=sfg,
                    next_rank=nr,
                    prev_rank=pr,
                )

                if not forward_only:
                    input_tensors.append(input_tensor)
                    output_tensors.append(output_tensor)
                    deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

        for i_group, sfg, nr, pr in zip(group_iter, send_forward_group, next_rank, prev_rank):
            # 传算掩盖：阻塞式接收vit fwd
            if vit_fwd_num > 0 and i + num_warmup_microbatches >= num_forward_end_backward_start - 1:
                set_vpp_rank(0)
                vit_reqs_list = vit_reqs_queue.pop(0)
                wait_helper(vit_reqs_list)

            # 传算掩盖：提前接收llm bwd
            if i_group == 0 and llm_bwd_num > 0:
                set_vpp_rank(1)
                for future_rbg, future_nr, future_pr in zip(receive_backward_group, next_rank, prev_rank):
                    recv_tensor_shapes = recv_backward_tensor_shapes.pop(0)
                    output_tensor_grad_tmp, reqs_list = recv_backward_with_stream(
                        recv_tensor_shapes,
                        config,
                        group=future_rbg,
                        next_rank=future_nr,
                        prev_rank=future_pr,
                        wait_on_reqs=False,
                    )
                    output_tensor_grad_queue.append(output_tensor_grad_tmp)
                    reqs_queue.append(reqs_list)

            # 传算掩盖：提前接收vit bwd
            if i_group == 0 and vit_bwd_num > 0 and i >= num_forward_end_backward_start and not forward_only:
                set_vpp_rank(0)
                for future_rbg, future_nr, future_pr in zip(receive_backward_group, next_rank, prev_rank):
                    recv_tensor_shapes = vit_recv_bwd_tensor_shapes.pop(0)
                    vit_output_tensor_grad_tmp, vit_reqs_list = recv_backward_with_stream(
                        recv_tensor_shapes,
                        config,
                        group=future_rbg,
                        next_rank=future_nr,
                        prev_rank=future_pr,
                        wait_on_reqs=False
                    )
                    vit_output_tensor_grad_queue.append(vit_output_tensor_grad_tmp)
                    vit_reqs_queue.append(vit_reqs_list)

            # vit fwd
            vit_output_tensor = None
            if vit_fwd_num > 0 and i + num_warmup_microbatches >= num_forward_end_backward_start - 1:
                set_vpp_rank(0)

                # vit fwd
                this_iterator = None
                this_model = model[0]
                input_tensor = vit_input_tensor_queue.pop(0)
                vit_output_tensor, num_tokens = forward_step(
                    forward_step_func,
                    this_iterator,
                    this_model,
                    num_microbatches,
                    input_tensor,
                    forward_data_store,
                    config,
                    collect_non_loss_data,
                    checkpoint_activations_microbatch,
                    check_first_val_step(
                        first_val_step, forward_only, (vit_fwd_num == mbn)
                        ),
                    current_microbatch=i + num_warmup_microbatches,
                    encoder_decoder_xattn=encoder_decoder_xattn,
                    batch=all_batchs[0].pop(0)
                )
                vit_fwd_num -= 1
                total_num_tokens += num_tokens

                # vit send fwd
                if parallel_state.get_pipeline_model_parallel_world_size() > 2 or not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    send_forward_with_stream(
                        vit_output_tensor,
                        send_tensor_shapes,
                        config,
                        group=sfg,
                        next_rank=nr,
                        prev_rank=pr,
                    )
                    if not forward_only:
                        vit_input_tensors.append(input_tensor)
                        vit_output_tensors.append(vit_output_tensor)
                        deallocate_output_tensor(vit_output_tensor[0], config.deallocate_pipeline_outputs)

        # llm 1f1b (尾层)
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True) and last_stage_febs_num > 0:
            for sbg, nr, pr in zip(send_backward_group, next_rank, prev_rank):
                set_vpp_rank(2)
                # 传算掩盖：阻塞式接收llm fwd (尾层)
                vdp_reqs_list = vdp_reqs_queue.pop(0)
                wait_helper(vdp_reqs_list)

                # llm fwd (尾层)
                this_iterator = None
                this_model = model[2]
                input_tensor_end = vdp_input_tensor_queue.pop(0)
                output_tensor_end, num_tokens = forward_step(
                    forward_step_func,
                    this_iterator,
                    this_model,
                    num_microbatches,
                    input_tensor_end,
                    forward_data_store,
                    config,
                    collect_non_loss_data,
                    checkpoint_activations_microbatch,
                    check_first_val_step(
                        first_val_step, forward_only, (last_stage_febs_num == mbn)
                        ),
                    current_microbatch=i + num_warmup_microbatches,
                    encoder_decoder_xattn=encoder_decoder_xattn,
                    is_end_stage=True,
                    batch=all_batchs[2].pop(0)
                )
                last_stage_febs_num -= 1
                total_num_tokens += num_tokens

                if not forward_only:
                    deallocate_output_tensor(output_tensor_end[0], config.deallocate_pipeline_outputs)
                    output_tensor_grad_end = [None] * len(recv_tensor_shapes)

                    # llm bwd (尾层)
                    input_tensor_grad_end = backward_step(
                        input_tensor_end, output_tensor_end, output_tensor_grad_end, model_type, config
                    )

                    # llm send bwd (尾层)
                    send_backward_with_stream(
                        input_tensor_grad_end,
                        send_tensor_shapes,
                        config,
                        group=sbg,
                        next_rank=nr,
                        prev_rank=pr,
                        is_end_stage=True,
                    )

        for i_group, sbg, nr, pr in zip(group_iter, send_backward_group, next_rank, prev_rank):
            # 传算掩盖：阻塞式接收llm bwd
            if llm_bwd_num > 0:
                set_vpp_rank(1)
                reqs_list = reqs_queue.pop(0)
                wait_helper(reqs_list)

            # 传算掩盖：提前接收llm fwd
            if i_group == 0 and not last_iteration and llm_fwd_num > 0:
                set_vpp_rank(1)
                for future_rfg, future_nr, future_pr in zip(receive_forward_group, next_rank, prev_rank):
                    if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                        recv_tensor_shapes = vit_recv_fwd_tensor_shapes.pop(0)
                    else:
                        recv_tensor_shapes = recv_forward_tensor_shapes.pop(0)
                    input_tensor_tmp, reqs_list = recv_forward_with_stream(
                        recv_tensor_shapes,
                        config,
                        group=future_rfg,
                        next_rank=future_nr,
                        prev_rank=future_pr,
                        is_end_stage=True,
                        wait_on_reqs=False,
                    )
                    input_tensor_queue.append(input_tensor_tmp)
                    reqs_queue.append(reqs_list)

            # llm bwd
            if llm_bwd_num > 0:
                set_vpp_rank(1)
                input_tensor = input_tensors.pop(0)
                output_tensor = output_tensors.pop(0)

                # llm bwd
                output_tensor_grad = output_tensor_grad_queue.pop(0)
                input_tensor_grad = backward_step(
                    input_tensor, output_tensor, output_tensor_grad, model_type, config
                )
                llm_bwd_num -= 1

                # send llm bwd
                send_backward_with_stream(
                    input_tensor_grad,
                    send_tensor_shapes,
                    config,
                    group=sbg,
                    next_rank=nr,
                    prev_rank=pr,
                    is_end_stage=True,
                )

        for i_group, sbg, nr, pr in zip(group_iter, send_backward_group, next_rank, prev_rank):
            # 传算掩盖：阻塞式接收vit bwd
            if vit_bwd_num > 0 and i >= num_forward_end_backward_start and not forward_only:
                set_vpp_rank(0)
                vit_reqs_list = vit_reqs_queue.pop(0)
                wait_helper(vit_reqs_list)

            if i_group == 0 and not last_iteration:
                # 传算掩盖：非首PP提前接收vit fwd (中间层)
                if vit_fwd_num > 0 and i + num_warmup_microbatches + 1 >= num_forward_end_backward_start - 1:
                    set_vpp_rank(0)
                    for future_rfg, future_nr, future_pr in zip(receive_forward_group, next_rank, prev_rank):
                        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                            # 首PP不需要接收，在队列中用None占位
                            vit_input_tensor_queue.append([None])
                            vit_reqs_queue.append([])
                        else:
                            recv_tensor_shapes = vit_recv_fwd_tensor_shapes.pop(0)
                            vit_input_tensor_tmp, vit_reqs_list = recv_forward_with_stream(
                                recv_tensor_shapes,
                                config,
                                group=future_rfg,
                                next_rank=future_nr,
                                prev_rank=future_pr,
                                wait_on_reqs=False,
                            )
                            vit_input_tensor_queue.append(vit_input_tensor_tmp)
                            vit_reqs_queue.append(vit_reqs_list)


                # 传算掩盖：首PP提前接收llm fwd (尾层)
                if parallel_state.is_pipeline_first_stage(ignore_virtual=True) and last_stage_febs_num > 0:
                    set_vpp_rank(2)
                    for future_rfg, future_nr, future_pr in zip(receive_forward_group, next_rank, prev_rank):
                        recv_tensor_shapes = recv_forward_tensor_shapes.pop(0)
                        vdp_input_tensor_tmp, vdp_reqs_list = recv_forward_with_stream(
                            recv_tensor_shapes,
                            config,
                            group=future_rfg,
                            next_rank=future_nr,
                            prev_rank=future_pr,
                            is_end_stage=True,
                            wait_on_reqs=False,
                        )
                        vdp_input_tensor_queue.append(vdp_input_tensor_tmp)
                        vdp_reqs_queue.append(vdp_reqs_list)

            elif i_group == 0 and vit_bwd_num > 0 and last_iteration:
                set_vpp_rank(0)
                # 传算掩盖：最后一轮提前接收cooldown的vit bwd
                for future_rbg, future_nr, future_pr in zip(receive_backward_group, next_rank, prev_rank):
                    recv_tensor_shapes = vit_recv_bwd_tensor_shapes.pop(0)
                    vit_output_tensor_grad_tmp, vit_reqs_list = recv_backward_with_stream(
                        recv_tensor_shapes,
                        config,
                        group=future_rbg,
                        next_rank=future_nr,
                        prev_rank=future_pr,
                        wait_on_reqs=False,
                    )
                    vit_output_tensor_grad_queue.append(vit_output_tensor_grad_tmp)
                    vit_reqs_queue.append(vit_reqs_list)

            # vit bwd
            if vit_bwd_num > 0 and i >= num_forward_end_backward_start and not forward_only:
                set_vpp_rank(0)
                # 如果这是最后一次bwd，则开启梯度同步
                if vit_bwd_num == 1:
                    if config.grad_sync_func is None or rank == 0:
                        enable_grad_sync()

                input_tensor = vit_input_tensors.pop(0)
                output_tensor = vit_output_tensors.pop(0)

                # vit bwd
                output_tensor_grad = vit_output_tensor_grad_queue.pop(0)
                input_tensor_grad = backward_step(
                    input_tensor, output_tensor, output_tensor_grad, model_type, config
                )
                vit_bwd_num -= 1

                # vit send bwd
                send_backward_with_stream(
                    input_tensor_grad,
                    send_tensor_shapes,
                    config,
                    group=sbg,
                    next_rank=nr,
                    prev_rank=pr,
                )

        # vit send fwd (pp==2 && is_last_stage)
        if (
            vit_output_tensor is not None
            and parallel_state.get_pipeline_model_parallel_world_size() == 2
            and parallel_state.is_pipeline_last_stage(ignore_virtual=True)
        ):
            set_vpp_rank(0)
            send_forward_with_stream(
                vit_output_tensor,
                send_tensor_shapes,
                config,
                group=send_forward_group,
            )
            if not forward_only:
                vit_input_tensors.append(input_tensor)
                vit_output_tensors.append(vit_output_tensor)
                deallocate_output_tensor(vit_output_tensor[0], config.deallocate_pipeline_outputs)

    # Run vit cooldown backward passes
    set_vpp_rank(0)
    if not forward_only:
        while vit_bwd_num > 0:
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                last_iteration = vit_bwd_num == get_vdp_size()
            else:
                last_iteration = vit_bwd_num == 1

            # 最后一轮开启梯度同步
            if last_iteration:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            for i_group, sbg, nr, pr in zip(group_iter, send_backward_group, next_rank, prev_rank):
                # 传算掩盖：阻塞式接收vit bwd
                vit_reqs_list = vit_reqs_queue.pop(0)
                wait_helper(vit_reqs_list)

                # 传算掩盖：提前接收下一轮vit bwd
                if i_group == 0 and not last_iteration:
                    for future_rbg, future_nr, future_pr in zip(receive_backward_group, next_rank, prev_rank):
                        recv_tensor_shapes = vit_recv_bwd_tensor_shapes.pop(0)
                        vit_output_tensor_grad_tmp, vit_reqs_list = recv_backward_with_stream(
                            recv_tensor_shapes,
                            config,
                            group=future_rbg,
                            next_rank=future_nr,
                            prev_rank=future_pr,
                            wait_on_reqs=False,
                        )
                        vit_output_tensor_grad_queue.append(vit_output_tensor_grad_tmp)
                        vit_reqs_queue.append(vit_reqs_list)

                input_tensor = vit_input_tensors.pop(0)
                output_tensor = vit_output_tensors.pop(0)

                # vit bwd
                output_tensor_grad = vit_output_tensor_grad_queue.pop(0)
                input_tensor_grad = backward_step(
                    input_tensor, output_tensor, output_tensor_grad, model_type, config
                )
                vit_bwd_num -= 1

                # send vit bwd
                send_backward_with_stream(
                    input_tensor_grad,
                    send_tensor_shapes,
                    config,
                    group=sbg,
                    next_rank=nr,
                    prev_rank=pr,
                )

        # Launch any remaining grad reductions.
        if no_sync_context is not None:
            enable_grad_sync()
            if config.grad_sync_func is not None:
                for this_model in model:
                    config.grad_sync_func(this_model.parameters())

    if config.finalize_model_grads_func is not None and not forward_only:

        # If defer_embedding_wgrad_compute is enabled we need to do the
        # weight gradient GEMM's here.
        finish_embedding_wgrad_compute(config, embedding_module)

        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        this_model = model if parallel_state.is_pipeline_first_stage(ignore_virtual=True) else [model[0]]
        config.finalize_model_grads_func(
            this_model, total_num_tokens if config.calculate_per_token_loss else None
        )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    # Drain any remaining VTP async sends before returning.
    _drain_vtp_sends()

    if hasattr(config, 'enable_cuda_graph') and config.enable_cuda_graph:
        create_cudagraphs()

    return forward_data_store
