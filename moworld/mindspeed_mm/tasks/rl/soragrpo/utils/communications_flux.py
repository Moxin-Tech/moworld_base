# Copyright (c) [2025] [FastVideo Team]
# Copyright (c) [2025] [ByteDance Ltd. and/or its affiliates.]
# SPDX-License-Identifier: [Apache License 2.0]
#
# This file has been modified by [ByteDance Ltd. and/or its affiliates.] in 2025.
#
# Original file was released under [Apache License 2.0], with the full license text
# available at [https://github.com/hao-ai-lab/FastVideo/blob/main/LICENSE].
#
# This modified file is released under the same license.

import torch
import torch.distributed as dist

from mindspeed_mm.tasks.rl.soragrpo.utils.parallel_states import nccl_info


def _all_to_all(
        input_: torch.Tensor,
        world_size: int,
        group: dist.ProcessGroup,
        scatter_dim: int,
        gather_dim: int,
):
    input_list = [
        t.contiguous()
        for t in torch.tensor_split(input_, world_size, scatter_dim)
    ]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


class _AllToAll(torch.autograd.Function):
    """All-to-all communication.

    Args:
        input_: input matrix
        process_group: communication group
        scatter_dim: scatter dimension
        gather_dim: gather dimension
    """

    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim):
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.world_size = dist.get_world_size(process_group)
        output = _all_to_all(input_, ctx.world_size, process_group,
                             scatter_dim, gather_dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = _all_to_all(
            grad_output,
            ctx.world_size,
            ctx.process_group,
            ctx.gather_dim,
            ctx.scatter_dim,
        )
        return (
            grad_output,
            None,
            None,
            None,
        )


def all_to_all(
        input_: torch.Tensor,
        scatter_dim: int = 2,
        gather_dim: int = 1,
):
    return _AllToAll.apply(input_, nccl_info.group, scatter_dim, gather_dim)


def prepare_sequence_parallel_data(
        encoder_hidden_states, pooled_prompt_embeds, text_ids, caption
):
    if nccl_info.sp_size == 1:
        return (
            encoder_hidden_states,
            pooled_prompt_embeds,
            text_ids,
            caption,
        )

    def prepare(
            encoder_hidden_states, pooled_prompt_embeds, text_ids, caption
    ):
        encoder_hidden_states = all_to_all(
            encoder_hidden_states, scatter_dim=1, gather_dim=0
        )
        pooled_prompt_embeds = all_to_all(
            pooled_prompt_embeds, scatter_dim=1, gather_dim=0
        )
        text_ids = all_to_all(text_ids, scatter_dim=1, gather_dim=0)
        return (
            encoder_hidden_states,
            pooled_prompt_embeds,
            text_ids,
            caption,
        )

    sp_size = nccl_info.sp_size

    (
        encoder_hidden_states,
        pooled_prompt_embeds,
        text_ids,
        caption,
    ) = prepare(
        encoder_hidden_states.repeat(1, sp_size, 1),
        pooled_prompt_embeds.repeat(1, sp_size, 1, 1),
        text_ids.repeat(1, sp_size),
        caption,
    )

    return encoder_hidden_states, pooled_prompt_embeds, text_ids, caption


def sp_parallel_dataloader_wrapper(
        dataloader, device, train_batch_size, sp_size, train_sp_batch_size
):
    while True:
        for data_item in dataloader:
            encoder_hidden_states, pooled_prompt_embeds, text_ids, caption = data_item
            encoder_hidden_states = encoder_hidden_states.to(device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(device)
            text_ids = text_ids.to(device)
            frame = 19
            if frame == 1:
                yield encoder_hidden_states, pooled_prompt_embeds, text_ids, caption
            else:
                encoder_hidden_states, pooled_prompt_embeds, text_ids, caption = prepare_sequence_parallel_data(
                    encoder_hidden_states, pooled_prompt_embeds, text_ids, caption
                )
                if not train_batch_size * sp_size >= train_sp_batch_size:
                    raise AssertionError("train_batch_size * sp_size should be greater than train_sp_batch_size")
                for i in range(train_batch_size * sp_size // train_sp_batch_size):
                    st_idx = i * train_sp_batch_size
                    ed_idx = (i + 1) * train_sp_batch_size
                    encoder_hidden_states = encoder_hidden_states[st_idx:ed_idx]
                    pooled_prompt_embeds = pooled_prompt_embeds[st_idx:ed_idx]
                    text_ids = text_ids[st_idx:ed_idx]
                    yield (
                        encoder_hidden_states,
                        pooled_prompt_embeds,
                        text_ids,
                        caption,
                    )
