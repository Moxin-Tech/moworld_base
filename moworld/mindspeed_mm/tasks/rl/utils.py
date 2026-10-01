# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
from typing import Tuple
import json
import math

import torch

from megatron.core import mpu


def read_json_file(filename):
    """Reade JSON File"""
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def find_probability(score, data):
    bin_index = math.floor(score / data['bin_width'])
    lower = bin_index * data['bin_width']
    upper = lower + data['bin_width']
    key = f"{lower}-{upper}"
    if key in data:
        return data[key] / data['total_num']  # Probability
    return 0.0  # If score is out of bounds


def get_attr_from_wrapped_model(model, target_attr):
    def recursive_search(module):
        if hasattr(module, target_attr):
            return getattr(module, target_attr)

        for _, child in getattr(module, '_modules').items():
            result = recursive_search(child)
            if result is not None:
                return result

        return None

    return [recursive_search(model)]


def vocab_parallel_log_softmax(logits):
    """
    Compute log softmax values for each sets of scores in vocab parallel.

    Args:
        logits (Tensor): Input logits.

    Returns:
        Tensor: Log softmax values.
    """
    # Step 1: Compute the local max value for numerical stability
    z_max = logits.max(dim=-1, keepdim=True).values

    # Step 2: Perform all-reduce to get the global max across all processes
    torch.distributed.all_reduce(
        z_max,
        op=torch.distributed.ReduceOp.MAX,
        group=mpu.get_tensor_model_parallel_group()
    )

    # Step 3: Compute the log(exp(x - z_max)) for local logits
    local_exp = torch.exp(logits - z_max)

    # Step 4: Compute local sum of exp(x - z_max)
    local_sum_exp = local_exp.sum(dim=-1, keepdim=True)

    # Step 5: Perform all-reduce to get the global sum of exp(x - z_max) across all processes
    torch.distributed.all_reduce(
        local_sum_exp,
        op=torch.distributed.ReduceOp.SUM,
        group=mpu.get_tensor_model_parallel_group()
    )

    # Step 6: Compute the log of the global sum of exp(x - z_max)
    log_sum_exp = local_sum_exp.log()

    # Step 7: Compute and return the log softmax values
    return logits - z_max - log_sum_exp


def compute_log_probs(
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index=-100,
        per_token=False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the log probabilities of the given labels under the given logits.

    In the tensor parallelism case, it takes into account the vocab parallelism and
    performs the necessary adjustments to the labels and logits.

    Args:
        logits: The logits tensor.
        labels: The label tensor.
        ignore_index: The index to ignore for masking in input_ids.
        per_token: Set to True if you want to get per-token log probabilities.

    Returns:
        A tuple containing the log probabilities and the valid length.
    """
    if mpu.get_tensor_model_parallel_world_size() > 1:
        tp_vocab_size = logits.size(2)

        labels -= mpu.get_tensor_model_parallel_rank() * tp_vocab_size
        labels = labels.masked_fill(torch.logical_or(labels < 0, labels >= tp_vocab_size), 0)
        loss_mask = labels != 0

        per_token_log_probs = torch.gather(
            vocab_parallel_log_softmax(logits), dim=2, index=labels.unsqueeze(2)).squeeze(2) * loss_mask

        all_log_probs = per_token_log_probs.sum(-1)
        valid_length = loss_mask.sum(-1)

        torch.distributed.all_reduce(
            all_log_probs,
            op=torch.distributed.ReduceOp.SUM,
            group=mpu.get_tensor_model_parallel_group()
        )

        torch.distributed.all_reduce(
            valid_length,
            op=torch.distributed.ReduceOp.SUM,
            group=mpu.get_tensor_model_parallel_group()
        )

        if per_token:
            torch.distributed.all_reduce(
                per_token_log_probs,
                op=torch.distributed.ReduceOp.SUM,
                group=mpu.get_tensor_model_parallel_group()
            )

    else:
        label_pad_token_id = ignore_index
        loss_mask = labels != label_pad_token_id
        labels[labels == label_pad_token_id] = 0  # dummy token
        per_token_log_probs = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        per_token_log_probs = per_token_log_probs * loss_mask
        all_log_probs = per_token_log_probs.sum(-1)
        valid_length = loss_mask.sum(-1)


    return all_log_probs, valid_length, per_token_log_probs
