# Sorting algorithm for data balance
import torch
from google.protobuf.internal.wire_format import INT64_MAX


def post_global_balancing_greedy_without_pad(
        global_data_length: torch.Tensor,
        num_replicas: int,
        image_encoder_dp: int = None,
        max_batch_capacity: int = INT64_MAX
) -> list[list[tuple[int, int, torch.Tensor]]]:
    if image_encoder_dp is None:
        image_encoder_dp = num_replicas
    max_batch_capacity = max_batch_capacity * num_replicas // image_encoder_dp

    per_dp_size = len(global_data_length) // num_replicas
    num_groups = len(global_data_length) // max_batch_capacity

    sort_indice = torch.argsort(global_data_length[:, 2], descending=True)
    global_data_length = global_data_length[sort_indice]
    dp_group_total_length = torch.stack(
        [
            torch.arange(num_groups, device='cpu'),
            torch.zeros(num_groups, dtype=torch.long, device='cpu')],
        dim=1
    )
    lengths_per_sequence = (global_data_length[:, 2] ** 2).cpu()
    balanced_image_dp_batch = torch.empty(
        (num_groups, max_batch_capacity, global_data_length.shape[-1]),
        dtype=global_data_length.dtype, device=global_data_length.device
    )

    balanced_image_dp_batch_idxs = [1] * num_groups
    group_lengths_range = torch.arange(len(dp_group_total_length))

    # The bucket has no data; add items in sequence.
    balanced_image_dp_batch[:, 0] = global_data_length[: num_groups]
    dp_group_total_length[:, 1] = lengths_per_sequence[: num_groups]

    # prior queue strategy
    if max_batch_capacity > 1:
        for i, sequence_length in enumerate(global_data_length[num_groups:]):
            target_index = dp_group_total_length[:, 1].argmin()
            target_dp_group = dp_group_total_length[target_index][0]
            balanced_image_dp_batch[target_dp_group, balanced_image_dp_batch_idxs[target_dp_group]] = sequence_length

            balanced_image_dp_batch_idxs[target_dp_group] += 1
            if balanced_image_dp_batch_idxs[target_dp_group] >= max_batch_capacity:
                mask = group_lengths_range != target_index
                dp_group_total_length = dp_group_total_length[mask]
                group_lengths_range = group_lengths_range[: -1]
            else:
                dp_group_total_length[target_index][1] += lengths_per_sequence[i + num_groups]
    balanced_batchs = balanced_image_dp_batch.flatten(0, 1).split([per_dp_size] * num_replicas)

    return balanced_batchs


def post_mbs_balancing_greedy_without_pad(
        global_data_length: torch.Tensor,
        num_replicas: int,
        **kwargs
) -> list[list[tuple[int, int, torch.Tensor]]]:
    sort_indice = torch.argsort(global_data_length[:, 2], descending=True)
    global_data_length = global_data_length[sort_indice]
    lengths_per_sequence = (global_data_length[:, 2] ** 2).cpu()

    dp_group_total_length = torch.empty(num_replicas, dtype=torch.long)
    dp_group_total_length[:] = lengths_per_sequence[: num_replicas]
    balanced_image_dp_batch = [[global_data_length[i]] for i in range(num_replicas)]

    for i, sequence_lentgh in enumerate(global_data_length[num_replicas:]):
        target_dp_group = dp_group_total_length.argmin()
        balanced_image_dp_batch[target_dp_group].extend([sequence_lentgh])
        dp_group_total_length[target_dp_group] += lengths_per_sequence[i + num_replicas]

    return balanced_image_dp_batch


SORTING_ALGO_FUNC = {
    'post_global_balancing_greedy_without_pad': post_global_balancing_greedy_without_pad,
    'post_mbs_balancing_greedy_without_pad': post_mbs_balancing_greedy_without_pad,
}
