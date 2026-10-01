# Copyright 2025 HuggingFace Inc. team. All rights reserved.

from typing import Callable, Optional, Union
import torch
from packaging import version
import transformers

if version.parse(transformers.__version__) >= version.parse("4.54.0.dev0"):
    from transformers.masking_utils import (
        causal_mask_function,
        _ignore_causal_mask_sdpa,
        prepare_padding_mask,
        _is_torch_greater_or_equal_than_2_5,
    )

    def sdpa_mask_older_torch(
        batch_size: int,
        cache_position: torch.Tensor,
        kv_length: int,
        kv_offset: int = 0,
        mask_function: Callable = causal_mask_function,
        attention_mask: Optional[torch.Tensor] = None,
        local_size: Optional[int] = None,
        allow_is_causal_skip: bool = True,
        allow_torch_fix: bool = True,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """
        NOTE: This function is only used when torch version is torch<2.5 - see `sdpa_mask_recent_torch` otherwise.

        Create a 4D boolean mask of shape `(batch_size, 1, query_length, kv_length)` where a value of True indicates that
        the element should take part in the attention computation, and False that it should not.
        If `allow_torch_fix=True` (the default), rows corresponding to query tokens that do not attend
        to any other tokens (due to padding) will be fully attended to instead, in order to avoid `nan` propagation (this does
        not change the final result).

        Args:
            batch_size (`int`):
                The batch size of the input sequence.
            cache_position (`torch.Tensor`):
                A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
            kv_length (`int`):
                The size that the key and value states will have during the attention computation.
            kv_offset (`int`, optional):
                An optional offset to indicate at which first position the key and values states will refer to.
            mask_function (`Callable`):
                The mask factory function describing the mask pattern.
            attention_mask (`torch.Tensor`, optional):
                The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length)
            local_size (`int`, optional):
                The size of the local attention, if we do not use full attention. This is used only if `allow_is_causal_skip=True`
                to try to skip mask creation if possible.
            allow_is_causal_skip (`bool`, optional):
                Whether to allow to return `None` for the mask under conditions where we can use the `is_causal` argument in
                `torch.sdpa` instead. Default to `True`.
            allow_torch_fix (`bool`, optional):
                Whether to update the mask in case a query is not attending to any tokens, to solve a bug in torch's older
                versions. We need an arg to skip it when using eager. By default `True`.
        """
        q_length = cache_position.shape[0]
        # Potentially pad the 2D mask, and slice it correctly
        padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset)

        # Under specific conditions, we can avoid materializing the mask, instead relying on the `is_causal` argument
        if allow_is_causal_skip and _ignore_causal_mask_sdpa(padding_mask, q_length, kv_length, kv_offset, local_size):
            return None

        # Similar to `kv_arange = torch.arange(start=kv_offset, end=kv_offset + kv_length, device=cache_position.device)`
        # but without data-dependent slicing (i.e. torch.compile friendly)
        kv_arange = torch.arange(kv_length, device=cache_position.device)
        kv_arange += kv_offset

        # This creates the 4D mask easily. Note that we do not include vmap over the batch_idx dimension as well,
        # as vmap cannot handle slicing a tensor from scalar tensor (it internally calls `.item()` which vmap does not allow
        # However, in more recent version of Pytorch, a trick was introduced to handle it - which is the reason we have
        # `sdpa_mask_recent_torch`, as it allows more general `mask_function`

        # MS adapt: fix vmap
        causal_mask = mask_function(slice(None), None, cache_position.reshape(cache_position.shape[0], 1), kv_arange.reshape(1, kv_arange.shape[0]))
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, -1, -1, -1)
        if padding_mask is not None:
            causal_mask = causal_mask * padding_mask[:, None, None, :]

        # Due to a bug in versions of torch<2.5, we need to update the mask in case a query is not attending to any tokens (due to padding)
        if not _is_torch_greater_or_equal_than_2_5 and allow_torch_fix:
            causal_mask |= torch.all(~causal_mask, dim=-1, keepdim=True)
        return causal_mask
