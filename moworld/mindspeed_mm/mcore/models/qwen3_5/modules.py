from contextlib import nullcontext
from typing import Optional, Union, List
from dataclasses import dataclass
from einops import rearrange
import copy

import torch
from torch import Tensor, nn

from megatron.core import tensor_parallel
from megatron.core.transformer.attention import (
    BaseInferenceContext,
    PackedSeqParams,
    SelfAttention,
    SelfAttentionSubmodules,
    deprecate_inference_params,
    CrossAttentionSubmodules
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.enums import Fp8Recipe
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.transformer.transformer_block import TransformerBlock, TransformerBlockSubmodules
from megatron.core.utils import WrappedTensor, deprecate_inference_params, make_viewless_tensor, divide, is_te_min_version
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.tensor_parallel.mappings import all_gather_last_dim_from_tensor_parallel_region
from megatron.core.parallel_state import get_tensor_model_parallel_rank
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.models.common.embeddings.rope_utils import (
    _apply_rotary_pos_emb_bshd,
    get_pos_emb_on_this_cp_rank,
)


try:
    from megatron.core.extensions.transformer_engine import (
        SplitAlongDim,
        TELinear,
        TENorm,
        set_save_original_input,
        te_checkpoint
    )
    HAVE_TE = True
except ImportError:
    SplitAlongDim, TELinear, TENorm, set_save_original_input, te_checkpoint = None, None, None, None, None
    HAVE_TE = False

from mindspeed.te.pytorch.utils import get_tensor_model_parallel_group_if_none
from mindspeed_mm.mcore.process_group_configs import ProcessGroupCollection
from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.mcore.models.qwen3_5.rope import apply_rotary_pos_emb_absolute


class Qwen3_5MultimodalRotaryEmbedding(nn.Module):
    """Multimodal Rotary Embedding for language model.
    only support for qwen3.5

    Args:
        kv_channels (int): Projection weights dimension in multi-head attention. Obtained
            from transformer config
        rotary_percent (float): Percent of rotary dimension to use for rotary position
            embeddings.
        rotary_interleaved (bool, optional): If True, interleaved rotary position embeddings.
            Defaults to False.
        seq_len_interpolation_factor (float, optional): scale of linearly interpolating RoPE
            for longer sequences. The value must be a float larger than 1.0. Defaults to None
        rotary_base (int, optional): Base period for rotary position embeddings. Defaults to
            10000.
    """

    def __init__(
        self,
        kv_channels: int,
        rotary_percent: float = 1.0,
        rotary_interleaved: bool = False,
        seq_len_interpolation_factor: Optional[float] = None,
        rotary_base: int = 10000,
        cp_group: torch.distributed.ProcessGroup = None,
    ) -> None:
        super().__init__()

        dim = kv_channels
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)
        self.rotary_interleaved = rotary_interleaved

        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        self.inv_freq = 1.0 / (
            rotary_base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=torch.cuda.current_device()) / dim)
        )
        self.is_thd_format = False  # if is thd format, we do not need to split the rotary_pos_emb along CP

        # default mrope section is [24, 20, 20], if no mrope section is provided, use default mrope section
        self.mrope_section = [24, 20, 20]
        self.cp_group = cp_group

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THTHWHTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def forward(
        self,
        position_ids: torch.Tensor,
        mrope_section: List[int] | None,
        **kwargs,
    ) -> Tensor:
        """Forward pass of multimodal RoPE embedding.

        Args:
            position_ids (torch.Tensor): A postion_id tensor with shape [3, batchsize, seqlens]
            mrope_section (list[int]): Multimodal rope section is for channel dimension of temporal,
                height and width in rope calculation.
            packed_seq_params (PackedSeqParams, optional): Packed sequence params. Defaults to None.
        Returns:
            Tensor: Embeddings after applying RoPE.
        """
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        # Use fp32 for position indices to avoid precision loss when inv_freq is bf16.
        seq = position_ids.to(device=self.inv_freq.device, dtype=torch.float32)

        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor

        # shape (3, bs, dim, 1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].expand(3, seq.shape[1], -1, 1)
        # shape (3, bs, 1, seq_length)
        seq_expanded = seq[:, :, None, :].float()
        # shape (3, bs, seq_length, dim)
        freqs = (inv_freq_expanded @ seq_expanded).transpose(2, 3)
        if mrope_section is not None:
            freqs = self.apply_interleaved_mrope(freqs, mrope_section)
        else:
            # if mrope_section is not provided, use default mrope section
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)

        # shape (seq_length, bs, 1, 2 * dim)
        emb = emb[..., None, :].transpose(0, 1).contiguous()
        if self.cp_group.size() > 1 and not self.is_thd_format:
            # slice rotary_pos_emb along sequence dimension and select the parition of the current
            # CP rank
            emb = get_pos_emb_on_this_cp_rank(emb, 0, self.cp_group)
        return emb


def get_rope_index(
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    packed_seq_params: Optional[PackedSeqParams] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Different from the original implementation, Qwen3.5 use timestamps rather than absolute time position ids."""

    # Since we use timestamps to seperate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    if packed_seq_params is not None and attention_mask is None and input_ids is not None:
        # Build an attention mask from packed sequence metadata when one is not provided.
        # cu_seqlens_q entries are cumulative lengths; their diffs give per-sample lengths.
        cu_seqlens = packed_seq_params.cu_seqlens_q
        if cu_seqlens is not None and cu_seqlens.numel() >= 2:
            seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            attention_mask = torch.zeros_like(input_ids, dtype=input_ids.dtype)
            max_len = attention_mask.shape[1]
            for i, seq_len in enumerate(seq_lens.tolist()):
                valid = min(int(seq_len), max_len)
                attention_mask[i, :valid] = 1
        else:
            # Fallback to a dense mask if packed metadata is missing.
            attention_mask = torch.ones_like(input_ids)

    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        # Handle multi-dimensional attention masks
        elif attention_mask.dim() > 2:
            # Collapse to [batch, seq] while preserving padding information
            attention_mask = attention_mask.any(dim=-1)
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.squeeze(1)
            attention_mask = attention_mask.to(dtype=total_input_ids.dtype)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, sample_input_ids in enumerate(total_input_ids):
            sample_input_ids = sample_input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(sample_input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = sample_input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = sample_input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=total_input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            # Handle multi-dimensional attention mask
            if attention_mask.dim() > 2:
                # Collapse to [batch, seq] while preserving padding information
                attention_mask = attention_mask.any(dim=-1)
                if attention_mask.dim() == 3:
                    attention_mask = attention_mask.squeeze(1)
                attention_mask = attention_mask.to(dtype=torch.long)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas


def apply_rotary_pos_emb_thd_absolute(
    t: Tensor, cu_seqlens: Tensor, freqs: Tensor, rotary_interleaved: bool = False
) -> Tensor:
    """A baseline implementation of applying RoPE for `thd` format.

    Args:
        t (Tensor): Input tensor T is of shape [t, h, d]
        cu_seqlens(Tensor):  Cumulative sum of sequence lengths in a batch for `t`,
        with shape [b + 1] and dtype torch.int32. Currently unused but kept for API consistency.
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [max_s, 1, 1, d]

    Returns:
        Tensor: Shape [t, h, d]. The input tensor after applying RoPE.
    """
    return _apply_rotary_pos_emb_bshd(t[:, None], freqs, rotary_interleaved=rotary_interleaved).squeeze(1)


def apply_rotary_pos_emb_absolute(
    t: Tensor,
    freqs: Tensor,
    config = None,
    cu_seqlens: Optional[Tensor] = None,
):
    """
    Reroute to the appropriate apply_rotary_pos_emb function depending on
    bshd (conventional) / thd (packed seq) format

    In Qwen3-VL, the shape of freqs is (seq_length, bs, 1, 2 * dim) instead of [max_seqlen, 1, 1, 2 * dim]
    """
    assert not config.apply_rope_fusion
    orig_t_dtype = t.dtype
    if config.apply_rotary_pos_emb_in_fp32:
        t = t.float()

    if cu_seqlens is None:
        result = _apply_rotary_pos_emb_bshd(t, freqs, rotary_interleaved=config.rotary_interleaved)
    else:
        result = apply_rotary_pos_emb_thd_absolute(t, cu_seqlens, freqs, rotary_interleaved=config.rotary_interleaved)

    if config.apply_rotary_pos_emb_in_fp32:
        result = result.to(orig_t_dtype)

    return result


@dataclass
class Qwen3_5SelfAttentionSubmodules(SelfAttentionSubmodules):
    """Submodules for the Qwen3Next self-attention layer with NPU."""
    q_proj: Union[ModuleSpec, type] = None
    k_proj: Union[ModuleSpec, type] = None
    v_proj: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None


class Qwen3_5SelfAttention(SelfAttention):
    """
    Overrides the SelfAttention class, the difference is that qwen3.5 uses apply_rotary_pos_emb_absolute
    instead of apply_rotary_pos_emb
    """
    def __init__(
        self,
        config: TransformerConfig,
        submodules: Union[SelfAttentionSubmodules, CrossAttentionSubmodules],
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str = "self",
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        pp_layer_offset: Optional[int] = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attention_type,
            pg_collection=pg_collection,
            cp_comm_type=cp_comm_type
        )
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        else:
            if not hasattr(pg_collection, 'tp') or not hasattr(pg_collection, 'cp'):
                raise ValueError("pg_collection must have 'tp' and 'cp' process groups")
        self.config = config
        self.layer_number = layer_number
        self._pp_layer_offset = pp_layer_offset

        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type

        # For normal attention without groups, num_query_groups == num_attention_heads,
        # so these two will be the same
        self.query_projection_size = self.config.kv_channels * self.config.num_attention_heads
        self.kv_projection_size = self.config.kv_channels * self.config.num_query_groups

        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp

        # Per attention head and per partition values
        world_size = self.pg_collection.tp.size()
        self.hidden_size_per_attention_head = divide(
            self.query_projection_size, self.config.num_attention_heads
        )
        if self.config.num_query_groups < world_size:
            # When num_kv_heads < tp_size, each TP rank (post AG) initially produces
            # activations for 1 kv_head and (num_q_heads / num_kv_heads) q_heads.
            # We then pull out the appropriate (num_q_heads / tp_size) q_heads.
            self.num_query_groups_per_partition = 1
            self.num_attention_heads_per_partition = divide(
                self.config.num_attention_heads, self.config.num_query_groups
            )
        else:
            # When num_kv_heads >= tp_size, each TP rank produces activations for
            # (num_kv_heads / tp_size) kv_heads and (num_q_heads / tp_size) q_heads.
            self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)
            self.num_attention_heads_per_partition = divide(
                self.config.num_attention_heads, world_size
            )
        self.world_size = world_size

        # To support both CUDA Graphs and key value with different hidden size
        self.key_hidden_size = self.hidden_size_per_attention_head
        self.val_hidden_size = self.hidden_size_per_attention_head

        if self.config.num_query_groups < world_size:
            # TE throws an assertion error if num_kv_heads / num_query_groups
            # is not divisible by TP size.
            tmp_config = copy.deepcopy(self.config)
            tmp_config.num_query_groups = world_size
        else:
            tmp_config = self.config

        self.linear_qkv_out_dim = self.query_projection_size + 2 * self.kv_projection_size
        if self.config.attention_output_gate:
            self.linear_qkv_out_dim += self.config.kv_channels * self.config.num_attention_heads
        self.linear_qkv = submodules.linear_qkv(
            self.config.hidden_size,
            self.linear_qkv_out_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='qkv',
        )

        self.core_attention = submodules.core_attention(
            config=tmp_config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
            cp_comm_type=cp_comm_type,
            softmax_scale=self.config.softmax_scale,
        )

        self.checkpoint_core_attention = (
            self.config.recompute_granularity == 'selective'
            and "core_attn" in self.config.recompute_modules
        )

        self.offload_qkv_linear = (
            self.config.fine_grained_activation_offloading
            and "qkv_linear" in self.config.offload_modules
        )

        self.offload_core_attention = (
            self.config.fine_grained_activation_offloading
            and "core_attn" in self.config.offload_modules
        )

        self.offload_attn_proj = (
            self.config.fine_grained_activation_offloading
            and "attn_proj" in self.config.offload_modules
        )

        # Output.
        self.linear_proj = build_module(
            submodules.linear_proj,
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='proj',
        )

        if (
            HAVE_TE
            and isinstance(self.linear_proj, TELinear)
            and (
                (
                    self.config.fp8
                    and self.config.fp8_recipe != 'delayed'
                    and is_te_min_version("2.6.0dev0")
                )
                or (self.config.fp4 and is_te_min_version("2.7.0.dev0"))
            )
        ):
            # For fp8/fp4 training, the output of the fused core_attn is saved by itself, and
            # linear_proj also saves the quantized tensor of this output. Here we set the
            # linear_proj to save the original input tensors to avoid the extra memory usage of
            # the quantized tensor.
            set_save_original_input(self.linear_proj)


    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        rotary_pos_emb: (Tensor | tuple[Tensor, Tensor]) | None = None,
        rotary_pos_cos: Tensor | None = None,
        rotary_pos_sin: Tensor | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: int | None = None,
        *,
        inference_params: BaseInferenceContext | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Perform a forward pass through the attention module.

        Args:
            hidden_states (Tensor): Hidden states.
            attention_mask (Tensor): Attention mask.
            key_value_states (Optional[Tensor]): Key/value states (for cross attention).
            inference_context (Optional[BaseInferenceContext]): Inference context that manages
                KV cache.
            rotary_pos_emb (Optional[Union[Tensor, Tuple[Tensor, Tensor]]]): Rotary
                embedding tensor(s).
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            attention_bias (Optional[Tensor]): Attention bias.
            packed_seq_params (Optional[PackedSeqparams]): Parameters used for THD format.
            sequence_len_offset (Optional[int]): Sequence length offset used for
                inference CUDA graphs.

        Return:
            (Tuple[Tensor, Tensor]) Attention output and bias.

        """
        # Check if we need to skip RoPE
        # no_rope is 0-indexed array and self.layer_number is 1-indexed
        no_rope = self.config.no_rope_freq[self.layer_number - 1] if self.config.no_rope_freq else False
        if no_rope:
            rotary_pos_emb = None

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # hidden_states: [sq, b, h]
        if self.config.flash_decode and not self.training and inference_context is not None:
            rotary_pos_emb = None

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        gate = None
        if self.config.attention_output_gate:
            query, key, value, gate = self.get_query_key_value_tensors(
                hidden_states, key_value_states, output_gate=True
            )
        else:
            query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)

        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================

        in_decode_mode = inference_context is not None and inference_context.is_decode_only() and not self.training

        # This branch only runs in the decode phase of flash decoding and returns after the linear
        # projection. This conditional is not used in the prefill phase or non-flash-decoding cases.
        if in_decode_mode and self.config.flash_decode:
            inference_key_memory, inference_value_memory = inference_context.key_value_memory_dict[self.layer_number]
            output = self.flash_decode(
                sequence_len_offset=sequence_len_offset,
                query_layer=query,
                key_layer=key,
                value_layer=value,
                inference_key_memory=inference_key_memory,
                inference_value_memory=inference_value_memory,
                rotary_cos=rotary_pos_cos,
                rotary_sin=rotary_pos_sin,
                rotary_interleaved=self.config.rotary_interleaved,
            )
            out = output.transpose(0, 1).contiguous()
            context_layer = out.view(out.size(0), out.size(1), -1)
            if gate is not None:
                context_layer = self._apply_output_gate(context_layer, gate)
            output, bias = self.linear_proj(context_layer)
            return output, bias

        if in_decode_mode and self.config.enable_cuda_graph and inference_context.is_static_batching():
            raise ValueError("CUDA graphs must use flash decode with static batching!")

        query, key, value, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(
            inference_context,
            query,
            key,
            value,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
        )

        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        if rotary_pos_emb is not None and not self.config.flash_decode:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None:
                if packed_seq_params.cu_seqlens_q_padded is not None:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
                else:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q
                if packed_seq_params.cu_seqlens_kv_padded is not None:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
                else:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None

            if q_pos_emb is not None:
                if inference_context is None or inference_context.is_static_batching():
                    query = apply_rotary_pos_emb_absolute(
                        query,
                        q_pos_emb,
                        config=self.config,
                        cu_seqlens=cu_seqlens_q,
                    )
                else:
                    query = inference_context.apply_rotary_emb_query(
                        query,
                        q_pos_emb,
                        self.config,
                        cu_seqlens_q,
                        self.model_comm_pgs.cp,
                    )
            if k_pos_emb is not None:
                key = apply_rotary_pos_emb_absolute(
                    key,
                    k_pos_emb,
                    config=self.config,
                    cu_seqlens=cu_seqlens_kv,
                )

        # ==================================
        # core attention computation
        # ==================================

        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            if inference_context is None or inference_context.is_static_batching():
                # Static batching attention kernel.
                core_attn_out = self.core_attention(
                    query,
                    key,
                    value,
                    attention_mask,
                    attn_mask_type=attn_mask_type,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                )

            else:
                # Dynamic batching attention kernel.
                q, k, v = (query, key, value)
                cu_query_lengths, max_seqlen_q = inference_context.cu_query_lengths()
                cu_kv_lengths, kv_lengths, kv_lengths_decode_only, max_seqlen_k = inference_context.cu_kv_lengths()

                core_attn_out = self.flash_decode_and_prefill(
                    q,
                    k,
                    v,
                    max_seqlen_q,
                    max_seqlen_k,
                    cu_query_lengths,
                    cu_kv_lengths,
                    kv_lengths,
                    kv_lengths_decode_only,
                )
                core_attn_out = rearrange(core_attn_out, "s b h d -> s b (h d)")

        if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        # Output gate (for Gated Attention in hybrid architectures like Qwen3.5)
        if gate is not None:
            core_attn_out = self._apply_output_gate(core_attn_out, gate)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.linear_proj(core_attn_out)

        return output, bias

    def get_query_key_value_tensors(
        self,
        hidden_states: Tensor,
        key_value_states: Tensor | None = None,
        output_gate: bool = False,
        split_qkv: bool = True,
    ) -> (
        tuple[Tensor, Tensor, Tensor, Tensor]
        | tuple[Tensor, Tensor, Tensor]
        | tuple[Tensor, list[int]]
    ):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        If `output_gate` is True, then also derives `gate` tensor.
        If `split_qkv=False`, then the unsplit mixed_qkv tensor is returned.
        """
        # If no output gate: Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        # If have output gate: Attention heads [sq, b, h] --> [sq, b, ng * (2 * np/ng + 2) * hn)]
        mixed_qkv, _ = self.linear_qkv(hidden_states)
        num_query_heads_per_group = (
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        )
        num_qkv_heads_per_group = num_query_heads_per_group + 2
        if output_gate:
            num_qkv_heads_per_group += num_query_heads_per_group

        if self.config.num_query_groups < self.world_size:
            # Note that weights are interleaved in the following manner:
            # q1 q2 k1 v1 | q3 q4 k2 v2 | q5 q6 k3 v3 | ...
            # When tp_size > num_kv_heads, we split "q1 q2 k1 v1" over multiple
            # ranks, so a rank does not have a clean partitioning of just the q_heads
            # it needs. Instead, we perform the following steps:
            # 1. Assemble the full "q1 q2 k1 v1 | q3 q4 k2 v2 | q5 q6 k3 v3 | ..."
            #    through an AG.
            # 2. Pull out the right slice (e.g., "q1 q2 k1 v1" or "q3 q4 k2 v2").
            # 3. Split q_heads (e.g., q1, q2), k_heads (e.g., k1), v_heads (e.g., v1).
            # 4. Further index into query to get only the q_heads that this rank is
            #    responsible for (e.g., q1).
            # The block of code below performs steps 1 and 2.
            mixed_qkv = all_gather_last_dim_from_tensor_parallel_region(mixed_qkv)
            idx = self.pg_collection.tp.rank() // (
                self.world_size // self.config.num_query_groups
            )
            size = mixed_qkv.size()[-1] // self.config.num_query_groups
            mixed_qkv = mixed_qkv[:, :, idx * size : (idx + 1) * size]

        # If no output gate: [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        # If have output gate: [sq, b, hp] --> [sq, b, ng, (2 * np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            num_qkv_heads_per_group * self.hidden_size_per_attention_head,
        )

        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        # Split the tensor into query, gate, key, and value.
        if output_gate:
            if not split_qkv:
                raise ValueError("split_qkv not supported for gated attention yet.")
            # If have output gate: [sq, b, ng, (2 * np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], [sq, b, ng, np/ng * hn],
            # [sq, b, ng, hn], [sq, b, ng, hn]
            split_arg_list = [
                num_query_heads_per_group * self.hidden_size_per_attention_head,
                num_query_heads_per_group * self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head,
            ]

            if SplitAlongDim is not None:
                (query, gate, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list)
            else:
                (query, gate, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)
        else:
            # If no output gate: [sq, b, ng, (np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], None, [sq, b, ng, hn], [sq, b, ng, hn]
            split_arg_list = [
                num_query_heads_per_group * self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head,
            ]

            # Return unsplit mixed_qkv and split_arg_list
            if not split_qkv:
                return mixed_qkv, split_arg_list

            if SplitAlongDim is not None:
                (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list)
            else:
                (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)

        # Query [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)

        if self.config.num_query_groups < self.world_size:
            # query above corresponds to (num_q_heads / num_kv_heads) q_heads.
            # Index appropriately into query to get (num_q_heads / tp_size) q_heads.
            # This is step 4 in the list of steps above.
            idx = self.pg_collection.tp.rank() % (
                self.world_size // self.config.num_query_groups
            )
            size = self.num_attention_heads_per_partition // (
                self.world_size // self.config.num_query_groups
            )
            query = query[:, :, idx * size : (idx + 1) * size, :]

        if self.q_layernorm is not None:
            query = self.q_layernorm(query)

        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        if self.config.test_mode:
            self.run_realtime_tests()

        if output_gate:
            # Gate [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
            gate = gate.reshape(*gate.shape[:2], -1, self.hidden_size_per_attention_head)
            if self.config.num_query_groups < self.world_size:
                idx = get_tensor_model_parallel_rank() % (
                    self.world_size // self.config.num_query_groups
                )
                size = self.num_attention_heads_per_partition // (
                    self.world_size // self.config.num_query_groups
                )
                gate = gate[:, :, idx * size : (idx + 1) * size, :]
            return query, key, value, gate

        return query, key, value

    def _apply_output_gate(self, x, gate):
        x_dtype = x.dtype
        gate = gate.contiguous()
        gate = gate.view(*x.shape)
        x = x * torch.sigmoid(gate.float())
        x = x.to(x_dtype)
        return x


class Qwen3_5VisionPatchEmbed(nn.Module):
    """
    Vision Patch Embed for Qwen3.5 vision model.
    """

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )

        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class Qwen3_5VisionRotaryEmbedding(nn.Module):
    """
    Vision Rotary Embedding for Qwen3.5 vision model.
    """

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, seqlen: int) -> torch.Tensor:
        if not hasattr(self, "inv_freq"):
            inv_freq = 1.0 / (
                self.theta
                ** (
                    torch.arange(
                        0,
                        self.dim,
                        2,
                        dtype=torch.float,
                        device=torch.cuda.current_device(),
                    )
                    / self.dim
                )
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Qwen3_5VisionTransformerBlock(TransformerBlock):
    """
    Vision Transformer Block for Qwen3.5 vision model.
    """

    def __init__(
        self,
        config,
        spec: Union[TransformerBlockSubmodules, ModuleSpec],
        post_layer_norm: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(
            config=config,
            spec=spec,
            post_layer_norm=post_layer_norm,
            pre_process=pre_process,
            post_process=post_process,
        )
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection
        self.cp_group = pg_collection.cp
        self.tp_group = pg_collection.tp
        self.pp_group = pg_collection.pp

    def _checkpointed_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor,
        context_mask: Tensor,
        rotary_pos_emb: Tensor,
        attention_bias: Tensor,
        packed_seq_params: PackedSeqParams,
        use_inner_fp8_context: bool,
    ):
        """Forward method with activation checkpointing."""

        def custom(start: int, end: int):
            def custom_forward(hidden_states, attention_mask, context, context_mask, rotary_pos_emb):
                for index in range(start, end):
                    layer = self._get_layer(index)
                    inner_fp8_context = (
                        get_fp8_context(self.config, layer.layer_number - 1)
                        if use_inner_fp8_context
                        else nullcontext()
                    )
                    with inner_fp8_context:
                        hidden_states, context = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            attention_bias=attention_bias,
                            inference_context=None,
                            packed_seq_params=packed_seq_params,
                        )
            return custom_forward

        def checkpoint_handler(forward_func):
            """Determines whether to use the `te_checkpoint` or `tensor_parallel.checkpoint`"""
            if self.config.fp8:
                return te_checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.tp_group,
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                )
            else:
                return tensor_parallel.checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                )

        if self.config.recompute_method == "uniform":
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            layer_idx = 0

            while layer_idx < self.num_layers_per_pipeline_rank:
                hidden_states, context = checkpoint_handler(
                    custom(layer_idx, layer_idx + self.config.recompute_num_layers)
                )
                layer_idx += self.config.recompute_num_layers

        elif self.config.recompute_method == "block":
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            recompute_skip_num_layers = 0
            for layer_idx in range(self.num_layers_per_pipeline_rank):
                # Skip recomputation when input grad computation is not needed.
                # Need to have at least one input tensor with gradient computation
                # for re-enterant autograd engine.
                if self.config.fp8 and not hidden_states.requires_grad:
                    recompute_skip_num_layers += 1
                if (
                    layer_idx >= recompute_skip_num_layers
                    and layer_idx < self.config.recompute_num_layers + recompute_skip_num_layers
                ):
                    hidden_states, context = checkpoint_handler(
                        custom(layer_idx, layer_idx + 1)
                    )
                else:
                    hidden_states, context = custom(layer_idx, layer_idx + 1)(
                        hidden_states,
                        attention_mask,
                        context,
                        context_mask,
                        rotary_pos_emb,
                    )
        else:
            raise ValueError("Invalid activation recompute method.")

        return hidden_states

    def forward(
        self,
        hidden_states: Union[Tensor, WrappedTensor],
        attention_mask: Optional[Tensor],
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ):
        """
        Perform the forward pass through the transformer block.

        This method handles the core computation of the transformer, including
        self-attention, optional cross-attention, and feed-forward operations.

        Args:
            hidden_states (Union[Tensor, WrappedTensor]): Input tensor of shape [s, b, h]
                where s is the sequence length, b is the batch size, and h is the hidden size.
                Can be passed as a WrappedTensor during inference to avoid an obsolete
                reference in the calling function.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask for cross-attention context
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            attention_bias (Tensor): Bias tensor for Q * K.T of shape in shape broadcastable
                to [b, num_head, sq, skv], e.g. [1, 1, sq, skv].
                Used as an alternative to apply attention mask for TE cuDNN attention.
            inference_context (BaseInferenceContext, optional): Parameters for inference-time
                optimizations.
            packed_seq_params (PackedSeqParams, optional): Parameters for packed sequence
                processing.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: The output hidden states tensor of shape
            [s, b, h], and optionally the updated context tensor if cross-attention is used.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # Delete the obsolete reference to the initial input tensor if necessary
        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # If fp8_recipe is delayed, wrap the entire pass with get_fp8_context(),
        # otherwise do nothing extra at the outer level
        # if we are using other fp8 recipes, then the context manager enter&exit are free
        # we can wrap fp8_context within the for loop over layers, so that we can fine-grained
        # control which layer will be fp8 or bf16
        use_outer_fp8_context = self.config.fp8 and self.config.fp8_recipe == Fp8Recipe.delayed
        use_inner_fp8_context = self.config.fp8 and self.config.fp8_recipe != Fp8Recipe.delayed
        outer_fp8_context = get_fp8_context(self.config) if use_outer_fp8_context else nullcontext()

        with rng_context, outer_fp8_context:
            # Forward pass.
            if self.config.recompute_granularity == "full" and self.training:
                hidden_states = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                    use_inner_fp8_context=use_inner_fp8_context,
                )
            else:
                for l_no, layer in enumerate(self.layers):
                    inner_fp8_context = (
                        get_fp8_context(self.config, layer.layer_number - 1)
                        if use_inner_fp8_context
                        else nullcontext()
                    )
                    with self.offload_context, inner_fp8_context:
                        hidden_states, context = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            rotary_pos_cos=rotary_pos_cos,
                            rotary_pos_sin=rotary_pos_sin,
                            attention_bias=attention_bias,
                            inference_context=inference_context,
                            packed_seq_params=packed_seq_params,
                            sequence_len_offset=sequence_len_offset,
                        )

                    if (
                        torch.is_grad_enabled()
                        and self.config.cpu_offloading
                        and self.group_prefetch_offload_commit_async is not None
                    ):
                        hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

        # Final layer norm.
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            # TENorm produces a "viewed" tensor. This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        # If this TransformerBlock is empty, input and output hidden states will be the same node
        # on the computational graph and will lead to unexpected errors in pipeline schedules.
        if not self.pre_process and len(self.layers) == 0 and not self.final_layernorm:
            hidden_states = hidden_states.clone()

        return hidden_states


@dataclass
class PatchMergerSubmodules:
    """
    Patch Merger Submodules for Qwen3.5 vision model.
    """

    patch_norm: Union[ModuleSpec, type] = None
    linear_fc1: Union[ModuleSpec, type] = None
    linear_fc2: Union[ModuleSpec, type] = None


class Qwen3_5VisionPatchMerger(MultiModalModule):
    """
    Vision Patch Merger for Qwen3.5 vision model.
    """

    def __init__(
        self,
        config,
        submodules: PatchMergerSubmodules,
        use_postshuffle_norm=False,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(config=config)

        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.input_size = config.hidden_size
        if self.use_postshuffle_norm:
            self.input_size = self.hidden_size
        self.tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=False)

        self.patch_norm = build_module(
            submodules.patch_norm,
            config=self.config,
            hidden_size=self.input_size,
            eps=self.config.layernorm_epsilon,
        )

        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.hidden_size,
            self.hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="patch_fc1",
        )

        self.activation_func = self.config.activation_func

        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.hidden_size,
            self.config.out_hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="patch_fc2",
        )

    def forward(self, hidden_states):
        if self.use_postshuffle_norm:
            hidden_states = hidden_states.view(-1, self.hidden_size)
        hidden_states = self.patch_norm(hidden_states)
        hidden_states = hidden_states.view(-1, self.hidden_size)

        hidden_states, _ = self.linear_fc1(hidden_states)
        hidden_states = self.activation_func(hidden_states)
        output, _ = self.linear_fc2(hidden_states)

        return output


class Qwen3_5TextTransformerBlock(TransformerBlock):
    """
    Transformer Block for Qwen3.5 model.
    """

    def __init__(
        self,
        config,
        spec: Union[TransformerBlockSubmodules, ModuleSpec],
        post_layer_norm: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(
            config=config,
            spec=spec,
            post_layer_norm=post_layer_norm,
            pre_process=pre_process,
            post_process=post_process,
        )
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection
        self.cp_group = pg_collection.cp
        self.tp_group = pg_collection.tp
        self.pp_group = pg_collection.pp

    def _checkpointed_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor,
        context_mask: Tensor,
        rotary_pos_emb: Tensor,
        attention_bias: Tensor,
        packed_seq_params: PackedSeqParams,
        use_inner_fp8_context: bool,
    ):
        """Forward method with activation checkpointing."""

        def custom(start: int, end: int):
            def custom_forward(
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
            ):
                for index in range(start, end):
                    layer = self._get_layer(index)
                    inner_fp8_context = (
                        get_fp8_context(self.config, layer.layer_number - 1)
                        if use_inner_fp8_context
                        else nullcontext()
                    )
                    with inner_fp8_context:
                        hidden_states, context = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            attention_bias=attention_bias,
                            inference_context=None,
                            packed_seq_params=packed_seq_params,
                        )
                return hidden_states, context

            return custom_forward

        def checkpoint_handler(forward_func):
            """Determines whether to use the `te_checkpoint` or `tensor_parallel.checkpoint`"""
            if self.config.fp8:
                return te_checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.tp_group,
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb
                )
            else:
                return tensor_parallel.checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb
                )

        if self.config.recompute_method == "uniform":
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            layer_idx = 0
            while layer_idx < self.num_layers_per_pipeline_rank:
                hidden_states, context = checkpoint_handler(
                    custom(layer_idx, layer_idx + self.config.recompute_num_layers)
                )

                layer_idx += self.config.recompute_num_layers

        elif self.config.recompute_method == "block":
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            recompute_skip_num_layers = 0
            for layer_idx in range(self.num_layers_per_pipeline_rank):
                # Skip recomputation when input grad computation is not needed.
                # Need to have at least one input tensor with gradient computation
                # for re-enterant autograd engine.
                if self.config.fp8 and not hidden_states.requires_grad:
                    recompute_skip_num_layers += 1
                if (
                    layer_idx >= recompute_skip_num_layers
                    and layer_idx < self.config.recompute_num_layers + recompute_skip_num_layers
                ):
                    hidden_states, context = checkpoint_handler(custom(layer_idx, layer_idx + 1))
                else:
                    hidden_states, context = custom(layer_idx, layer_idx + 1)(
                        hidden_states,
                        attention_mask,
                        context,
                        context_mask,
                        rotary_pos_emb
                    )
        else:
            raise ValueError("Invalid activation recompute method.")

        return hidden_states

    def forward(
        self,
        hidden_states: Union[Tensor, WrappedTensor],
        attention_mask: Optional[Tensor],
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None
    ):
        """
        Perform the forward pass through the transformer block.

        This method handles the core computation of the transformer, including
        self-attention, optional cross-attention, and feed-forward operations.

        Args:
            hidden_states (Union[Tensor, WrappedTensor]): Input tensor of shape [s, b, h]
                where s is the sequence length, b is the batch size, and h is the hidden size.
                Can be passed as a WrappedTensor during inference to avoid an obsolete
                reference in the calling function.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask for cross-attention context
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            attention_bias (Tensor): Bias tensor for Q * K.T of shape in shape broadcastable
                to [b, num_head, sq, skv], e.g. [1, 1, sq, skv].
                Used as an alternative to apply attention mask for TE cuDNN attention.
            inference_context (BaseInferenceContext, optional): Parameters for inference-time
                optimizations.
            packed_seq_params (PackedSeqParams, optional): Parameters for packed sequence
                processing.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: The output hidden states tensor of shape
            [s, b, h], and optionally the updated context tensor if cross-attention is used.
        """
        inference_context = deprecate_inference_params(inference_context, inference_params)

        # Delete the obsolete reference to the initial input tensor if necessary
        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()

        if not self.pre_process:
            hidden_states = self.input_tensor

        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # If fp8_recipe is delayed, wrap the entire pass with get_fp8_context(),
        # otherwise do nothing extra at the outer level
        # if we are using other fp8 recipes, then the context manager enter&exit are free
        # we can wrap fp8_context within the for loop over layers, so that we can fine-grained
        # control which layer will be fp8 or bf16
        use_outer_fp8_context = self.config.fp8 and self.config.fp8_recipe == Fp8Recipe.delayed
        use_inner_fp8_context = self.config.fp8 and self.config.fp8_recipe != Fp8Recipe.delayed
        outer_fp8_context = get_fp8_context(self.config) if use_outer_fp8_context else nullcontext()

        with rng_context, outer_fp8_context:
            # Forward pass.
            if self.config.recompute_granularity == "full" and self.training:
                hidden_states = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                    use_inner_fp8_context=use_inner_fp8_context
                )
            else:

                for l_no, layer in enumerate(self.layers):
                    inner_fp8_context = (
                        get_fp8_context(self.config, layer.layer_number - 1)
                        if use_inner_fp8_context
                        else nullcontext()
                    )
                    with self.offload_context, inner_fp8_context:
                        hidden_states, context = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            rotary_pos_cos=rotary_pos_cos,
                            rotary_pos_sin=rotary_pos_sin,
                            attention_bias=attention_bias,
                            inference_context=inference_context,
                            packed_seq_params=packed_seq_params,
                            sequence_len_offset=sequence_len_offset,
                        )

                    if (
                        torch.is_grad_enabled()
                        and self.config.cpu_offloading
                        and self.group_prefetch_offload_commit_async is not None
                    ):
                        hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

        # Final layer norm.
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            # TENorm produces a "viewed" tensor. This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        # If this TransformerBlock is empty, input and output hidden states will be the same node
        # on the computational graph and will lead to unexpected errors in pipeline schedules.
        if not self.pre_process and len(self.layers) == 0 and not self.final_layernorm:
            hidden_states = hidden_states.clone()

        return hidden_states
