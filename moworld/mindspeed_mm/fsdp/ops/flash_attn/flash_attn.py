from typing import Optional, Union, List, Tuple
import logging

from einops import rearrange
import torch
import torch.distributed as dist
from transformers.modeling_flash_attention_utils import (
    _flash_attention_forward as _transformers_flash_attention_forward,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ...distributed.parallel_state import get_parallel_state
from ...utils.device import IS_NPU_AVAILABLE
from ...distributed.context_parallel.utils import cal_split_sizes
from ...distributed.context_parallel.communication import all_to_all
from .skip_recompute_flash_attn import skip_recompute_flash_attention

if IS_NPU_AVAILABLE:
    import torch_npu
    from ...distributed.context_parallel.ring_context_parallel.ring_context_parallel import (
        ringattn_context_parallel,
        ringattn_context_parallel_tnd_general,
    )


logger = logging.getLogger(__name__)
_flash_attention_forward = None


ATTN_MASK_NPU_CACHE = {}


def get_attn_mask(device, seq_len=2048):
    """Get or create NPU attention mask"""
    if device not in ATTN_MASK_NPU_CACHE:
        ATTN_MASK_NPU_CACHE[device] = torch.triu(torch.ones([seq_len, seq_len], device=device), diagonal=1).bool()
    return ATTN_MASK_NPU_CACHE[device]


def _convert_cu_seq_lens(
    cu_seq_lens: Optional[Union[torch.Tensor, List[int]]], input_layout: Optional[str] = None
) -> Tuple[Optional[List[int]], Optional[str]]:
    """
    Converts cu_seq_lens to a list and validates input_layout constraints.

    Args:
        cu_seq_lens: Cumulative sequence lengths, can be a Tensor or List.
        input_layout: The layout of the input tensor (e.g., 'BNSD', 'BSND').

    Returns:
        A tuple containing the processed cu_seq_lens (as list) and the validated layout.
    """

    # 1. Convert Tensor to List if necessary
    if isinstance(cu_seq_lens, torch.Tensor):
        cu_seq_lens = cu_seq_lens.tolist()

    if input_layout:
        # 2. Normalize layout string
        input_layout = input_layout.upper()

        # Check constraints for packed layouts (BNSD/BSND)
        if input_layout in ["BNSD", "BSND"]:
            if cu_seq_lens is not None and len(cu_seq_lens) > 2:
                raise RuntimeError(
                    f"NPU flash attention layout {input_layout} does not support packing data "
                    f"when cu_seq_lens length > 2"
                )

        # Auto-correct layout if cu_seq_lens is missing
        elif input_layout == "1TND" and (cu_seq_lens is None or len(cu_seq_lens) == 2):
            # Fallback to standard layout if packing info is missing
            input_layout = "BSND"
            cu_seq_lens = None

        elif input_layout == "1NTD" and (cu_seq_lens is None or len(cu_seq_lens) == 2):
            # Fallback to standard layout if packing info is missing
            input_layout = "BNSD"
            cu_seq_lens = None

    return cu_seq_lens, input_layout


def transformers_flash_attention_forward(
    query,
    key,
    value,
    attention_mask,
    **kwargs,
):
    attn_implementation = kwargs.pop("attn_implementation")
    return _transformers_flash_attention_forward(
        query,
        key,
        value,
        attention_mask,
        implementation=attn_implementation,
        **kwargs,
    )


def do_ring_attention(
    q,
    k,
    v,
    head_num,
    softmax_scale,
    is_causal,
    fa_layout="SBH",
    attn_mask=None,
    dropout_p=0.0,
    seq_split_lens: Optional[list[int] | torch.Tensor] = None,
):
    ps = get_parallel_state()
    cp_group = ps.get_ring_group()
    cp_size = ps.get_ring_group_size()
    rank = ps.get_ring_rank()
    cp_global_ranks = ps.get_ring_device_mesh().mesh.tolist()

    cp_para = dict()

    cp_para["causal"] = is_causal
    cp_para["cp_group"] = cp_group
    cp_para["cp_size"] = cp_size
    cp_para["rank"] = rank

    cp_para["cp_global_ranks"] = cp_global_ranks
    cp_para["cp_group_for_send_recv_overlap"] = None
    cp_para["megatron_cp_in_bnsd"] = fa_layout.upper() == "BNSD"

    if fa_layout.upper() == "SBH" or fa_layout.upper() == "BNSD":
        # 输入shapes是一维list
        if seq_split_lens is not None:
            seq_split_lens = seq_split_lens.cpu().tolist()
        output = ringattn_context_parallel(
            q, k, v, head_num, cp_para, softmax_scale, attn_mask, dropout_p, shapes=seq_split_lens
        )
    elif not is_causal and fa_layout.upper() == "TND":
        # 输入shapes是二维tensor
        output = ringattn_context_parallel_tnd_general(
            q, k, v, head_num, cp_para, softmax_scale, attn_mask, dropout_p, shapes=seq_split_lens
        )
    elif is_causal and fa_layout.upper() == "TND":
        raise NotImplementedError(f"Ring Attention TND layout not support causal mask now.")
    else:
        raise ValueError(
            f"Ring Attention only support fa layout: `SBH`、`SBND` and `TND`, bug got {fa_layout.upper()}."
        )

    return output


def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    cu_seq_lens_q: Optional[Union[torch.LongTensor, List[int]]] = None,
    cu_seq_lens_k: Optional[Union[torch.LongTensor, List[int]]] = None,
    input_layout: str = "1NTD",  # Input layout for query, key, and value.
    # Only effective on NPU when attn_implementation is 'flash_attention_2'.
    # Valid options: BNSD, BSND, 1NTD, 1TND.
    ring_in_bnsd: bool = False,  # do_ring_attention input_layout is sbh, calculate in bnsd layout, only used for ring cp or hybrid cp.
    skip_ulysses: bool = False,  # Skip ulysses for some ViT cases like internvl3.5
    total_seq_len: int = None,  # unaligned cp need this
    seq_split_lens: torch.Tensor = None,  # unaligned cp need this
    skip_flash_attn_recompute: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        logger.warning_once(
            "`flash_attention_2` does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )

    # Notice: Device & Implementation Detection
    # Detect if we are running on NPU with Flash Attention 2/3 implementation.
    use_npu_fusion_fa = IS_NPU_AVAILABLE and module.config._attn_implementation in [
        "flash_attention_2",
        "flash_attention_3",
    ]

    # Input Preprocessing (NPU Specific)
    if use_npu_fusion_fa:
        # Normalize layout and validate supported formats.
        input_layout = input_layout.upper()
        if input_layout not in ["BSND", "BNSD", "1TND", "1NTD"]:
            raise RuntimeError(
                f"Flash Attention input layout only supports `BSND`, `BNSD`, `1TND`, and `1NTD`. " f"Got {input_layout}"
            )

        # Convert cumulative sequence lengths to list format for NPU API.
        # (For ring cp, cu_seqlens info in seq_split_lens)
        if not dist.is_initialized() or not (get_parallel_state().is_ring_enable() and seq_split_lens is not None):
            cu_seq_lens_q, input_layout = _convert_cu_seq_lens(cu_seq_lens_q, input_layout=input_layout)
            cu_seq_lens_k, input_layout = _convert_cu_seq_lens(cu_seq_lens_k, input_layout=input_layout)

        # Determine sequence length based on layout.
        if input_layout in ["BSND", "1TND"]:
            seq_len = query.shape[1]
        elif input_layout in ["BNSD", "1NTD"]:
            seq_len = query.shape[2]
    else:
        # This is before the transpose
        seq_len = query.shape[2]

    if any(dim == 0 for dim in query.shape):
        raise ValueError(
            "Tensor query has shape  with a zero dimension.\n"
            "FlashAttention does not support inputs with dim=0.\n"
            "Please check your input shapes or use SDPA instead."
        )

    if not use_npu_fusion_fa:
        # FA2 uses non-transposed inputs
        query = query.transpose(1, 2)  # bsnd or 1tnd
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (usually our RMSNorm modules handle it correctly)
    target_dtype = None
    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(module.config, "_pre_quantization_dtype"):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype

    # Instead of relying on the value set in the module directly, we use the is_causal passed in kwargs if it is presented
    is_causal = kwargs.pop("is_causal", None)
    if is_causal is None:
        is_causal = module.is_causal

    # Modification: ============= CONTEXT PARALLEL (CP) =============
    ps = get_parallel_state()
    is_ulysses_enabled = ps.is_ulysses_enable() if dist.is_initialized() else False
    is_ring_enabled = ps.is_ring_enable() if dist.is_initialized() else False

    if use_npu_fusion_fa:
        if input_layout in ["BSND", "1TND"]:
            head_dim_index = 2
            seq_dim_index = 1
        elif input_layout in ["BNSD", "1NTD"]:
            head_dim_index = 1
            seq_dim_index = 2
    else:
        head_dim_index = 2
        seq_dim_index = 1
    q_head_num = query.shape[head_dim_index]
    kv_head_num = key.shape[head_dim_index]

    # ulysses validation
    if is_ulysses_enabled:
        ulysses_size = ps.get_ulysses_group_size()
        if q_head_num % ulysses_size != 0:
            raise ValueError(f"num_query_heads ({q_head_num}) must be divisible by ulysses_size ({ulysses_size})")
        if ulysses_size > kv_head_num:
            if ulysses_size % kv_head_num != 0:
                raise ValueError(
                    f"ulysses_size ({ulysses_size}) must be divisible by num_key_value_heads ({kv_head_num})"
                )
            n_repeat = ulysses_size // kv_head_num
            # Shape before: (batch_size, seq_len, kv_head_num, head_dim)
            # This repeats the K/V heads (dim 2) to match the ulysses_size (SP world size)
            # Shape after: (batch_size, seq_len, kv_head_num * n_repeat, head_dim) where (kv_head_num * n_repeat) == ulysses_size
            key = torch.repeat_interleave(key, dim=head_dim_index, repeats=n_repeat)
            value = torch.repeat_interleave(value, dim=head_dim_index, repeats=n_repeat)

    if seq_split_lens is not None:
        if not isinstance(seq_split_lens, torch.Tensor):
            raise ValueError(f"seq_split_lens should be instance of torch.Tensor, bug got {type(seq_split_lens)}")
        if seq_split_lens.ndim != 1 and seq_split_lens.ndim != 2:
            raise ValueError(
                f"seq_split_lens should be a 1-dimensional tensor or a 2-dimensional tensor, bug got {seq_split_lens.shape}"
            )

    if is_ring_enabled:
        if not IS_NPU_AVAILABLE:
            raise ValueError(f"Ring Attention now only support in NPU.")

        if skip_flash_attn_recompute:
            raise ValueError(f"Not support skip_flash_attn_recompute for ring attention.")

        if use_npu_fusion_fa:
            if input_layout in ["1TND", "1NTD"]:
                ring_fa_layout = "TND"
            else:
                ring_fa_layout = "SBH"
        else:
            raise RuntimeError(f"Ring Attention now only support when _attn_implementation is flash_attention_2")

        # Validate tensor layout for Ring Attention
        # For TND format, ensure input is [1, n, t, d] where t = seq_len * batch_size
        if ring_fa_layout.upper() == "TND" and query.shape[0] != 1:
            raise ValueError(
                f"When Ring Attention's fa layout is `TND`, input format should be [1, n, t, d], which t equals seq_len * batch_size."
            )

        # For causal attention, Ring Attention doesn't need mask
        if is_causal:
            attention_mask = None

        # Split attention mask across ring groups
        if attention_mask is not None:
            if len(attention_mask.shape) == 2:  # [S_q, S_k]
                seq_dim = 0
            elif len(attention_mask.shape) == 3:  # [B, S_q, S_k]
                seq_dim = 1
            else:  # [B, 1, S_q, S_k]
                seq_dim = 2

            mask_row = attention_mask.chunk(ps.get_ring_group_size(), dim=seq_dim)[ps.get_ring_rank()].contiguous()
            attention_mask = [m.contiguous() for m in mask_row.chunk(ps.get_ring_group_size(), dim=seq_dim + 1)]

        if is_ulysses_enabled:
            # Calculate sequence length per ring rank
            if seq_split_lens is not None:
                if seq_split_lens.ndim == 1:
                    # For 1D seq_split_lens: directly get the sequence length for this ring rank
                    seq_len_this_ring_rank = seq_split_lens[ps.get_ring_rank()]
                else:
                    # For 2D seq_split_lens: sum the elements for this ring rank
                    seq_len_this_ring_rank = seq_split_lens[ps.get_ring_rank()].sum()
            elif total_seq_len is not None:
                # Calculate split sizes based on total sequence length and ring group size, then get this ring's portion
                seq_len_this_ring_rank = cal_split_sizes(total_seq_len, ps.get_ring_group_size())[ps.get_ring_rank()]
            else:
                seq_len_this_ring_rank = None

            # ulysses a2a
            query = all_to_all(
                query,
                ps.get_ulysses_group(),
                scatter_dim=head_dim_index,
                gather_dim=seq_dim_index,
                gather_size=seq_len_this_ring_rank,
            )
            key = all_to_all(
                key,
                ps.get_ulysses_group(),
                scatter_dim=head_dim_index,
                gather_dim=seq_dim_index,
                gather_size=seq_len_this_ring_rank,
            )
            value = all_to_all(
                value,
                ps.get_ulysses_group(),
                scatter_dim=head_dim_index,
                gather_dim=seq_dim_index,
                gather_size=seq_len_this_ring_rank,
            )

            # Update number of query heads after all-to-all
            q_head_num = q_head_num // ps.get_ulysses_group_size()

        # ring attention only support input layout: TND or SBH
        if ring_fa_layout.upper() == "TND":
            if input_layout in ["1NTD"]:
                # layout: 1NTD -> 1TND
                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)

            # layout：1TND -> TND
            query = query.reshape(-1, query.shape[-2], query.shape[-1])
            key = key.reshape(-1, key.shape[-2], key.shape[-1])
            value = value.reshape(-1, value.shape[-2], value.shape[-1])
        else:
            if input_layout in ["BSND"]:
                # layout：BSND -> SBH
                query = rearrange(query, "B S N D -> S B (N D)")
                key = rearrange(key, "B S N D -> S B (N D)")
                value = rearrange(value, "B S N D -> S B (N D)")
            else:
                # layout：BNSD -> SBH
                query = rearrange(query, "B N S D -> S B (N D)")
                key = rearrange(key, "B N S D -> S B (N D)")
                value = rearrange(value, "B N S D -> S B (N D)")

        ring_cal_layout = "BNSD" if ring_in_bnsd else ring_fa_layout
        if ring_fa_layout == "TND" and ring_cal_layout == "BNSD":
            raise NotImplementedError(f"Ring attention calculated in bnsd not support packing data")

        # ring attention calculate
        attn_output = do_ring_attention(
            query,
            key,
            value,
            q_head_num,
            softmax_scale=scaling,
            is_caual=is_causal,
            fa_layout=ring_cal_layout,
            attn_mask=attention_mask,
            dropout_p=dropout,
            seq_split_lens=seq_split_lens,
        )  # Output in sbh or tnd layout

        # Convert back to original layout: BSND or 1TND
        if ring_fa_layout.upper() == "TND":
            attn_output = attn_output.unsqueeze(0)
        else:
            attn_output = rearrange(attn_output, "S B (N D) -> B S N D", N=q_head_num)

        # usp
        if is_ulysses_enabled:
            attn_output = all_to_all(attn_output, ps.get_ulysses_group(), scatter_dim=1, gather_dim=2)

        return attn_output, None

    else:
        if is_ulysses_enabled and not skip_ulysses:
            # ulysses a2a
            query = all_to_all(
                query,
                ps.get_ulysses_group(),
                scatter_dim=head_dim_index,
                gather_dim=seq_dim_index,
                gather_size=total_seq_len,
            )
            key = all_to_all(
                key,
                ps.get_ulysses_group(),
                scatter_dim=head_dim_index,
                gather_dim=seq_dim_index,
                gather_size=total_seq_len,
            )
            value = all_to_all(
                value,
                ps.get_ulysses_group(),
                scatter_dim=head_dim_index,
                gather_dim=seq_dim_index,
                gather_size=total_seq_len,
            )

            # Only after all_to_all we got the full seq_len
            seq_len = query.shape[seq_dim_index]
            q_head_num = query.shape[head_dim_index]

        if use_npu_fusion_fa:
            if input_layout in ["BNSD", "BSND"]:
                layout = input_layout
            elif input_layout in ["1TND"]:
                layout = "TND"
            elif input_layout in ["1NTD"]:
                layout = "TND"
                # layout: 1NTD -> 1TND
                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)

            if input_layout in ["1TND", "1NTD"]:
                query = query.squeeze(0)
                key = key.squeeze(0)
                value = value.squeeze(0)

            if attention_mask is None and is_causal:
                attention_mask = get_attn_mask(device=query.device)

            # Check if the attention mask is valid
            if attention_mask is not None and (
                attention_mask.ndim != 2 or attention_mask.shape[0] != attention_mask.shape[1]
            ):
                attention_mask = get_attn_mask(device=query.device) if is_causal else None

            if skip_flash_attn_recompute:
                attn_output = skip_recompute_flash_attention(
                    query,
                    key,
                    value,
                    q_head_num,
                    layout,
                    pse=None,
                    padding_mask=None,
                    atten_mask=attention_mask,
                    actual_seq_qlen=cu_seq_lens_q,
                    actual_seq_kvlen=cu_seq_lens_k,
                    scale=scaling,
                    keep_prob=1 - dropout,
                    inner_precise=0,
                    sparse_mode=3 if is_causal else 0,
                )
            else:
                attn_output = torch_npu.npu_fusion_attention(
                    query,
                    key,
                    value,
                    q_head_num,
                    layout,
                    pse=None,
                    padding_mask=None,
                    atten_mask=attention_mask,
                    actual_seq_qlen=cu_seq_lens_q,
                    actual_seq_kvlen=cu_seq_lens_k,
                    scale=scaling,
                    keep_prob=1 - dropout,
                    inner_precise=0,
                    sparse_mode=3 if is_causal else 0,
                )[0]

            if input_layout in ["1TND", "1NTD"]:
                attn_output = attn_output.unsqueeze(0)

        else:
            if skip_flash_attn_recompute:
                raise NotImplementedError(
                    "Skipping flash_attn recompute is only supported on NPU when `_attn_implementation` is `flash_attention_2`."
                )

            attn_output = _flash_attention_forward(
                query,
                key,
                value,
                attention_mask,
                query_length=seq_len,
                is_causal=is_causal,
                dropout=dropout,
                softmax_scale=scaling,
                sliding_window=sliding_window,
                softcap=softcap,
                use_top_left_mask=False,
                target_dtype=target_dtype,
                attn_implementation=module.config._attn_implementation,
                layer_idx=module.layer_idx if hasattr(module, "layer_idx") else None,
                cu_seq_lens_q=cu_seq_lens_q,
                cu_seq_lens_k=cu_seq_lens_k,
                **kwargs,
            )

        # Ulysses: attention a2a
        if is_ulysses_enabled and not skip_ulysses:
            if use_npu_fusion_fa and input_layout in ["1NTD"]:
                # attn_output layout: TND
                seq_dim_index = 1
                head_dim_index = 2
            attn_output = all_to_all(
                attn_output, ps.get_ulysses_group(), scatter_dim=seq_dim_index, gather_dim=head_dim_index
            )

        if use_npu_fusion_fa:
            if input_layout in ["BNSD"]:
                attn_output = attn_output.transpose(1, 2)

        return attn_output, None


def apply_transformers_attention_patch():
    # ============= REGISTER CP-ENABLED FLASH ATTENTION IMPLEMENTATIONS =============
    ALL_ATTENTION_FUNCTIONS.register("flash_attention_2", flash_attention_forward)
    ALL_ATTENTION_FUNCTIONS.register("flash_attention_3", flash_attention_forward)

    global _flash_attention_forward
    _flash_attention_forward = transformers_flash_attention_forward
