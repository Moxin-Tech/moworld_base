from typing import Optional
import torch

from ...utils.device import IS_NPU_AVAILABLE, get_current_stream
from ...train.training_context import TrainingContext, TrainingStage
from ...features.memory.async_offload import OffloadManager, SwapTensor


if IS_NPU_AVAILABLE:
    import torch_npu


class SkipRecomputeFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        head_num: int,
        layout: str,
        pse: Optional[torch.Tensor],
        padding_mask: Optional[torch.Tensor],
        atten_mask: Optional[torch.Tensor],
        actual_seq_qlen,
        actual_seq_kvlen,
        scale,
        keep_prob,
        inner_precise,
        sparse_mode
    ):
        layer_idx, depth = TrainingContext().get_layer_index(), TrainingContext().get_model_depth()

        # Check the training state to determine if we should perform the forward computation
        if TrainingContext().get_training_stage() == TrainingStage.FORWARD:
            # --- Scenario 1: Forward Pass ---
            # Perform the actual Flash Attention computation on the NPU
            attn_output, softmax_max, softmax_sum, *_ = torch_npu.npu_fusion_attention(
                q,
                k,
                v,
                head_num,
                layout,
                pse=pse,
                padding_mask=padding_mask,
                atten_mask=atten_mask,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
                scale=scale,
                keep_prob=keep_prob,
                inner_precise=inner_precise,
                sparse_mode=sparse_mode,
            )

            # Offload the intermediate results (attn_output, softmax_max, softmax_sum) to CPU memory.
            # This allows us to skip recomputing Flash Attention during the backward pass.
            swap_tensors = [attn_output, softmax_max, softmax_sum]
            for swap_tensor in swap_tensors:
                key, after_block = OffloadManager().get_cnt(layer_idx)
                d2h_stream = OffloadManager().swap_stream
                if after_block:
                    # Clean up NPU memory for tensors from the previous layer that are no longer needed
                    OffloadManager().del_npu_tensor("{}_".format(layer_idx - 1))

                # Keep the last layer's tensors on NPU; for others, asynchronously transfer them to CPU memory (D2H)
                if layer_idx == depth - 1:
                    OffloadManager().put_npu_tensor(SwapTensor(swap_tensor, key))
                else:
                    swap_tensor = SwapTensor(swap_tensor, key)
                    swap_tensor.launch_d2h(d2h_stream)
                    OffloadManager().put(key, swap_tensor)

        else:
            # --- Scenario 2: Backward Pass (Recompute Skipped) ---
            # Instead of recomputing, directly restore the intermediate results saved during the forward pass
            if layer_idx == depth - 1:
                # Pop the saved results directly from NPU memory for the last layer
                softmax_sum = OffloadManager().pop_npu_tensor().tensor
                softmax_max = OffloadManager().pop_npu_tensor().tensor
                attn_output = OffloadManager().pop_npu_tensor().tensor
                # Prefetch data for the next iteration to hide data transfer latency
                OffloadManager().prefetch_get(layer_idx, 0, OffloadManager().swap_stream, OffloadManager().swap_stream)
            else:
                # Fetch the keys corresponding to the three intermediate variables for the current layer
                layer_items_keys = OffloadManager().get_layer_items_keys(layer_idx)[-3:]
                swap_tensors = []
                # Iterate in reverse order to transfer data asynchronously from CPU memory back to NPU (H2D)
                for swap_key in reversed(layer_items_keys):
                    swap_tensor = OffloadManager().get(swap_key)
                    swap_tensor.launch_h2d(OffloadManager().swap_stream)
                    # Wait for the H2D transfer to complete to ensure data readiness
                    get_current_stream().wait_event(swap_tensor.h2d_event)

                    swap_tensors.append(swap_tensor.tensor)

                    # Release the cached memory and trigger prefetching for the next chunk of data
                    OffloadManager().clear(swap_key)
                    _, tensor_idx = swap_key.split("_")
                    OffloadManager().prefetch_get(layer_idx, int(tensor_idx), OffloadManager().swap_stream, OffloadManager().swap_stream)
                softmax_sum, softmax_max, attn_output = swap_tensors

        ctx.head_num = head_num
        ctx.layout = layout
        ctx.pse = pse
        ctx.padding_mask = padding_mask
        ctx.atten_mask = atten_mask
        ctx.actual_seq_qlen = actual_seq_qlen
        ctx.actual_seq_kvlen = actual_seq_kvlen
        ctx.scale = scale
        ctx.keep_prob = keep_prob
        ctx.inner_precise = inner_precise
        ctx.sparse_mode = sparse_mode

        ctx.save_for_backward(q, k, v, attn_output, softmax_max, softmax_sum)

        return attn_output


    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, attn_output, softmax_max, softmax_sum = ctx.saved_tensors

        head_num = ctx.head_num
        layout = ctx.layout
        pse = ctx.pse
        padding_mask = ctx.padding_mask
        atten_mask = ctx.atten_mask
        actual_seq_qlen = ctx.actual_seq_qlen
        actual_seq_kvlen = ctx.actual_seq_kvlen
        scale = ctx.scale
        keep_prob = ctx.keep_prob
        inner_precise = ctx.inner_precise
        sparse_mode = ctx.sparse_mode

        grad_q, grad_k, grad_v, *_ = torch_npu.npu_fusion_attention_grad(
            q, k, v,
            grad_output,
            head_num,
            layout,
            softmax_max=softmax_max,
            softmax_sum=softmax_sum,
            attention_in=attn_output,
            pse=pse,
            padding_mask=padding_mask,
            atten_mask=atten_mask,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
            scale_value=scale,
            keep_prob=keep_prob,
            inner_precise=inner_precise,
            sparse_mode=sparse_mode
        )

        return (
            grad_q, grad_k, grad_v,  # Gradients for q, k, v
            None, None, None,        # head_num, layout, pse (replace with grad_pse if pse requires gradients)
            None, None,              # padding_mask, atten_mask
            None, None,              # actual_seq_qlen, actual_seq_kvlen
            None, None, None, None   # scale, keep_prob, inner_precise, sparse_mode
        )


def skip_recompute_flash_attention(
    q, k, v,
    head_num,
    layout,
    scale,
    pse=None,
    padding_mask=None,
    atten_mask=None,
    actual_seq_qlen=None,
    actual_seq_kvlen=None,
    keep_prob=1.0,
    inner_precise=0,
    sparse_mode=0
):
    attn_output = SkipRecomputeFlashAttention.apply(
        q, k, v, head_num, layout, pse, padding_mask, atten_mask, actual_seq_qlen, actual_seq_kvlen, scale, keep_prob, inner_precise, sparse_mode
    )

    return attn_output
