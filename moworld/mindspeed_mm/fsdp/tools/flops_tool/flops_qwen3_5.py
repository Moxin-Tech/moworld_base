# The FLOPs calculation method is based on veomni, with some bugs fixed.

# Copyright 2025 Bytedance Ltd. and/or its affiliates

import argparse
from transformers import PretrainedConfig, AutoConfig


class Qwen35FlopsCounter:
    def __init__(self, config: PretrainedConfig):
        self.config = config

    def _estimate_qwen3_vit_flop(self, images_seqlens, config):
        """
        Estimate the FLOPS of the vision encoder
        """

        if config is None:
            return 0
        tokens_sum = sum(images_seqlens)

        num_heads = config.num_heads
        depth = config.depth

        dim = config.hidden_size
        mlp_hidden_dim = config.intermediate_size
        out_hidden_size = config.out_hidden_size

        spatial_merge_size = config.spatial_merge_size

        head_dim = dim // num_heads

        # every vision token's patch_embed comes from a conv of (C, T, H, W) -> (dim,)
        patch_embed_N = dim * config.in_channels * config.temporal_patch_size * config.patch_size * config.patch_size
        # Qwen3 VL vision mlp does not use GLU, thus 2.
        mlp_N = dim * mlp_hidden_dim * 2
        attn_linear_N = dim * (4 * dim)  # qkv and output proj
        merger_N = (out_hidden_size + (dim * (spatial_merge_size**2))) * (dim * (spatial_merge_size**2))

        # Qwen3 VL uses deep stack, one merger for every deepstack layer
        deepstack_merger_N = merger_N * len(config.deepstack_visual_indexes)
        # non-attn all_layer parm
        dense_N = patch_embed_N + (mlp_N + attn_linear_N) * depth + deepstack_merger_N + merger_N

        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum

        # In Qwen3 VL, full attention is used in all vision layers.
        full_attn_layer_num = depth

        # full attn layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in images_seqlens:
            seqlen_square_sum += seqlen * seqlen

        # Notice: This is the attention flops calculation for the full scenario,
        # which differs from veomni.
        # Forward pass: 4 * seqlen_square_sum * head_dim * num_heads
        # Backward pass: (8 + 2) * seqlen_square_sum * head_dim * num_heads.
        # This is because the FA operator recomputes the kv matrix during backward.
        # This implementation applies to both GPU and NPU.
        attn_qkv_flops = 14 * seqlen_square_sum * head_dim * num_heads * full_attn_layer_num

        vit_flops = dense_N_flops + attn_qkv_flops

        return vit_flops

    @staticmethod
    def _compute_hybrid_attn_params(config):
        """
        Compute hybrid attention (full + GatedDeltaNet) linear param count and layer info.

        Layers alternate between full attention and GatedDeltaNet (linear attention) in groups
        of `full_attention_interval` layers: (full_attention_interval - 1) linear layers followed
        by 1 full attention layer.

        Full attention (Qwen3_5Attention) projections:
            q_proj:  hidden_size -> num_attention_heads * head_dim  (output gate ignored, see note)
            k_proj:  hidden_size -> num_key_value_heads * head_dim
            v_proj:  hidden_size -> num_key_value_heads * head_dim
            o_proj:  num_attention_heads * head_dim -> hidden_size

        Note: q_proj actually outputs 2x (half query, half gate via sigmoid), but the gate
        contribution is ignored here for consistency with existing qwen3_next estimation.

        GatedDeltaNet (Qwen3_5GatedDeltaNet) projections:
            in_proj_qkv:  hidden_size -> 2 * linear_k_size + linear_v_size
            in_proj_z:    hidden_size -> linear_v_size          (output gate)
            in_proj_b:    hidden_size -> linear_num_value_heads (beta/gating scalar per head)
            in_proj_a:    hidden_size -> linear_num_value_heads (alpha/decay scalar per head)
            out_proj:     linear_v_size -> hidden_size
            conv1d:       depthwise, channels = 2 * linear_k_size + linear_v_size, kernel = conv_kernel_dim

        where:
            linear_k_size = linear_num_key_heads * linear_key_head_dim
            linear_v_size = linear_num_value_heads * linear_value_head_dim

        This only counts projection and conv1d parameter FLOPs. The GatedDeltaNet
        recurrence FLOPs are computed separately by _compute_gdn_recurrence_flops.
        """
        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)

        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        full_attention_interval = config.full_attention_interval
        num_full_attn_layers = config.num_hidden_layers // full_attention_interval

        # Full attention: q_proj + k_proj + v_proj + o_proj
        full_attn_linear_N = hidden_size * (q_size * 2 + k_size + v_size + num_attention_heads * head_dim)

        # GatedDeltaNet linear projections and depthwise conv1d
        linear_k_size = config.linear_num_key_heads * config.linear_key_head_dim
        linear_v_size = config.linear_num_value_heads * config.linear_value_head_dim
        # in_proj_qkv                          + in_proj_z    + in_proj_b + in_proj_a              + out_proj
        linear_attn_size = 2 * linear_k_size + 2 * linear_v_size + 2 * config.linear_num_value_heads + linear_v_size
        # depthwise conv1d: each of (2 * linear_k_size + linear_v_size) channels has its own kernel
        conv_N = config.linear_conv_kernel_dim * (2 * linear_k_size + linear_v_size)
        linear_attn_linear_N = hidden_size * linear_attn_size + conv_N

        # Each group has 1 full attention layer + (full_attention_interval - 1) GatedDeltaNet layers
        num_linear_attn_per_group = full_attention_interval - 1
        attn_linear_N = (full_attn_linear_N + num_linear_attn_per_group * linear_attn_linear_N) * num_full_attn_layers

        return attn_linear_N, num_full_attn_layers, head_dim, num_attention_heads

    @staticmethod
    def _compute_gdn_recurrence_flops(config, tokens_sum, num_full_attn_layers):
        """
        Compute FLOPs for the GatedDeltaNet recurrence across all GDN layers.

            S_t = gamma_t * S_{t-1} + eta_t * (v_t - S_{t-1} @ k_t) @ k_t^T
            o_t = S_t @ q_t

        where S_t is the state matrix of shape (linear_value_head_dim, linear_key_head_dim)
        per value head.

        Note: in practice, training uses the chunked implementation (chunk_gated_delta_rule)
        which reorganizes the computation into chunk-level matrix multiplications for better
        hardware utilization. However, chunking is purely an implementation optimization that
        does not change the total arithmetic — it computes the same result as the recurrent
        form. We therefore use the recurrent form as the theoretical FLOPs baseline.

        Per step per head, the dominant ops (forward) are:
            S_{t-1} @ k_t        (mat-vec, (d_v,d_k)@(d_k,)=(d_v,)):  2 * d_v * d_k FLOPs
            (...) @ k_t^T        (outer product, (d_v,)⊗(d_k,)=(d_v,d_k)):  d_v * d_k FLOPs
            o_t = S_t @ q_t      (mat-vec, (d_v,d_k)@(d_k,)=(d_v,)):  2 * d_v * d_k FLOPs
        where d_v = linear_value_head_dim, d_k = linear_key_head_dim.

        Following the same convention as quadratic attention (Q@K + attn@V):
            fwd: (2 + 1 + 2) * d_v * d_k = 5 * d_v * d_k per step per head
            fwd + bwd (3x): 15 * d_v * d_k per step per head
        """
        num_gdn_layers = config.num_hidden_layers - num_full_attn_layers
        return (
            15
            * config.linear_key_head_dim
            * config.linear_value_head_dim
            * config.linear_num_value_heads
            * tokens_sum
            * num_gdn_layers
        )

    def _estimate_qwen3_5_family_flops(self, tokens_sum, batch_seqlens, **kargs):
        """
        Estimate the FLOPS of the Qwen3.5 model family (dense/MoE MLP + hybrid attention + ViT).

        Handles both Qwen3.5 (dense) and Qwen3.5-MoE by checking for MoE-specific config
        attributes. Both variants share hybrid attention and ViT; only the MLP differs.

        Text model (from text_config):
            Dense MLP per layer (SwiGLU, 3 projections):
                gate_proj:  hidden_size -> intermediate_size
                up_proj:    hidden_size -> intermediate_size
                down_proj:  intermediate_size -> hidden_size

            MoE per layer (when num_experts is present):
                TopkGate router:   hidden_size -> num_experts
                Routed experts (top-k activated, each SwiGLU):
                    gate_proj:  hidden_size -> moe_intermediate_size
                    up_proj:    hidden_size -> moe_intermediate_size
                    down_proj:  moe_intermediate_size -> hidden_size
                    -> 3 projections * num_experts_per_tok active experts
                Shared expert (always active, SwiGLU):
                    gate_proj:  hidden_size -> shared_expert_intermediate_size
                    up_proj:    hidden_size -> shared_expert_intermediate_size
                    down_proj:  shared_expert_intermediate_size -> hidden_size

            Hybrid attention: see _compute_hybrid_attn_params docstring.

            LM head:
                lm_head:       hidden_size -> vocab_size

        Quadratic attention FLOPs (only full attention layers):
            Per layer: 2 * seq_len^2 * head_dim * num_attention_heads (Q@K + attn@V)
            fwd + bwd (3x) -> 6x total -> coefficient 12

        Vision encoder: delegates to _estimate_qwen3_vit_flop.
        """
        text_config = self.config.text_config
        hidden_size = text_config.hidden_size
        vocab_size = text_config.vocab_size
        num_hidden_layers = text_config.num_hidden_layers

        # hybrid attention linear projection params (full + GatedDeltaNet)
        attn_linear_N, num_full_attn_layers, head_dim, num_attention_heads = self._compute_hybrid_attn_params(
            text_config
        )

        # MLP params: MoE or dense depending on config
        is_moe = hasattr(text_config, "num_experts")
        if is_moe:
            # MoE per layer: router gate + routed expert MLPs (top-k) + shared expert MLP
            moe_gata_N = hidden_size * text_config.num_experts
            moe_expertmlp_N = hidden_size * text_config.moe_intermediate_size * text_config.num_experts_per_tok * 3
            moe_sharedexpertmlp_N = hidden_size * text_config.shared_expert_intermediate_size * 3
            mlp_N = (moe_gata_N + moe_expertmlp_N + moe_sharedexpertmlp_N) * num_hidden_layers
        else:
            # dense MLP per layer: gate_proj + up_proj + down_proj (SwiGLU)
            mlp_N = hidden_size * text_config.intermediate_size * 3 * num_hidden_layers

        # Notice: only lm_head, fix bug in veomni
        lm_head_N = vocab_size * hidden_size * 1
        # linear projection flops: 6 (fwd + bwd) * params * tokens
        dense_N_flops = 6 * (mlp_N + attn_linear_N + lm_head_N) * tokens_sum

        # quadratic attention flops (Q@K and attn@V), only for full attention layers
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen

        # Notice: This is the attention flops calculation for the causal scenario,
        # which differs from veomni.
        # Forward pass: 2 * seqlen_square_sum * head_dim * num_heads
        # Backward pass: (4 + 1) * seqlen_square_sum * head_dim * num_heads.
        # This is because the FA operator recomputes the kv matrix during backward.
        # This implementation applies to both GPU and NPU.
        attn_qkv_flops = 7 * seqlen_square_sum * head_dim * num_attention_heads * num_full_attn_layers

        # GatedDeltaNet recurrence flops (state update + query, for all GDN layers)
        gdn_recurrence_flops = self._compute_gdn_recurrence_flops(text_config, tokens_sum, num_full_attn_layers)

        # vit flops (Qwen3-VL ViT)
        images_seqlens = kargs.get("images_seqlens", None)
        if images_seqlens is not None:
            vit_flops = self._estimate_qwen3_vit_flop(images_seqlens, self.config.vision_config)
        else:
            vit_flops = 0

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops + gdn_recurrence_flops + vit_flops

        return flops_all_token

    def estimate_flops(self, batch_seqlens, step_time, **kwargs):
        """
        Estimate the FLOPS based on the number of valid tokens in the current batch and the time taken.

        Args:
            batch_seqlens (List[int]): A list where each element represents the number of valid tokens in the current batch.

        Returns:
            estimated_flops (float): The estimated FLOPS based on the input tokens and time.
            promised_flops (float): The expected FLOPS of the current device.
        """
        tokens_sum = sum(batch_seqlens)

        estimated_flops = self._estimate_qwen3_5_family_flops(tokens_sum, batch_seqlens, **kwargs) / step_time

        return estimated_flops


def get_args():
    parser = argparse.ArgumentParser(description="Qwen3.5 and Qwen3.6 FLOPs Calculation Tool")
    parser.add_argument("--vit_seqlens", type=int, default=0, nargs="+", help="seqlen in vit")
    parser.add_argument("--llm_seqlens", type=int, default=16384, nargs="+", help="seqlen in language_model")
    parser.add_argument("--hf_path", type=str, default="/home/weights/Qwen3.5-35B-A3B/", help="HuggingFace config path")
    parser.add_argument("--device_num", type=int, default=1, help="Device num")
    parser.add_argument("--gbs", type=int, default=1, help="global batchsize")
    parser.add_argument("--step_time", type=float, help="Step time (s)")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    flopcounter = Qwen35FlopsCounter(config=AutoConfig.from_pretrained(args.hf_path))
    flops = flopcounter.estimate_flops(
        batch_seqlens=args.llm_seqlens, images_seqlens=args.vit_seqlens, step_time=args.step_time
    )
    flops = flops * args.gbs / args.device_num
    print(f"flops: {flops:.4e}")


"""
e.g.:
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python mindspeed_mm/fsdp/tools/flops_tool/flops_qwen3_5.py \
    --vit_seqlens 1024 \
    --llm_seqlens 16384 \
    --hf_path /home/weights/Qwen3.5-35B-A3B/ \
    --device_num 1 \
    --gbs 1 \
    --step_time 6.9
"""
