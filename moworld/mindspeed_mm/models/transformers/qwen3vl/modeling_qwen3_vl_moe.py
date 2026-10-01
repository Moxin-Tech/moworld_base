# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.

from typing import Optional, Union

import torch
import torch_npu
import torch.nn as nn
from torch.distributed.tensor import DTensor

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin

from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring
from transformers.utils.generic import OutputRecorder, check_model_inputs
from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import Qwen3VLMoeConfig, Qwen3VLMoeTextConfig

from megatron.training import get_args
from megatron.core import mpu
from mindspeed_mm.models.common.gmm import npu_group_gemm

from mindspeed_mm.models.common.communications import split_forward_gather_backward_with_cp
from mindspeed_mm.models.common.fused_moe import fused_ep_forward

from .output import Qwen3VLMoeCausalLMOutputWithPast

from .modules import (
    Qwen3VLTextAttention,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextMLP,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLLMHead,
    Qwen3VLEmptyModule
)

from .modeling_qwen3_vl import (
    Qwen3VLTextModel,
    Qwen3VLModel,
    Qwen3VLVisionModel,
    Qwen3VLForConditionalGeneration
)


class Qwen3VLMoeTextExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.intermediate_size = config.moe_intermediate_size
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts * self.hidden_size, 2 * self.expert_dim))
        self.down_proj = nn.Parameter(torch.empty((self.num_experts * self.expert_dim, self.hidden_size)))
        self.act_fn = ACT2FN[config.hidden_act]

    def _view_experts_weight(self):
        gate_up_proj = self.gate_up_proj.to_local() if isinstance(self.gate_up_proj, DTensor) else self.gate_up_proj
        gate_up_proj = gate_up_proj.view(-1, self.hidden_size, 2 * self.expert_dim)

        down_proj = self.down_proj.to_local() if isinstance(self.down_proj, DTensor) else self.down_proj
        down_proj = down_proj.view(-1, self.expert_dim, self.hidden_size)
        return gate_up_proj, down_proj

    def forward(
        self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, router_indices: torch.Tensor, router_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        When training it is more efficient to just loop over the experts and compute the output for each expert
        as otherwise the memory would explode.

        For inference we can sacrifice some memory and compute the output for all experts at once. By repeating the inputs.

        Args:
            hidden_states (torch.Tensor): (batch_size * token_num, hidden_size)
            routing_weights (torch.Tensor): (batch_size * token_num, num_experts)
            router_indices (torch.Tensor): (batch_size * token_num, top_k)
        Returns:
            torch.Tensor
        """
        gate_up_proj, down_proj = self._view_experts_weight()

        if router_logits is not None:
            routing_weights = torch.zeros_like(router_logits).scatter_(1, router_indices, routing_weights)

        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
        if self.training:
            next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=self.num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                # we sum on the top_k and on the sequence length to get which experts
                # are hit this time around
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in expert_hit[:]:
                with torch.no_grad():
                    _, token_idx = torch.where(expert_mask[expert_idx[0]])
                current_state = hidden_states[token_idx]
                gate_up = current_state @ gate_up_proj[expert_idx]
                gate, up = gate_up.chunk(2, dim=-1)
                gated_output = up * self.act_fn(gate)
                out = gated_output @ down_proj[expert_idx]
                weighted_output = out[0] * routing_weights[token_idx, expert_idx, None]
                next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))
            next_states = next_states.view(batch_size, -1, self.hidden_size)
        else:
            hidden_states = hidden_states.repeat(self.num_experts, 1)
            hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
            gate_up = torch.bmm(hidden_states, gate_up_proj)
            gate, up = gate_up.chunk(2, dim=-1)  # not supported for DTensors
            next_states = torch.bmm((up * self.act_fn(gate)), down_proj)
            next_states = next_states.reshape(self.num_experts, batch_size, -1, self.hidden_size)
            next_states = (
                next_states * routing_weights.transpose(0, 1).view(self.num_experts, batch_size, -1)[..., None]
            )
            next_states = next_states.sum(dim=0)
        return next_states

    @staticmethod
    def ep_forward(ep_group, self, hidden_states, routing_weights, router_indices, *args, **kwargs):
        raise NotImplementedError("must set `use_npu_fused_moe=True` when enable expert parallelism.")


class Qwen3VLNpuFusedMoETextExperts(Qwen3VLMoeTextExperts):
    """NPU fusd Moe"""
    def forward(
        self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, router_indices: torch.Tensor, router_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        gate_up_proj, down_proj = self._view_experts_weight()

        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
        permuted_hidden_states, row_ids_map = torch_npu.npu_moe_token_permute(hidden_states, router_indices.to(torch.int32))
        tokens_per_expert = torch.histc(router_indices, bins=self.num_experts, min=0, max=self.num_experts)
        intermediate_hidden_states = npu_group_gemm(permuted_hidden_states, gate_up_proj, tokens_per_expert)
        intermediate_activations = torch_npu.npu_swiglu(intermediate_hidden_states, dim=-1)
        output = npu_group_gemm(intermediate_activations, down_proj, tokens_per_expert)
        next_states = torch_npu.npu_moe_token_unpermute(output, row_ids_map, probs=routing_weights)
        next_states = next_states.view(batch_size, -1, self.hidden_size)
        return next_states

    @staticmethod
    def ep_forward(ep_group, self, hidden_states, routing_weights, router_indices, *args, **kwargs):
        gate_up_proj, down_proj = self._view_experts_weight()
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        hidden_states = fused_ep_forward(
            self.num_experts,
            routing_weights,
            router_indices,
            hidden_states,
            fc1_weight=gate_up_proj,
            fc2_weight=down_proj,
            ep_group=ep_group
        )
        hidden_states = hidden_states.view(batch_size, -1, self.hidden_size)
        return hidden_states


class Qwen3VLMoeTextSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

        self.use_npu_fused_moe = getattr(get_args().mm.model.text_decoder, "use_npu_fused_moe", True)
        if self.use_npu_fused_moe:
            self.experts = Qwen3VLNpuFusedMoETextExperts(config)
        else:
            self.experts = Qwen3VLMoeTextExperts(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = self.gate(hidden_states)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, router_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        hidden_states = hidden_states.reshape(batch_size, -1, self.hidden_size)

        routed_out = self.experts(hidden_states, routing_weights, router_indices, router_logits)
        return routed_out


class Qwen3VLMoeTextDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3VLMoeTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.self_attn = Qwen3VLTextAttention(config, layer_idx)
        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3VLMoeTextSparseMoeBlock(config)
        else:
            self.mlp = Qwen3VLTextMLP(config, intermediate_size=config.intermediate_size)
        self.input_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_size = config.hidden_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        if self.config.synchronize_per_layer:
            torch.npu.current_stream().synchronize()

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class Qwen3VLMoePreTrainedModel(PreTrainedModel):
    config: Qwen3VLMoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3VLMoeTextDecoderLayer", "Qwen3VLVisionBlock"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = False  # MoE models don't work with torch.compile (`torch.where(condition)` not supported)
    _supports_attention_backend = True
    _can_record_outputs = {
        "router_logits": OutputRecorder(nn.Linear, layer_name="mlp.gate", index=0),
        "hidden_states": Qwen3VLMoeTextDecoderLayer,
        "attentions": Qwen3VLTextAttention,
    }

    def _init_weights(self, module):
        """Initialize the weights."""
        super()._init_weights(module)
        if hasattr(self.config, "initializer_range"):
            std = self.config.initializer_range
        else:
            std = getattr(self.config.get_text_config(), "initializer_range", 0.02)
        if isinstance(module, Qwen3VLMoeTextExperts):
            module.gate_up_proj.data.normal_(mean=0.0, std=std)
            module.down_proj.data.normal_(mean=0.0, std=std)


@auto_docstring(
    custom_intro=(
        "Text part of Qwen3VLMoe, "
        "not a pure text-only model, as DeepStack integrates visual features into the early hidden states."
    )
)
class Qwen3VLMoeTextModel(Qwen3VLMoePreTrainedModel, Qwen3VLTextModel):
    config: Qwen3VLMoeTextConfig
    _no_split_modules = ["Qwen3VLMoeTextDecoderLayer"]

    def __init__(self, config: Qwen3VLMoeTextConfig):
        Qwen3VLMoePreTrainedModel.__init__(self, config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # Placeholder for FSDP2 hook registration on norm/gate params when align_fsdp_param_groups is enabled.
        self.norm_hook_module = Qwen3VLEmptyModule()

        self.layers = nn.ModuleList(
            [Qwen3VLMoeTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        if config.activation_offload:
            self.swap_stream = torch.npu.Stream()

        # Initialize weights and apply final processing
        self.post_init()


@auto_docstring
class Qwen3VLMoeModel(Qwen3VLMoePreTrainedModel, Qwen3VLModel):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: Qwen3VLMoeConfig
    _no_split_modules = ["Qwen3VLMoeTextDecoderLayer", "Qwen3VLVisionBlock"]

    def __init__(self, config):
        Qwen3VLMoePreTrainedModel.__init__(self, config)
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        self.language_model = Qwen3VLMoeTextModel._from_config(config.text_config)
        self.rope_deltas = None  # cache rope_deltas here

        # Initialize weights and apply final processing
        self.post_init()


def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        sum_expert_attention_mask = torch.sum(expert_attention_mask, dim=0)
        torch.distributed.all_reduce(
            sum_expert_attention_mask,
            op=torch.distributed.ReduceOp.SUM,
            group=mpu.get_context_parallel_group()
        )
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / sum_expert_attention_mask
        torch.distributed.all_reduce(
            tokens_per_expert,
            op=torch.distributed.ReduceOp.SUM,
            group=mpu.get_context_parallel_group()
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        sum_router_per_expert_attention_mask = torch.sum(router_per_expert_attention_mask, dim=0)
        torch.distributed.all_reduce(
            sum_router_per_expert_attention_mask,
            op=torch.distributed.ReduceOp.SUM,
            group=mpu.get_context_parallel_group()
        )
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / sum_router_per_expert_attention_mask

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


class Qwen3VLMoeForConditionalGeneration(Qwen3VLMoePreTrainedModel, Qwen3VLForConditionalGeneration):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = ["lm_head.weight"]
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: Qwen3VLMoeConfig

    def __init__(self, config):
        Qwen3VLMoePreTrainedModel.__init__(self, config)
        GenerationMixin.__init__(self)
        self.model = Qwen3VLMoeModel(config)
        self.lm_head = Qwen3VLLMHead(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        loss_ctx: Optional[callable] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLMoeCausalLMOutputWithPast]:

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        if loss_ctx:
            logits, loss = self.lm_head(hidden_states[:, slice_indices, :], loss_ctx=loss_ctx)
        else:
            logits, loss = self.lm_head(hidden_states[:, slice_indices, :])
            if labels is not None:
                loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        aux_loss = None
        if kwargs.get("output_router_logits", False):
            if attention_mask is not None:
                attention_mask = split_forward_gather_backward_with_cp(attention_mask, dim=1)
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.config.text_config.num_experts,
                self.config.text_config.num_experts_per_tok,
                attention_mask,
            )

        return Qwen3VLMoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            rope_deltas=outputs.rope_deltas,
        )
