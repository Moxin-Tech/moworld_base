# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import types
from typing import Dict, Literal, Optional, Tuple, Union

import torch
from torch import Tensor

from megatron.core import InferenceParams, parallel_state, tensor_parallel, mpu
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.extensions.transformer_engine import TEColumnParallelGroupedLinear, TELayerNormColumnParallelLinear, TERowParallelGroupedLinear, TERowParallelLinear
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.moe.experts import SequentialMLP
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler, get_capacity, save_to_aux_losses_tracker, topk_softmax_with_capacity
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TENorm, TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.global_vars import get_args

from mindspeed.core.context_parallel.ulysses_context_parallel.unaligned_cp.mapping import cal_split_sizes, split_forward_gather_backward, gather_forward_split_backward
from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput
from mindspeed.core.transformer.transformer import norm_recompute_forward
from mindspeed.model.transformer import should_recompute_norm
from mindspeed.utils import set_actual_seq_len

from mindspeed_mm.models.common.mm_gpt_model import MMGPTModel
from mindspeed_mm.models.common.transformer.multi_token_prediction import MultiTokenPredictionBlock, get_mtp_block_spec, tie_output_layer_state_dict, tie_word_embeddings_state_dict
from mindspeed_mm.models.vision.vision_encoders.qwen2vl_vit_model import Qwen2VLRotaryEmbedding_llm
from mindspeed_mm.utils.utils import ensure_valid


def group_limited_topk(
    scores: torch.Tensor,
    topk: int,
    num_tokens: int,
    num_experts: int,
    num_groups: int,
    group_topk: int,
):
    """Perform top-k routing on a subset of expert groups.

    When using group-limited routing:
    1. Experts are divided into 'moe_router_num_groups' equal-sized groups
    2. For each token, 'moe_router_group_topk' groups are selected based on routing scores
       (specifically, the sum of top-2 expert scores within each group)
    3. From these selected groups, 'moe_router_topk' individual experts are chosen
    Two common use cases:
    - Device-limited routing: Set 'moe_router_num_groups' equal to expert parallel size (EP)
      to limit each token to experts on a subset of devices

    - Node-limited routing: Set 'moe_router_num_groups' equal to number of nodes in EP group
      to limit each token to experts on a subset of nodes


    Args:
        scores (torch.Tensor): Softmax scores from the router.
        topk (int): The number of experts to select for each token.
        num_tokens (int): The number of tokens.
        num_experts (int): The number of experts.
        num_groups (int): Number of groups for routed experts.
        group_topk (int): Number of groups selected for each token.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Probs and indices tensor.
    """

    # Organize the experts into groups
    group_scores = scores.view(num_tokens, num_groups, -1).topk(2, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=group_topk, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)

    # Mask the experts based on selection groups
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, num_groups, num_experts // num_groups)
        .reshape(num_tokens, -1)
    )

    masked_scores = scores.masked_fill(~score_mask.bool(), float('-inf'))
    probs, top_indices = torch.topk(masked_scores, k=topk, dim=-1)

    return probs, top_indices


class MOETopKRouter(TopKRouter):
    def __init__(self, config):
        super().__init__(config)
        args = get_args()
        self.n_group = getattr(config, "n_group", args.expert_model_parallel_size)
        self.topk_group = getattr(config, "topk_group", None)
        self.norm_topk_prob = getattr(config, "norm_topk_prob", False)
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", None)
        self.score_function = getattr(config, "moe_router_score_function", "softmax")
        self.enable_expert_bias = getattr(config, "moe_router_enable_expert_bias", False)
        self.moe_router_topk_scaling_factor = getattr(config, "routed_scaling_factor", None)

        if self.enable_expert_bias:
            self.register_buffer(
                'local_tokens_per_expert',
                torch.zeros(self.config.num_moe_experts, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                'expert_bias', torch.zeros(self.config.num_moe_experts, dtype=torch.float32)
            )
        else:
            self.local_tokens_per_expert = None
            self.expert_bias = None

    def gating(self, input_tensor):
        if self.config.router_gating_in_fp32:
            def to_fp32(_input, weight):
                return _input.type(torch.float32), weight.type(torch.float32)
            self.fp32_checkpoint_manager = CheckpointWithoutOutput()
            input_tensor, weight = self.fp32_checkpoint_manager.checkpoint(to_fp32, False, input_tensor, self.weight)
            logits = torch.nn.functional.linear(input_tensor, weight)
            self.fp32_checkpoint_manager.discard_output()
            if logits.requires_grad:
                logits.register_hook(self.fp32_checkpoint_manager.recompute)
        else:
            logits = torch.nn.functional.linear(input_tensor, self.weight)

        return logits

    def routing(self, logits):
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Probs and the indices tensor.
        """
        args = get_args()

        logits = logits.view(-1, self.config.num_moe_experts)

        # Apply Z-Loss
        logits = self.apply_z_loss(logits)

        if (
            self.config.tensor_model_parallel_size > 1
            and self.config.moe_token_dispatcher_type == "alltoall"
        ):
            # Gather the logits from the TP region
            logits = gather_from_sequence_parallel_region(logits)

        if self.routing_type == "sinkhorn":
            scores, indices = self.sinkhorn_load_balancing(logits)
        elif self.routing_type == "aux_loss":
            scores, indices = self.aux_loss_load_balancing(logits)
        elif self.routing_type in ["none", "noaux_tc"]:
            # A naive top-k routing without load balancing
            scores, indices, tokens_per_expert = topk_softmax_with_capacity(
                logits,
                self.topk,
                capacity_factor=self.config.moe_expert_capacity_factor,
                pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
                drop_policy=self.config.moe_token_drop_policy,
                use_pre_softmax=self.config.moe_router_pre_softmax,
                num_groups=self.n_group,
                group_topk=self.topk_group,
                scaling_factor=self.moe_router_topk_scaling_factor,
                deterministic_mode=self.config.deterministic_mode,
                score_function=self.score_function,
                expert_bias=self.expert_bias,
            )
        else:
            raise ValueError(f"Unsupported MoE routing type: {self.routing_type}")

        # Prevent extra local tokens accumulation on evaluation or activation recomputation
        if self.enable_expert_bias and torch.is_grad_enabled():
            with torch.no_grad():
                self.local_tokens_per_expert += tokens_per_expert

        return scores, indices


class DeepSeekMoELayer(MoELayer):
    def __init__(self, config, submodules=None, layer_number=None):
        super().__init__(config=config, submodules=submodules, layer_number=layer_number)
        self.router = MOETopKRouter(config=self.config)


def get_mlp_module_spec(
    use_te=True, num_experts=None, moe_grouped_gemm=False, use_shared_experts=None
) -> ModuleSpec:
    if num_experts is None:
        # Dense MLP w/ or w/o TE modules.
        return ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=TELayerNormColumnParallelLinear if use_te else ColumnParallelLinear,
                linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
            ),
        )
    else:
        # Mixture of experts with modules in megatron core.
        if use_te and moe_grouped_gemm:
            linear_fc1 = TEColumnParallelGroupedLinear
            linear_fc2 = TERowParallelGroupedLinear
        else:
            linear_fc1 = ColumnParallelLinear
            linear_fc2 = RowParallelLinear

        use_te_grouped_gemm = use_te and TEColumnParallelGroupedLinear is not None

        if use_shared_experts is not None:
            shared_experts = ModuleSpec(module=SharedExpertMLP,
                                        params={"gate": False},
                                        submodules=MLPSubmodules(
                                            linear_fc1=linear_fc1,
                                            linear_fc2=linear_fc2,)
                                        )
        else:
            shared_experts = None

        return ModuleSpec(
            module=DeepSeekMoELayer,
            submodules=(
                MoESubmodules(
                    experts=ModuleSpec(
                        module=SequentialMLP,
                        submodules=MLPSubmodules(
                            linear_fc1=linear_fc1,
                            linear_fc2=linear_fc2,
                        )
                    ),
                    shared_experts=shared_experts
                )
                if not moe_grouped_gemm or use_te_grouped_gemm
                else None
            ),
        )


class MOETransformerBlock(TransformerBlock):
    def _build_layers(self):
        args = get_args()
        use_te = False

        def build_layer(layer_spec, layer_number):
            ffn_hidden_size = self.config.ffn_hidden_size
            # For deepseek
            if (
                    self.config.num_moe_experts
                    and self.config.first_k_dense_replace is not None
                    and self.config.moe_layer_freq is not None
            ):
                use_shared_experts = self.config.n_shared_experts
                if (
                        (layer_number - 1) >= self.config.first_k_dense_replace
                        and (layer_number - 1) % self.config.moe_layer_freq == 0
                ):
                    self.config.ffn_hidden_size = self.config.moe_intermediate_size
                    layer_spec.submodules.mlp = get_mlp_module_spec(use_te=use_te, num_experts=self.config.num_moe_experts,
                                                                    moe_grouped_gemm=self.config.moe_grouped_gemm,
                                                                    use_shared_experts=use_shared_experts)
                else:
                    layer_spec.submodules.mlp = get_mlp_module_spec(use_te=use_te, moe_grouped_gemm=self.config.moe_grouped_gemm,
                                                                    use_shared_experts=use_shared_experts)
            model = build_module(layer_spec, config=self.config, layer_number=layer_number)
            self.config.ffn_hidden_size = ffn_hidden_size
            return model

        # offset is implicit in TransformerLayer
        self.layers = torch.nn.ModuleList(
            [
                build_layer(layer_spec, i + 1)
                for i, layer_spec in enumerate(self.submodules.layer_specs)
            ]
        )

        # mtp require separate layernorms for main model and mtp modules, thus move finalnorm out of block
        has_layernorm = self.post_layer_norm and self.submodules.layer_norm
        mtp_process = hasattr(self.config, "mtp_num_layers") and self.config.mtp_num_layers
        if self.post_process and has_layernorm and not mtp_process:
            self.final_layernorm = build_module(
                self.submodules.layer_norm,
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.final_layernorm = None  # Either this or nn.Identity

        # For recompute norm
        if args.recompute_norm:
            for layer in self.layers:
                # 1F1B overlap has its own implementation for recompute_norm
                if should_recompute_norm(layer) and not args.moe_fb_overlap:
                    layer.forward = types.MethodType(norm_recompute_forward, layer)


class MOEModel(MMGPTModel):
    """MOEModel Transformer language model.

    Args:
        config (TransformerConfig): Transformer config
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers
        vocab_size (int): Vocabulary size
        max_sequence_length (int): maximum size of sequence. This is used for positional embedding
        pre_process (bool, optional): Include embedding layer (used with pipeline parallelism). Defaults to True.
        post_process (bool, optional): Include an output layer (used with pipeline parallelism). Defaults to True.
        fp16_lm_cross_entropy (bool, optional): Defaults to False.
        parallel_output (bool, optional): Do not gather the outputs, keep them split across tensor parallel ranks. Defaults to True.
        share_embeddings_and_output_weights (bool, optional): When True, input embeddings and output logit weights are shared. Defaults to False.
        position_embedding_type (Literal[learned_absolute,rope], optional):  Position embedding type.. Defaults to 'learned_absolute'.
        rotary_percent (float, optional): Percent of rotary dimension to use for rotary position embeddings. Ignored unless position_embedding_type is 'rope'. Defaults to 1.0.
        rotary_base (int, optional): Base period for rotary position embeddings. Ignored unless position_embedding_type is 'rope'. Defaults to 10000.
        seq_len_interpolation_factor (Optional[float], optional): scale of linearly interpolating RoPE for longer sequences. The value must be a float larger than 1.0. Defaults to None.
    """

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal['mrope', 'rope'] = 'rope',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        seq_len_interpolation_factor: Optional[float] = None,
    ) -> None:
        super(LanguageModule, self).__init__(config=config)

        args = get_args()
        self.transformer_layer_spec: ModuleSpec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type = position_embedding_type
        self.mtp_process = hasattr(config, "mtp_num_layers") and config.mtp_num_layers

        # megatron core pipelining currently depends on model type
        self.model_type = ModelType.encoder_or_decoder

        # These 2 attributes are needed for TensorRT-LLM export.
        self.max_position_embeddings = max_sequence_length
        self.rotary_percent = rotary_percent

        if self.pre_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
            )

        if self.position_embedding_type == 'mrope':
            if getattr(config, 'mrope_section', None) is None:
                raise AssertionError('mrope section should be provided for mrope!')
            self.rotary_pos_emb = Qwen2VLRotaryEmbedding_llm(config=config)
        elif self.position_embedding_type == 'rope':
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=args.qk_rope_head_dim if args.qk_rope_head_dim else self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
            )

        # Transformer.
        self.decoder = MOETransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )

        if self.post_process and self.mtp_process:
            mtp_block_spec = get_mtp_block_spec(
                config=self.config,
                transformer_layer_spec=transformer_layer_spec,
                use_transformer_engine=False,
            )
            self.mtp = MultiTokenPredictionBlock(
                config=self.config,
                spec=mtp_block_spec,
            )
            # move block main model final norm here when mtp enable
            self.final_layernorm = build_module(
                    TENorm,
                    config=self.config,
                    hidden_size=self.config.hidden_size,
                    eps=self.config.layernorm_epsilon,
                )
        else:
            self.final_layernorm = None

        # Output
        if post_process:
            if self.config.defer_embedding_wgrad_compute:
                # The embedding activation buffer preserves a reference to the input activations
                # of the final embedding projection layer GEMM. It will hold the activations for
                # all the micro-batches of a global batch for the last pipeline stage. Once we are
                # done with all the back props for all the microbatches for the last pipeline stage,
                # it will be in the pipeline flush stage. During this pipeline flush we use the
                # input activations stored in embedding activation buffer and gradient outputs stored
                # in gradient buffer to calculate the weight gradients for the embedding final linear layer.
                self.embedding_activation_buffer = []
                self.grad_output_buffer = []
            else:
                self.embedding_activation_buffer = None
                self.grad_output_buffer = None

            self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.pre_process
                and self.share_embeddings_and_output_weights,
                embedding_activation_buffer=self.embedding_activation_buffer,
                grad_output_buffer=self.grad_output_buffer,
            )

        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        if not len(input_tensor) == 1:
            raise AssertionError('input_tensor should only be length 1 for gpt/bert')
        self.decoder.set_input_tensor(input_tensor[0])

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
    ) -> Tensor:
        """Forward function of the GPT Model This function passes the input tensors
        through the embedding layer, and then the decoeder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units
        """
        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.

        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        if mpu.get_context_parallel_world_size() > 1:
            split_gather_sizes = cal_split_sizes(decoder_input.shape[0], mpu.get_context_parallel_world_size())
            decoder_input = split_forward_gather_backward(decoder_input, mpu.get_context_parallel_group(), 0,
                                                        split_gather_sizes, "down")
            input_ids = split_forward_gather_backward(input_ids, mpu.get_context_parallel_group(), 1,
                                                        split_gather_sizes, "down")
            position_ids = split_forward_gather_backward(position_ids, mpu.get_context_parallel_group(), 2,
                                                        split_gather_sizes, "down")

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        if self.position_embedding_type == 'mrope':
            param_dtype = torch.bfloat16
            if not getattr(self.config, 'bf16', False):
                raise AssertionError('mrope only support bf16 now!')
            rotary_pos_emb = self.rotary_pos_emb(input_ids.device, param_dtype, position_ids)
        elif self.position_embedding_type == 'rope':
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params, self.decoder, decoder_input, self.config, None
            )
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

        if getattr(self.config, 'use_remove_padding', False):
            if position_ids is not None and position_ids.dim() == 3:
                position_ids_fa = position_ids[0]
            position_ids_fa = position_ids_fa.flatten()
            indices_q = torch.arange(position_ids_fa.size(0), device=position_ids_fa.device, dtype=torch.int32)
            cu_seqlens = torch.cat(
                (
                    indices_q[position_ids_fa == 0],
                    torch.tensor(position_ids_fa.size(), device=position_ids_fa.device, dtype=torch.int32),
                )
            )
            set_actual_seq_len(tuple(cu_seqlens[1:].cpu().numpy().tolist()))

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )

        if mpu.get_context_parallel_world_size() > 1:
            hidden_states = gather_forward_split_backward(hidden_states, mpu.get_context_parallel_group(), 0,
                                                        split_gather_sizes, "up")

        if not self.post_process:
            return hidden_states

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        if self.mtp_process:
            hidden_states = self.mtp(
                input_ids=input_ids,
                position_ids=position_ids,
                labels=labels,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                packed_seq_params=packed_seq_params,
                embedding=self.embedding,
                output_layer=self.output_layer,
                output_weight=output_weight,
                compute_language_model_loss=self.compute_language_model_loss,
                **(extra_block_kwargs or {}),
            )

        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)

        logits, _ = self.output_layer(hidden_states, weight=output_weight)

        if labels is None:
            return logits.transpose(0, 1).contiguous(), None

        loss = self.compute_language_model_loss(labels, logits)

        return logits.transpose(0, 1).contiguous(), loss

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[Dict] = None
    ) -> ShardedStateDict:
        """ Sharded state dict implementation for GPTModel backward-compatibility (removing extra state).

        Args:
            prefix (str): Module name prefix.
            sharded_offsets (tuple): PP related offsets, expected to be empty at this module level.
            metadata (Optional[Dict]): metadata controlling sharded state dict creation.

        Returns:
            ShardedStateDict: sharded state dict for the GPTModel
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        output_layer_extra_state_key = f'{prefix}output_layer._extra_state'

        # Old GPT checkpoints only stored the output layer weight key. So we remove the _extra_state key
        # but check that it doesn't contain any data anyway
        output_extra_state = sharded_state_dict.pop(output_layer_extra_state_key, None)
        ensure_valid(not (
            output_extra_state and output_extra_state.data
        ), f'Expected output layer extra state to be empty, got: {output_extra_state}')

        return sharded_state_dict

    def shared_embedding_or_output_weight(self) -> Tensor:
        """Gets the embedding weight or output logit weights when share input embedding and
        output weights set to True or when use Multi-Token Prediction (MTP) feature.

        Returns:
            Tensor: During pre processing or MTP process it returns the input embeddings weight.
            Otherwise, during post processing it returns the final output layers weight.
        """
        if not self.pre_process and self.post_process and get_args().schedules_method == 'dualpipev':
            from mindspeed.core.pipeline_parallel.dualpipev.dualpipev_schedules import \
                get_shared_embedding_from_dual_chunk
            return get_shared_embedding_from_dual_chunk()
        if self.pre_process or self.mtp_process:
            # Multi-Token Prediction (MTP) need both embedding layer and output layer.
            # So there will be both embedding layer and output layer in the mtp process stage.
            # In this case, if share_embeddings_and_output_weights is True, the shared weights
            # will be stored in embedding layer, and output layer will not have any weight.
            if not hasattr(self, 'embedding'):
                raise AssertionError(f"embedding is needed in this pipeline stage, but it is not initialized.")
            return self.embedding.word_embeddings.weight
        elif self.post_process:
            return self.output_layer.weight
        return None

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[Dict] = None
    ) -> ShardedStateDict:
        """Sharded state dict implementation for GPTModel backward-compatibility.
        Removing extra state.
        Tie word embeddings and output layer in mtp process stage.

        Args:
            prefix (str): Module name prefix.
            sharded_offsets (tuple): PP related offsets, expected to be empty at this module level.
            metadata (Optional[Dict]): metadata controlling sharded state dict creation.

        Returns:
            ShardedStateDict: sharded state dict for the GPTModel
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        # Multi-Token Prediction (MTP) need both embedding layer and output layer in
        # mtp process stage.
        # If MTP is not placed in the pre processing stage, we need to maintain a copy of
        # embedding layer in the mtp process stage and tie it to the embedding in the pre
        # processing stage.
        # Also, if MTP is not placed in the post processing stage, we need to maintain a copy
        # of output layer in the mtp process stage and tie it to the output layer in the post
        # processing stage.
        if self.mtp_process and not self.pre_process:
            emb_weight_key = f'{prefix}embedding.word_embeddings.weight'
            emb_weight = self.embedding.word_embeddings.weight
            tie_word_embeddings_state_dict(sharded_state_dict, emb_weight, emb_weight_key)
        if self.mtp_process and not self.post_process:
            # We only need to tie the output layer weight if share_embeddings_and_output_weights
            # is False. Because if share_embeddings_and_output_weights is True, the shared weight
            # will be stored in embedding layer, and output layer will not have any weight.
            if not self.share_embeddings_and_output_weights:
                output_layer_weight_key = f'{prefix}output_layer.weight'
                output_layer_weight = self.output_layer.weight
                tie_output_layer_state_dict(
                    sharded_state_dict, output_layer_weight, output_layer_weight_key
                )

        return sharded_state_dict
