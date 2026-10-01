# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
from typing import Union, List, Optional

import torch
from torch import Tensor
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules, TransformerBlock
from megatron.core.transformer import TransformerLayer, TransformerLayerSubmodules
from megatron.core import InferenceParams, parallel_state, tensor_parallel, mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.training import get_args
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_viewless_tensor
from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.models.vision.projectors.multimodal_projector import MultimodalProjector


class VisionTransformerBlock(TransformerBlock):
    """Transformer class."""

    def __init__(
        self,
        config: TransformerConfig,
        spec: Union[TransformerBlockSubmodules, ModuleSpec],
        post_layer_norm: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
    ):
        super().__init__(config=config, spec=spec, post_layer_norm=post_layer_norm, pre_process=pre_process, post_process=post_process)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor = None,
        context_mask: Tensor = None,
        rotary_pos_emb: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        # hidden_states (float): [s, b, h]
        # attention_mask (bool): [1, 1, s, s]

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
        hidden_states = make_viewless_tensor(
            inp=hidden_states, requires_grad=True, keep_graph=True,
        )

        # Forward pass.
        if self.config.recompute_granularity == 'full' and self.training:
            hidden_states = self._checkpointed_forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                packed_seq_params=packed_seq_params,
            )
        else:
            for layer in self.layers:
                with self.offload_context:
                    hidden_states, context = layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        inference_params=inference_params,
                        packed_seq_params=packed_seq_params,
                    )

        # Final layer norm.
        if self.post_process and self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


class Qwen3VLVisionTransformerLayer(TransformerLayer):
    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
    ):
        super().__init__(config, submodules, layer_number, hidden_dropout)
        deepstack_visual_indexes = config.deepstack_visual_indexes
        if (layer_number - 1) in deepstack_visual_indexes:
            self.deepstack_layer = MultimodalProjector(
                config=config.projector_config,
                submodules=submodules.deepstack_layer.submodules,
                use_postshuffle_norm=True
            )
        else:
            self.deepstack_layer = None

    def forward(self, *args, **kwargs):
        """
        Perform a forward pass through the transformer layer.

        This method calls the core computation of a transformer layer, including
        self-attention, cross-attention (if applicable), and feed-forward operations.
        """
        output, context = super().forward(*args, **kwargs)
        if self.deepstack_layer is not None:
            deepstack_feature = self.deepstack_layer(output)
        else:
            deepstack_feature = None
        return output, context, deepstack_feature


class Qwen2VLVisionTransformerBlock(TransformerBlock):
    """Transformer class."""

    def __init__(
            self,
            config: TransformerConfig,
            spec: Union[TransformerBlockSubmodules, ModuleSpec],
            post_layer_norm: bool = True,
            pre_process: bool = True,
            post_process: bool = True,
    ):
        super().__init__(config=config, spec=spec, post_layer_norm=post_layer_norm, pre_process=pre_process,
                         post_process=post_process)

    def _checkpointed_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor,
        context_mask: Tensor,
        rotary_pos_emb: Tensor,
        packed_seq_params: PackedSeqParams,
        window_packed_seq_params: PackedSeqParams,
        full_packed_seq_params: PackedSeqParams,
        fullatt_block_indexes_now: List[int] = None,
    ):
        """Forward method with activation checkpointing."""

        def custom(start: int, end: int):
            def custom_forward(
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
                packed_seq_params=packed_seq_params,
            ):
                for index in range(start, end):
                    layer = self._get_layer(index)
                    current_mask = attention_mask
                    if len(fullatt_block_indexes_now) > 0 and index not in fullatt_block_indexes_now:
                        packed_seq_params = window_packed_seq_params
                    else:
                        packed_seq_params = full_packed_seq_params

                    if get_args().use_flash_attn is False:
                        packed_seq_params = None

                    hidden_states, context = layer(
                        hidden_states=hidden_states,
                        attention_mask=current_mask,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        inference_params=None,
                        packed_seq_params=packed_seq_params,
                    )
                return hidden_states, context

            return custom_forward

        def checkpoint_handler(forward_func):
            return tensor_parallel.checkpoint(
                forward_func,
                self.config.distribute_saved_activations,
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
            )

        if self.config.recompute_method == 'uniform':
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            l = 0
            while l < self.num_layers_per_pipeline_rank:
                hidden_states, context = checkpoint_handler(
                    custom(l, l + self.config.recompute_num_layers)
                )

                l += self.config.recompute_num_layers

        elif self.config.recompute_method == 'block':
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            recompute_skip_num_layers = 0
            for l in range(self.num_layers_per_pipeline_rank):
                # Skip recomputation when input grad computation is not needed.
                # Need to have at least one input tensor with gradient computation
                # for re-enterant autograd engine.
                if self.config.fp8 and not hidden_states.requires_grad:
                    recompute_skip_num_layers += 1
                if (
                    l >= recompute_skip_num_layers
                    and l < self.config.recompute_num_layers + recompute_skip_num_layers
                ):
                    hidden_states, context = checkpoint_handler(custom(l, l + 1))
                else:
                    hidden_states, context = custom(l, l + 1)(
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
            hidden_states: Tensor,
            attention_mask: Tensor,
            window_mask=None,
            cu_seqlens=None,
            cu_window_seqlens=None,
            context: Tensor = None,
            context_mask: Tensor = None,
            rotary_pos_emb: Tensor = None,
            inference_params: InferenceParams = None,
            packed_seq_params: PackedSeqParams = None,
    ):
        # hidden_states (float): [s, b, h]
        # attention_mask (bool): [1, 1, s, s]

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
        hidden_states = make_viewless_tensor(
            inp=hidden_states, requires_grad=True, keep_graph=True,
        )

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        fullatt_block_indexes_now = []
        if getattr(self.config, "window_attn_size", None) is not None:
            pp_rank = mpu.get_pipeline_model_parallel_rank()
            vp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
            if vp_rank:
                previous_layer = sum(sum(row[:pp_rank]) for row in self.config.pipeline_num_layers[:vp_rank]) + sum(
                    self.config.pipeline_num_layers[vp_rank][:pp_rank])
            else:
                if isinstance(self.config.pipeline_num_layers[0], list):
                    previous_layer = sum(self.config.pipeline_num_layers[0][:pp_rank])
                else:
                    previous_layer = sum(self.config.pipeline_num_layers[:pp_rank])
            for x in self.config.fullatt_block_indexes:
                fullatt_block_indexes_now.append(x - previous_layer)

        window_packed_seq_params = None
        full_packed_seq_params = None
        if get_args().use_flash_attn and packed_seq_params is None:
            if cu_window_seqlens is not None:
                if self.config.context_parallel_size == 1:
                    cu_window_seqlens = cu_window_seqlens.tolist()
                window_packed_seq_params = PackedSeqParams(cu_seqlens_q=cu_window_seqlens, cu_seqlens_kv=cu_window_seqlens)
            if cu_seqlens is not None:
                cu_full_seqlens = cu_seqlens
                if self.config.context_parallel_size == 1:
                    cu_full_seqlens = cu_full_seqlens.tolist()
                full_packed_seq_params = PackedSeqParams(cu_seqlens_q=cu_full_seqlens, cu_seqlens_kv=cu_full_seqlens)

        with rng_context:
            # Forward pass.
            if self.config.recompute_granularity == 'full' and self.training:
                hidden_states = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    packed_seq_params=packed_seq_params,
                    window_packed_seq_params=window_packed_seq_params,
                    full_packed_seq_params=full_packed_seq_params,
                    fullatt_block_indexes_now=fullatt_block_indexes_now,
                )
            else:
                for layer_num, layer in enumerate(self.layers):
                    with self.offload_context:
                        if getattr(self.config, "window_attn_size", None) is not None:
                            if layer_num in fullatt_block_indexes_now:
                                attention_mask_now = attention_mask
                                packed_seq_params = full_packed_seq_params
                            else:
                                attention_mask_now = window_mask
                                packed_seq_params = window_packed_seq_params
                        else:
                            attention_mask_now = attention_mask
                            packed_seq_params = full_packed_seq_params

                        if get_args().use_flash_attn is False:
                            packed_seq_params = None

                        hidden_states, context = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask_now,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            inference_params=inference_params,
                            packed_seq_params=packed_seq_params,
                        )

                    if (
                            torch.is_grad_enabled()
                            and self.config.cpu_offloading
                            and self.group_prefetch_offload_commit_async is not None
                    ):
                        hidden_states = self.group_prefetch_offload_commit_async(hidden_states)
        # Final layer norm.
        if self.post_process and self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


class Qwen3VLVisionTransformerBlock(TransformerBlock):
    """Transformer class."""

    def __init__(
            self,
            config: TransformerConfig,
            spec: Union[TransformerBlockSubmodules, ModuleSpec],
            post_layer_norm: bool = True,
            pre_process: bool = True,
            post_process: bool = True,
    ):
        super().__init__(config=config, spec=spec, post_layer_norm=post_layer_norm, pre_process=pre_process,
                         post_process=post_process)

    def _checkpointed_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor,
        context_mask: Tensor,
        rotary_pos_emb: Tensor,
        packed_seq_params: PackedSeqParams,
        window_packed_seq_params: PackedSeqParams,
        full_packed_seq_params: PackedSeqParams,
        fullatt_block_indexes_now: List[int] = None,
    ):
        """Forward method with activation checkpointing."""

        def custom(start: int, end: int):
            def custom_forward(
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
                packed_seq_params=packed_seq_params,
            ):
                for index in range(start, end):
                    layer = self._get_layer(index)
                    current_mask = attention_mask
                    if len(fullatt_block_indexes_now) > 0 and index not in fullatt_block_indexes_now:
                        packed_seq_params = window_packed_seq_params
                    else:
                        packed_seq_params = full_packed_seq_params

                    if get_args().use_flash_attn is False:
                        packed_seq_params = None

                    hidden_states, context, deepstack_feature = layer(
                        hidden_states=hidden_states,
                        attention_mask=current_mask,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        inference_params=None,
                        packed_seq_params=packed_seq_params,
                    )
                return hidden_states, context, deepstack_feature

            return custom_forward

        def checkpoint_handler(forward_func):
            return tensor_parallel.checkpoint(
                forward_func,
                self.config.distribute_saved_activations,
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
            )

        deepstack_feature_lists = []
        if self.config.recompute_method == 'uniform':
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            l = 0
            while l < self.num_layers_per_pipeline_rank:
                hidden_states, context = checkpoint_handler(
                    custom(l, l + self.config.recompute_num_layers)
                )

                l += self.config.recompute_num_layers

        elif self.config.recompute_method == 'block':
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            recompute_skip_num_layers = 0
            for l in range(self.num_layers_per_pipeline_rank):
                # Skip recomputation when input grad computation is not needed.
                # Need to have at least one input tensor with gradient computation
                # for re-enterant autograd engine.
                if self.config.fp8 and not hidden_states.requires_grad:
                    recompute_skip_num_layers += 1
                if (
                    l >= recompute_skip_num_layers
                    and l < self.config.recompute_num_layers + recompute_skip_num_layers
                ):
                    hidden_states, context, deepstack_feature = checkpoint_handler(custom(l, l + 1))
                else:
                    hidden_states, context, deepstack_feature = custom(l, l + 1)(
                        hidden_states,
                        attention_mask,
                        context,
                        context_mask,
                        rotary_pos_emb,
                    )
                if deepstack_feature is not None:
                    deepstack_feature_lists.append(deepstack_feature)
        elif self.config.recompute_method == 'custom':
            if len(set(self.config.deepstack_visual_indexes) - set(self.config.custom_recompute_split)) != 0:
                raise ValueError(f"when using custom recompute, deepstack_visual_indexes should be included in custom_recompute_split, \
                                 but got {self.config.deepstack_visual_indexes} and {self.config.custom_recompute_split}")
            if len(self.layers) != self.config.custom_recompute_split[-1]:
                raise ValueError(f"when using custom recompute, num_layers should == custom_recompute_split[-1], \
                                 but got {self.config.num_layers} and {self.config.custom_recompute_split[-1]}")
            bound = [0] + self.config.custom_recompute_split
            recompute_split = [(bound[i], bound[i + 1]) for i in range(len(bound) - 1)]
            for l_start, l_end in recompute_split:
                hidden_states, _, deepstack_feature = checkpoint_handler(custom(l_start, l_end))
                if deepstack_feature is not None:
                    deepstack_feature_lists.append(deepstack_feature)
        else:
            raise ValueError("Invalid activation recompute method.")

        return hidden_states, deepstack_feature_lists

    def forward(
            self,
            hidden_states: Tensor,
            attention_mask: Tensor,
            window_mask=None,
            cu_seqlens=None,
            cu_window_seqlens=None,
            context: Tensor = None,
            context_mask: Tensor = None,
            rotary_pos_emb: Tensor = None,
            inference_params: InferenceParams = None,
            packed_seq_params: PackedSeqParams = None,
    ):
        # hidden_states (float): [s, b, h]
        # attention_mask (bool): [1, 1, s, s]

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
        hidden_states = make_viewless_tensor(
            inp=hidden_states, requires_grad=True, keep_graph=True,
        )

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        fullatt_block_indexes_now = []
        if getattr(self.config, "window_attn_size", None) is not None:
            pp_rank = mpu.get_pipeline_model_parallel_rank()
            vp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
            if vp_rank:
                previous_layer = sum(sum(row[:pp_rank]) for row in self.config.pipeline_num_layers[:vp_rank]) + sum(
                    self.config.pipeline_num_layers[vp_rank][:pp_rank])
            else:
                previous_layer = sum(self.config.pipeline_num_layers[:pp_rank])
            for x in self.config.fullatt_block_indexes:
                fullatt_block_indexes_now.append(x - previous_layer)

        window_packed_seq_params = None
        full_packed_seq_params = None
        if get_args().use_flash_attn and packed_seq_params is None:
            if cu_window_seqlens is not None:
                cu_window_seqlens = cu_window_seqlens[1:]
                if self.config.context_parallel_size == 1:
                    cu_window_seqlens = cu_window_seqlens.tolist()
                window_packed_seq_params = PackedSeqParams(cu_seqlens_q=cu_window_seqlens, cu_seqlens_kv=cu_window_seqlens)
            if cu_seqlens is not None:
                cu_full_seqlens = cu_seqlens[1:]
                if self.config.context_parallel_size == 1:
                    cu_full_seqlens = cu_full_seqlens.tolist()
                full_packed_seq_params = PackedSeqParams(cu_seqlens_q=cu_full_seqlens, cu_seqlens_kv=cu_full_seqlens)
        deepstack_feature_lists = [] # NOTE: No recompute for now
        with rng_context:
            # Forward pass.
            if self.config.recompute_granularity == 'full' and self.training:
                hidden_states, deepstack_feature_lists = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    packed_seq_params=packed_seq_params,
                    window_packed_seq_params=window_packed_seq_params,
                    full_packed_seq_params=full_packed_seq_params,
                    fullatt_block_indexes_now=fullatt_block_indexes_now,
                )
            elif self.config.recompute_granularity == 'selective' and self.training:
                hidden_states, deepstack_feature_lists = self._checkpointed_forward(
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                    packed_seq_params,
                    window_packed_seq_params,
                    full_packed_seq_params,
                    fullatt_block_indexes_now
                )
            else:
                for layer_num, layer in enumerate(self.layers):
                    with self.offload_context:
                        if getattr(self.config, "window_attn_size", None) is not None:
                            if layer_num in fullatt_block_indexes_now:
                                attention_mask_now = attention_mask
                                packed_seq_params = full_packed_seq_params
                            else:
                                attention_mask_now = window_mask
                                packed_seq_params = window_packed_seq_params
                        else:
                            attention_mask_now = attention_mask
                            packed_seq_params = full_packed_seq_params

                        if get_args().use_flash_attn is False:
                            packed_seq_params = None

                        output_tensors = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask_now,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            inference_params=inference_params,
                            packed_seq_params=packed_seq_params,
                        )
                        if len(output_tensors) == 3:
                            hidden_states, context, deepstack_feature = output_tensors
                            if deepstack_feature is not None:
                                deepstack_feature_lists.append(deepstack_feature)
                        else:
                            hidden_states, context = output_tensors

                    if (
                            torch.is_grad_enabled()
                            and self.config.cpu_offloading
                            and self.group_prefetch_offload_commit_async is not None
                    ):
                        hidden_states = self.group_prefetch_offload_commit_async(hidden_states)
        # Final layer norm.
        if self.post_process and self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states, deepstack_feature_lists
