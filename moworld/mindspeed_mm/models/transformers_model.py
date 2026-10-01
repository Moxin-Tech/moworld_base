from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoConfig

from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core import mpu

from mindspeed_mm.data.data_utils.constants import AVG_PER_STEP_TOKEN_NUM
from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.models.common.chunkloss import chunk_loss, calculate_lm_loss, fixed_cross_entropy
from mindspeed_mm.models.common.communications import split_forward_gather_backward_with_cp
from mindspeed_mm.models.transformers.modelhub import ModelHub


class TransformersModel(MultiModalModule):
    """Transformer-based multi-modal model wrapper inherited from MultiModalModule.

    Core wrapper class for initializing, loading and running transformer-based vision-language
    multi-modal models with multiple loss calculation strategies and distributed parallel training support.
    Implements context parallel loss computation, chunk-based memory-efficient loss calculation,
    model sharding and MoE auxiliary loss for large-scale model training.

    Attributes:
        config: Core transformer model configuration parsed from global arguments.
        transformer_config: HuggingFace AutoConfig instance for the underlying transformer model.
        model: Initialized transformer multi-modal model instance.
        loss_compute_mode: Loss calculation mode, supports `default` and `chunk`.
        loss_chunk_size: Chunk size for memory-efficient chunk loss calculation (default: 1024).
        router_aux_loss_coef: Coefficient for MoE model router auxiliary loss (default: 0.0).
    """
    def __init__(self, config) -> None:
        """Initialize the TransformersModel with given configuration and load pretrained weights.

        Args:
            config: General configuration for the multi-modal transformer model,
            the configuration content is derived from model.json.
        """
        super().__init__(config=config)
        args = get_args()

        hf_path = args.mm.model.init_from_hf_path
        trust_remote_code = args.trust_remote_code
        self.config = core_transformer_config_from_args(args)
        self.transformer_config = AutoConfig.from_pretrained(hf_path, trust_remote_code=trust_remote_code)

        model_cls = ModelHub.build(config, self.transformer_config)

        self._set_loss_cfg(args)

        if callable(getattr(model_cls, 'overwrite_transformer_config', None)):
            self.transformer_config = model_cls.overwrite_transformer_config(self.transformer_config)

        if args.init_model_with_meta_device:
            self.model = model_cls._from_config(self.transformer_config).float()
            for m in self.model.modules():
                if getattr(m, "_is_hf_initialized", False):
                    m._is_hf_initialized = False
        else:
            self.model = model_cls.from_pretrained(
                hf_path,
                config=self.transformer_config,
                dtype=torch.float32,
                low_cpu_mem_usage=True,
                device_map="cpu",
                trust_remote_code=trust_remote_code
            )
        print_rank_0("> load model successfully")

        self.model.train()

        if callable(getattr(self.model, 'freeze', None)):
            self.model.freeze(config)

        self.model.use_cache = False

    def forward(
            self,
            input_ids: torch.Tensor,
            pixel_values: Optional[torch.Tensor] = None,
            image_grid_thw: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            *args, **kwargs
    ) -> torch.Tensor:
        loss_dict = {}

        # aux loss (for moe model)
        if self.router_aux_loss_coef > 0.0:
            kwargs["output_router_logits"] = True

        if self.loss_compute_mode == "dynamic_chunk":
            kwargs["total_size"] = self.loss_chunk_size

        if self.loss_compute_mode in ["chunk", "dynamic_chunk"]:
            loss_ctx, loss_mask = self.build_loss_ctx(labels, chunk_size=self.loss_chunk_size, **kwargs)
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                position_ids=position_ids,
                attention_mask=attention_mask,
                cache_position=cache_position,
                use_cache=False,
                loss_ctx=loss_ctx,
                **kwargs
            )
            loss_dict["loss"] = outputs.loss
            loss_dict["loss_mask"] = loss_mask
        else:
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                position_ids=position_ids,
                attention_mask=attention_mask,
                cache_position=cache_position,
                use_cache=False,
                **kwargs
            )
            logits = outputs.logits.contiguous().float()

            loss_ctx, loss_mask = self.build_loss_ctx(labels, chunk_size=None, **kwargs)
            loss_dict["loss"] = loss_ctx(logits)
            loss_dict["loss_mask"] = loss_mask

        if hasattr(outputs, "aux_loss") and self.router_aux_loss_coef > 0:
            loss_dict["loss"] += self.router_aux_loss_coef * outputs.aux_loss
            loss_dict["aux_loss"] = outputs.aux_loss

        return loss_dict

    def fully_shard(
        self,
        process_group,
        fsdp2_config_path,
        **kwargs
    ):
        # If the model has its own 'fully_shard' method, use it directly
        if hasattr(self.model, 'fully_shard') and callable(getattr(self.model, 'fully_shard')):
            return self.model.fully_shard(
                process_group=process_group,
                fsdp2_config_path=fsdp2_config_path,
                **kwargs
            )
        return False

    def calculate_chunk_size(self, batch_size: int, total_size: int) -> int:
        """
        Calculate dynamic Chunk Size to ensure batch_size * chunk_size ≤ total size,
        where chunk_size is the largest power of two not exceeding the theoretical maximum value.

        Args:
            batch_size (int): Input batch size
            total_size (int): Upper limit of total tokens (batch_size * chunk_size),
                typically configured as the maximum token capacity of the device (e.g., 4096/8192 tokens).

        Returns:
            int: Dynamic Chunk Size that meets the requirements, returns 1 by default (when input is invalid)
        """
        if batch_size <= 0 or total_size <= 0:
            print_rank_0(f"[ERROR] Batch size={batch_size} or total size={total_size} must be a positive integer!")
            return 1
        if batch_size >= total_size:
            print_rank_0(f"[ERROR] Batch size={batch_size} exceeds total size={total_size}!")
            return 1

        max_possible_chunk_size = total_size // batch_size

        if max_possible_chunk_size == 0:
            print_rank_0(f"[ERROR] No valid Chunk Size for batch size batch_size={batch_size}!")
            return 1

        max_power_of_two_chunk_size = 1 << (max_possible_chunk_size.bit_length() - 1)

        if max_power_of_two_chunk_size > max_possible_chunk_size:
            max_power_of_two_chunk_size = max_power_of_two_chunk_size >> 1  # Right shift by 1 bit = divide by 2

        return max_power_of_two_chunk_size

    def build_loss_ctx(
        self,
        labels,
        ignore_index=-100,
        chunk_size=1024,
        **kwargs
    ):
        bs = labels.shape[0]
        total_size = kwargs.get("total_size", None)
        if total_size:
            chunk_size = self.calculate_chunk_size(bs, total_size)
            print_rank_0(f"[INFO] Batch size={bs}, chunk size={chunk_size}")
        labels = F.pad(labels, (0, 1), value=ignore_index)
        # Shift labels to match the input sequence for next-token prediction.
        shift_labels = labels[..., 1:].contiguous()

        # Create a mask to identify valid tokens (typically > -1 means non-special tokens)
        loss_mask = shift_labels > -1

        # Retrieve loss_type arguments to determine loss reduction behavior.
        if self.loss_type == "per_sample_loss":
            # Compute per-sample loss: alpha scales each sample by total valid tokens in the batch.
            alpha = loss_mask.sum(1) * loss_mask.shape[0]  # shape: [batch_size]
            reduction = "none"  # Keep per-token losses for sample-wise aggregation.
        elif self.loss_type == "per_token_loss":
            # Use raw sum loss without normalization here;
            avg_per_step_token_num = kwargs.get(AVG_PER_STEP_TOKEN_NUM, None)
            if avg_per_step_token_num is None:
                raise KeyError(f"per_token_loss must use PrefetchGradAccDataLoader")
            torch.distributed.all_reduce(avg_per_step_token_num, op=torch.distributed.ReduceOp.AVG)
            alpha = avg_per_step_token_num
            reduction = "sum"
        elif self.loss_type == "token_loss":
            alpha = loss_mask.sum()
            torch.distributed.all_reduce(alpha, op=torch.distributed.ReduceOp.AVG)
            reduction = "none"
        elif self.loss_type == "square_loss":
            loss_weight = (labels != -100).sum(dim=-1).float()
            loss_weight = 1 / loss_weight.sqrt()
            loss_weight = torch.where(labels != -100, loss_weight.unsqueeze(1), 0.0)
            shift_weights = loss_weight[..., 1:].contiguous().view(-1)
            shift_weight_sum = shift_weights.sum()
            torch.distributed.all_reduce(shift_weight_sum, op=torch.distributed.ReduceOp.AVG)
            alpha = shift_weight_sum / shift_weights
            reduction = "none"
        elif self.loss_type == "default":
            # Default: normalize loss by total number of valid tokens in the batch.
            alpha = loss_mask.sum() # scalar
            reduction = "sum"
        else:
            raise NotImplementedError(f"{self.loss_type} is not implemented!")

        if mpu.get_context_parallel_world_size() > 1:
            shift_labels = split_forward_gather_backward_with_cp(shift_labels, dim=-1, pad_val=ignore_index)
            if self.loss_type == "square_loss":
                alpha = split_forward_gather_backward_with_cp(alpha.view(bs, -1), chunk_size, dim=1).view(-1)

        if chunk_size:
            # Split shifted labels into chunks along the sequence dimension for memory-efficient processing.
            chunk_labels = torch.split(shift_labels, chunk_size, dim=1)

            if self.loss_type == "square_loss":
                alpha = torch.split(alpha.view(bs, -1), chunk_size, dim=1)

            # Prepare keyword arguments for each chunk to be passed to the chunked loss function.
            loss_ctx_kwargs = [
                {
                    "shift_labels": chunk_labels[i],
                    "ignore_index": ignore_index,
                    "reduction": reduction,
                    "alpha": alpha[i].view(-1) if isinstance(alpha, (list, tuple)) else alpha,
                }
                for i in range(len(chunk_labels))
            ]

            # Return a closure that computes the chunked language modeling loss using the prepared config.
            def loss_ctx(hidden_states, head_weight, head_bias):
                return chunk_loss(
                    hidden_states,
                    head_weight,
                    head_bias,
                    loss_forward=calculate_lm_loss,
                    loss_kwargs_chunks=loss_ctx_kwargs,
                    chunk_size=chunk_size
                )

        else:
            def loss_ctx(logits):
                logits = logits.view(-1, logits.shape[-1])
                labels = shift_labels.view(-1)
                return fixed_cross_entropy(
                    logits, labels,
                    alpha=alpha,
                    reduction=reduction
                )

        return loss_ctx, loss_mask

    def _set_loss_cfg(self, args):
        # Retrieve loss configuration from model.json if available
        loss_cfg = getattr(args.mm.model, "loss_cfg", None)
        # loss_cfg param: compute_mode, chunk_size, router_aux_loss_coef
        # compute_mode: default, chunk(use chunk loss)
        # chunk_size: valid when compute mode is set to chunk (default 1024)
        # router_aux_loss_coef: float (use for moe model, default 0.0)
        self.loss_compute_mode = "default"
        self.loss_chunk_size = 1024
        self.router_aux_loss_coef = 0.0
        self.loss_type = "default"
        if loss_cfg is not None:
            self.loss_compute_mode = getattr(loss_cfg, "compute_mode", "default")
            self.loss_type = getattr(loss_cfg, "loss_type", "default")
            if self.loss_compute_mode == "default":
                pass
            elif self.loss_compute_mode == "chunk":
                self.loss_chunk_size = getattr(loss_cfg, "chunk_size", 1024)
            elif self.loss_compute_mode == "dynamic_chunk":
                self.loss_chunk_size = getattr(loss_cfg, "chunk_size", 4096)
            else:
                raise NotImplementedError(f"Unrecognized loss_compute_mode: {self.loss_compute_mode}.")

            if self.loss_type not in ["default", "per_sample_loss", "per_token_loss", "token_loss", "square_loss"]:
                raise NotImplementedError(f"Not implemented loss_type: {self.loss_type}.")

            self.router_aux_loss_coef = getattr(loss_cfg, "router_aux_loss_coef", 0.0)
