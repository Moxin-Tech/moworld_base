# pylint: skip-file
import torch
from packaging import version
import transformers
if version.parse(transformers.__version__) >= version.parse("5.2.0"):
    from transformers.utils.output_capturing import _CAN_RECORD_REGISTRY

from mindspeed.fsdp.utils.str_match import module_name_match
from ..params.feature_args import FeatureArguments
from ..params.parallel_args import ParallelArguments
from ..features.memory.async_offload import async_offload_modules, get_offload_modules
from ..features.memory.chunkloss.chunkloss_lm_head import apply_chunkloss_module, get_chunkloss_module
from ..features.communication.chunk_mbs import get_chunkmbs_modules, apply_chunkmbs_module
from ..features.memory.recompute import recompute_modules


class FeaturesApplier:
    def __init__(self, feature_config: FeatureArguments):
        self.config = feature_config

    def get_needed_modules(self, modules, plan):
        matched_submodules = []
        for plan_name in plan:
            for name, module in modules.named_modules():
                if module_name_match(plan_name, name):
                    if (name, module) not in matched_submodules:
                        matched_submodules.append((name, module))
        return matched_submodules

    def apply_recompute_models(self, model):
        if not getattr(self.config, "recompute", False) or not getattr(self.config, "recompute_plan", None):
            return

        model = recompute_modules(model, self.config.recompute_plan)

    def apply_activation_offload_modules(self, model):
        if (
            getattr(self.config, "activation_offload_plan", None) is None
            or not getattr(self.config, "enable_activation_offload", False)
            or getattr(self.config.activation_offload_plan, "apply_modules", None) is None
        ):
            return

        activation_offload_modules = get_offload_modules(model, getattr(self.config.activation_offload_plan, "apply_modules"))
        async_offload_modules(activation_offload_modules)

    def apply_chunkloss(self, model):
        if self.config.enable_chunk_loss:
            setattr(model, "enable_chunk_loss", True)
            setattr(model, "chunk_size", self.config.chunkloss_plan.chunk_size)
        elif self.config.enable_dynamic_chunk_loss:
            setattr(model, "enable_dynamic_chunk_loss", True)
        else:
            return
        chunkloss_module = get_chunkloss_module(model, self.config.chunkloss_plan)
        apply_chunkloss_module(chunkloss_module)

    def apply_aux_loss_capture(self, model):
        # This function is designed to automatically capture router logits from each MoE layer
        # when 'loss_cfg.router_aux_loss_coef' is configured and greater than 0.
        # These captured logits are essential for calculating the auxiliary loss.
        if (
            getattr(self.config, "loss_cfg", None) is None
            or getattr(self.config.loss_cfg, "router_aux_loss_coef", 0.0) <= 0.0
        ):
            return

        # This logic applies to transformers version 5.2.0 and later.
        # Please use with caution for earlier versions.
        if version.parse(transformers.__version__) >= version.parse("5.2.0"):
            for sub_module in model.modules():
                if hasattr(sub_module, "_can_record_outputs") and len(sub_module._can_record_outputs) > 0:
                    # After applying FSDP sharding via fully_shard, the module paths change
                    # (e.g., 'model.layers.0' becomes 'model.layers.fsdp.0'), causing a mismatch with the
                    # registry keys which are based on the original model structure from
                    # from_pretrained. We need to update the _CAN_RECORD_REGISTRY with the
                    # new class keys from the sharded sub-modules.
                    _CAN_RECORD_REGISTRY[str(sub_module.__class__)] = sub_module._can_record_outputs

    def apply_chunk_mbs(self, model):
        if not getattr(self.config, "enable_chunk_mbs", False) or not getattr(self.config, "chunkmbs_plan", None):
            return

        chunk_mbs_modules = get_chunkmbs_modules(model, self.config.chunkmbs_plan.apply_modules)
        apply_chunkmbs_module(chunk_mbs_modules=chunk_mbs_modules, chunkmbs_cfg=self.config.chunkmbs_plan)

    def apply_optimizer_hook(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        from mindspeed.fsdp.quantization.core.cache import hook_optimizer_step

        hook_optimizer_step(model, optimizer)

    def pre_fully_shard_apply(self, model):
        # The order of these three operations is critical and must not be changed.
        # 1. Recompute: Wraps the forward pass to save memory by recomputing strategy.
        # 2. Activation Offload: Wraps the logic to move activations to CPU to free up device memory.
        # 3. Chunk MBS: Splits the input batch into micro-batches. This must be the outermost wrapper
        #    to ensure that the micro-batch slicing logic executes *before* the data enters the
        #    recomputation and offloading logic.
        self.apply_recompute_models(model=model)
        self.apply_activation_offload_modules(model=model)
        self.apply_chunk_mbs(model=model)

        self.apply_chunkloss(model=model)

    def post_fully_shard_apply(self, model):
        self.apply_aux_loss_capture(model=model)
