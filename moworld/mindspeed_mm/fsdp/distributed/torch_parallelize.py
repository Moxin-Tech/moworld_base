import torch

from mindspeed.fsdp.distributed.tensor_parallel.tensor_parallel import tensor_parallel_modules

from .expert_parallel.expert_parallel import expert_parallelize_modules
from .expert_parallel.expert_fully_shard_parallel import expert_fully_shard_modules
from .fully_shard_parallel import fully_shard_parallel_modules, set_modules_to_prefetch
from .parallel_state import get_parallel_state
from ..params.parallel_args import ParallelArguments
from ..params.training_args import TrainingArguments


class ParallelApplier:
    def __init__(self, parallel_config: ParallelArguments, training_config: TrainingArguments):
        self.config = parallel_config
        self.training_config = training_config
        self.parallel_state = get_parallel_state()

    def apply_fsdp_modules(self, model, training_config):
        model = fully_shard_parallel_modules(model, self.parallel_state.get_fsdp_device_mesh(), self.config.fsdp_plan,
                                             training_config)
        return model

    def apply_tp_modules(self, model):
        if self.config.tensor_parallel_size == 1:
            return
        model = tensor_parallel_modules(model, self.parallel_state.get_tp_device_mesh(), self.config.tp_plan)

    def apply_ep_modules(self, model):
        if not self.config.ep_plan.apply_efsdp_modules:
            self.config.ep_plan.apply_efsdp_modules = self.config.ep_plan.apply_modules
        if self.config.ep_plan._gradient_divide_factor is None:
            self.config.ep_plan._gradient_divide_factor = torch.distributed.get_world_size()
        if self.config.expert_parallel_size > 1 and self.config.ep_plan.apply_modules:
            model = expert_parallelize_modules(model, self.parallel_state.get_ep_device_mesh(), self.config.ep_plan)
            model = expert_fully_shard_modules(model, self.parallel_state.get_efsdp_device_mesh(), self.config.ep_plan, self.config.fsdp_plan)
            # Remove modules already handled by efsdp from the standard FSDP plan to prevent duplicate fully_shard errors
            self.config.fsdp_plan.apply_modules = [x for x in self.config.fsdp_plan.apply_modules if x not in self.config.ep_plan.apply_efsdp_modules]

    def set_modules_to_prefetch(self, model):
        if self.config.fsdp_plan.num_to_forward_prefetch > 0 or self.config.fsdp_plan.num_to_backward_prefetch > 0:
            ep_plan = self.config.ep_plan if self.config.expert_parallel_size > 1 and self.config.ep_plan.apply_modules else None
            set_modules_to_prefetch(model, fsdp_plan=self.config.fsdp_plan, ep_plan=ep_plan)
        return model

    def apply_quantization_modules(self, model):
        """Apply quantization based on quantization_format + quantization_recipe"""
        if not self.training_config.quantization_plan.recipe_name:
            return
        try:
            # recompute must set all now, TODO fix this bug
            if self.config.recompute:
                self.training_config.quantization_plan.fsdp_low_precision_all_gather_mode = "all"

            from mindspeed.fsdp.quantization.converter.model_converter import build_model_converter

            model_converters = build_model_converter(self.training_config.quantization_plan)
            model_converters.convert(model)
        except Exception as e:
            raise RuntimeError(f"Failed to convert quantization plan") from e

    def __call__(self, model):
        # Apply configuration-based parallel strategies
        # Order matters: TP -> EP -> Recompute -> FSDP
        self.apply_quantization_modules(model=model) # must set before fully_shard
        self.apply_tp_modules(model=model)
        self.apply_ep_modules(model=model)
        model = self.apply_fsdp_modules(model=model, training_config=self.training_config)
        model = self.set_modules_to_prefetch(model=model)
        return model
