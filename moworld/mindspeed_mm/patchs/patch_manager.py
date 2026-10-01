# Copyright (c) 2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from megatron.training import get_args
from megatron.training.utils import print_rank_0
from mindspeed.patch_utils import MindSpeedPatchesManager as pm

from mindspeed_mm.patchs import (
    adaptive_clip_grad_patch,
    infer_fa_patch,
    models_patches,
    fsdp1_patches,
    training_patches,
    fsdp2_patches,
    optimizer_patch,
    bridge_patch
)
from mindspeed_mm.patchs.layerwise_disaggregated_training import (
    distributed_data_parallel_patch,
    parallel_state_patch,
    schedules_patch,
    utils_patch,
    u_shaped_split_learning_patch,
    vlm_model_patch
)


class PatchesManager:
    configs = {
        "ae_float32": [
            ("megatron.core.transformer.module.Float16Module.__init__", models_patches.float16Module_init),
            ("megatron.core.transformer.module.Float16Module.forward", models_patches.float16Module_forward)
        ],
        "adaptive_clip_grad_norm": [
            ("megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.__init__", adaptive_clip_grad_patch.adaptive_clip_grad_norm_optimizer_init_wrapper),
            ("megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.clip_grad_norm", adaptive_clip_grad_patch.adaptive_clip_grad_norm_wrapper)
        ],
        "infer_fa": [("megatron.core.transformer.dot_product_attention.DotProductAttention.forward", infer_fa_patch.dot_product_attention_forward_infer_wrapper)],
        "use_fsdp1": [
            ("megatron.training.training.get_model", fsdp1_patches.fsdp1_get_model)
        ],
        "clip_grad_async": [
            ("megatron.core.optimizer.clip_grads.get_grad_norm_fp32", adaptive_clip_grad_patch.get_grad_norm_fp32_async),
            ("megatron.core.optimizer.clip_grads.clip_grad_by_total_norm_fp32", adaptive_clip_grad_patch.clip_grad_by_total_norm_fp32_async)
        ],

        # Enable this patch when loading model weights from a .pt checkpoint file in distributed training.
        # This will override the default model loading behavior to handle distributed checkpoint format.
        "get_dist_model_load_from_pt": [
            ("megatron.training.training.get_model", training_patches.get_dist_model_load_from_pt)
        ],
        "bridge_patch": [
            ("megatron.training.training.get_model", bridge_patch.get_model)
        ],
        "scale_grad": [
            ("megatron.core.distributed.TorchFullyShardedDataParallel.scale_gradients", fsdp2_patches.scale_gradients)
        ],
        "muon_optimizer": [
            ("megatron.core.optimizer._get_param_groups", optimizer_patch._get_param_groups),
            ("megatron.core.optimizer._get_param_groups_and_buffers", optimizer_patch._get_param_groups_and_buffers),
            ("megatron.core.optimizer._get_megatron_optimizer_based_on_param_groups", optimizer_patch._get_megatron_optimizer_based_on_param_groups)
        ],
        "layerwise_disaggregated_training": [
            ("megatron.core.parallel_state.create_group", parallel_state_patch.create_group),
            ("megatron.core.pipeline_parallel.schedules.get_forward_backward_func", schedules_patch.get_forward_backward_func),
            ("megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_without_interleaving", schedules_patch.forward_backward_pipelining_without_interleaving),
            ("megatron.training.training.build_train_valid_test_datasets", u_shaped_split_learning_patch.build_train_valid_test_datasets_wrapper),
            ("megatron.training.training.setup_model_and_optimizer", vlm_model_patch.setup_model_and_optimizer),
            ("megatron.training.utils.print_rank_last", print_rank_0),
            ("megatron.core.distributed.distributed_data_parallel.finish_grad_sync", distributed_data_parallel_patch.finish_grad_sync),
            ("megatron.core.distributed.distributed_data_parallel.register_grad_ready", distributed_data_parallel_patch.register_grad_ready),
            ("megatron.training.utils.reduce_max_stat_across_model_parallel_group", utils_patch.ldt_reduce_max_stat_across_model_parallel_group),
            ("megatron.training.utils.logical_and_across_model_parallel_group", utils_patch.ldt_logical_and_across_model_parallel_group),
            ("megatron.core.optimizer.clip_grads.get_grad_norm_fp32", utils_patch.ldt_get_grad_norm_fp32),
            ("torch.distributed.barrier", utils_patch.ldt_vdp_barrier_wrapper),
            ('torch.distributed.all_gather_into_tensor', utils_patch.vtp_all_gather_into_tensor_wrapper)
        ],
    }

    @staticmethod
    def register_patch(orig_func_name, new_func=None):
        pm.register_patch(orig_func_name, new_func, force_patch=True)

    @staticmethod
    def apply_patches():
        pm.apply_patches()

    @staticmethod
    def apply_patches_from_config():
        cfg = get_args().mm.model
        if hasattr(cfg, "patch"):
            cfg = cfg.patch.to_dict()
            for key in cfg.keys() & PatchesManager.configs.keys():
                if not cfg.get(key):
                    continue
                for orig_func_name, new_func in PatchesManager.configs[key]:
                    PatchesManager.register_patch(orig_func_name, new_func)
        PatchesManager.apply_patches()
