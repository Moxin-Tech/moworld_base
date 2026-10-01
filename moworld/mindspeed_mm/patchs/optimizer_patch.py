# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.optim import SGD as CPUSGD
from torch.optim import AdamW as CPUAdam

try:
    from transformer_engine.pytorch.optimizers import FusedAdam as Adam
    from transformer_engine.pytorch.optimizers import FusedSGD as SGD
except ImportError:
    try:
        from apex.optimizers import FusedAdam as Adam
        from apex.optimizers import FusedSGD as SGD
    except ImportError:
        warnings.warn(
            f'Transformer Engine and Apex are not installed. Falling back to Torch optimizers.'
        )

        # Apex's FusedAdam is a drop-in replacement for torch's AdamW.
        # pylint: disable-next=line-too-long.
        from torch.optim import AdamW as Adam, SGD

from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import is_te_min_version
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.grad_scaler import ConstantGradScaler, DynamicGradScaler
from megatron.core.optimizer.optimizer import (
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
)
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.optimizer import _update_min_and_max_lr_in_param_groups
from megatron.training import get_args

from ..optimizer.muon import Muon



def _get_param_groups(
    model_chunks: List[MegatronModule],
    no_weight_decay_cond: Optional[Callable],
    scale_lr_cond: Optional[Callable],
    lr_mult: float,
    lr: float,
    min_lr: float,
    decoupled_lr: Optional[float],
    decoupled_min_lr: Optional[float],
    use_muon: bool = False,
) -> List[Dict]:
    """Create parameter groups for optimizer.

    Creates parameter groups based on weight decay condition (regularized vs
    non regularized), learning rate scale condition (lr vs lr_mult * lr),
    and whether it is expert parameters. scale_lr_cond is used during finetuning
    where head of the network requires a scaled version of the base learning rate.

    Args:
        model_chunks (List[MegatronModule]): model chunks to create parameter
            groups for.
        no_weight_decay_cond (func, optional): function to determine whether a
            parameter should not perform weight decay.
        scale_lr_cond (func, optional): function to determine whether a parameter
            should have a scaled learning rate.
        lr_mult (float): learning rate multiplier for parameters that
            satisfy scale_lr_cond.
        lr (float): learning rate.
        min_lr (float): minimum learning rate.
        decoupled_lr (Optional[float]): optional decoupled learning rate.
        decoupled_min_lr (Optional[float]): optional decoupled minimum learning rate.
        use_muon (bool): if True, generate Muon-compatible groups (matrix/vector),
            otherwise generate default groups.

    Returns:
        List of parameter groups.
    """

    use_decoupled_learning_rate = decoupled_lr is not None

    # Map (wd_mult, lr_mult, is_expert_parallel, is_decoupled_lr) to params.
    params_map = {}
    for model_chunk in model_chunks:
        ddp_config = model_chunk.ddp_config
        if ddp_config.use_custom_fsdp:
            named_parameters = model_chunk.optimizer_named_parameters()
        else:
            named_parameters = model_chunk.named_parameters()

        for name, param in named_parameters:
            if (
                ddp_config.use_custom_fsdp
                and ddp_config.data_parallel_sharding_strategy == "optim_grads_params"
            ):
                param_shard = param
                param = param.orig_param

            if not param.requires_grad:
                continue

            is_expert_parallel = not getattr(param, 'allreduce', True)

            if no_weight_decay_cond is not None:
                no_wd = no_weight_decay_cond(name, param)
            else:
                # Do not regularize biases and norm parameters.
                no_wd = name.endswith(".bias") or len(param.shape) == 1

            if scale_lr_cond is not None:
                scale_lr = scale_lr_cond(name, param)
            else:
                scale_lr = False

            if not no_wd and not scale_lr:
                wd_mult, _lr_mult = 1.0, 1.0
            elif not no_wd and scale_lr:
                wd_mult, _lr_mult = 1.0, lr_mult
            elif no_wd and not scale_lr:
                wd_mult, _lr_mult = 0.0, 1.0
            else:
                wd_mult, _lr_mult = 0.0, lr_mult

            is_decoupled_lr = False
            # For input/embedding and output layer: embedding.word_embeddings.weight /
            # output_layer.weight.
            if use_decoupled_learning_rate and getattr(
                param, 'is_embedding_or_output_parameter', False
            ):
                is_decoupled_lr = True

            if use_muon:
                is_2d_matrix = len(param.shape) == 2
                no_muon = name.endswith(".bias") or "embedding" in name or "output_layer" in name or not is_2d_matrix
                key = (wd_mult, _lr_mult, is_expert_parallel, is_decoupled_lr, no_muon)
            else:
                key = (wd_mult, _lr_mult, is_expert_parallel, is_decoupled_lr)
            if key not in params_map:
                params_map[key] = []
            if (
                ddp_config.use_custom_fsdp
                and ddp_config.data_parallel_sharding_strategy == "optim_grads_params"
            ):
                params_map[key].append(param_shard)
            else:
                params_map[key].append(param)

    param_groups = []
    if use_muon:
        for (wd_mult, _lr_mult, is_expert_parallel, is_decoupled_lr, no_muon), params in params_map.items():
            if len(params) == 0:
                continue

            param_group = {
                'params': params,
                'wd_mult': wd_mult,
                'lr_mult': _lr_mult,
                'is_expert_parallel': is_expert_parallel,
                'is_decoupled_lr': is_decoupled_lr,
                'use_muon': not no_muon,
            }
            param_groups.append(param_group)
    else:
        for (wd_mult, _lr_mult, is_expert_parallel, is_decoupled_lr), params in params_map.items():
            if len(params) == 0:
                continue

            param_group = {
                'params': params,
                'wd_mult': wd_mult,
                'lr_mult': _lr_mult,
                'is_expert_parallel': is_expert_parallel,
                'is_decoupled_lr': is_decoupled_lr,
            }
            param_groups.append(param_group)

    param_groups = _update_min_and_max_lr_in_param_groups(
        param_groups,
        lr=lr,
        min_lr=min_lr,
        decoupled_lr=decoupled_lr,
        decoupled_min_lr=decoupled_min_lr,
    )

    return param_groups


def _get_param_groups_and_buffers(
    model_chunks: List[MegatronModule],
    model_chunk_offset: int,
    config: OptimizerConfig,
    no_weight_decay_cond: Optional[Callable],
    scale_lr_cond: Optional[Callable],
    lr_mult: float,
    filter_fn: Callable,
    buffer_name: str,
) -> Tuple[List[Dict], Dict[int, List[_ParamAndGradBuffer]]]:
    """Returns parameter groups and buffer for optimizer.

    Args:
        model_chunks (List[MegatronModule]): model chunks to create parameter
            groups for.
        model_chunk_offset (int): offset of model_chunks in global model_chunks list.
        config (OptimizerConfig): optimizer configuration object.
        no_weight_decay_cond (func, optional): function to determine whether a
            parameter should not perform weight decay.
        scale_lr_cond (func, optional): function to determine whether a parameter
            should have a scaled learning rate.
        lr_mult (float): learning rate multiplier for parameters that
            satisfy scale_lr_cond.
        lr (float): learning rate.
        min_lr (float): minimum learning rate.
        filter_fn (callable): filtering function for param_groups.
        buffer_name (str): name of buffer.

    Returns:
        List of parameter groups and dictionary of model chunk IDs to buffers.
    """
    param_groups = _get_param_groups(
        model_chunks,
        no_weight_decay_cond,
        scale_lr_cond,
        lr_mult,
        lr=config.lr,
        min_lr=config.min_lr,
        decoupled_lr=config.decoupled_lr,
        decoupled_min_lr=config.decoupled_min_lr,
        use_muon=config.optimizer == 'muon'
    )
    param_groups = list(filter(filter_fn, param_groups))
    buffers = {}
    for model_chunk_idx, model_chunk in enumerate(model_chunks):
        if hasattr(model_chunk, buffer_name):
            buffers[model_chunk_idx + model_chunk_offset] = getattr(model_chunk, buffer_name)

    return param_groups, buffers


def _get_megatron_optimizer_based_on_param_groups(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    param_groups: List,
    per_model_buffers: Optional[Dict[int, List[_ParamAndGradBuffer]]] = None,
    model_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    data_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    data_parallel_group_gloo: Optional[torch.distributed.ProcessGroup] = None,
    data_parallel_group_idx: Optional[int] = None,
    distributed_optimizer_instance_id: Optional[int] = 0,
) -> MegatronOptimizer:
    """Get Megatron optimizer based on parameter groups.

    Args:
        config (OptimizerConfig): optimizer configuration object.
        model_chunks (list): list of model chunks.
        param_groups (list): list of parameter groups.
        per_model_buffers (dict, optional): buffers for distributed optimizer. Defaults to None.
        data_parallel_group (torch.distributed.ProcessGroup, optional): data-parallel group for
            distributed optimizer. Defaults to None.
        data_parallel_group_gloo (torch.distributed.ProcessGroup, optional): gloo data-parallel
            group for distributed optimizer. Defaults to None.
        data_parallel_group_idx (int, optional): data-parallel group index for distributed
            optimizer. Defaults to None.
        distributed_optimizer_instance_id (int, optional): Distributed optimizer instance. Defaults
            0.

    Returns:
        Instance of MegatronOptimizer.
    """
    # when freezing sub-models we may have no trainable parameters on a rank and
    # hence an empty param_groups. However, we still need to create an optimizer
    # for the purposes of grad stats reductions
    if param_groups:
        if config.optimizer_cpu_offload:
            if torch.__version__ < '2.3.0':
                warnings.warn(
                    "CPU offload is recommended for PyTorch >= 2.3.0, "
                    "untested versions below this may have convergence issues."
                )
            gpu_optimizer_cls = Adam if config.optimizer == 'adam' else SGD
            cpu_optimizer_cls = CPUAdam if config.optimizer == 'adam' else CPUSGD
            if config.use_torch_optimizer_for_cpu_offload:
                gpu_optimizer_cls = cpu_optimizer_cls
            if config.optimizer == 'adam':
                gpu_optimizer_cls = Adam
                cpu_optimizer_cls = CPUAdam
                optimizer_defaults = dict(
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                    betas=(config.adam_beta1, config.adam_beta2),
                    eps=config.adam_eps,
                    bias_correction=True,
                    fused=True,  # this flag is used to improve the performance of the cpu optimizer
                )
            else:
                gpu_optimizer_cls = SGD
                cpu_optimizer_cls = CPUSGD
                optimizer_defaults = dict(
                    lr=config.lr, weight_decay=config.weight_decay, momentum=config.sgd_momentum
                )

            optimizer = HybridDeviceOptimizer(
                param_groups,
                offload_fraction=config.optimizer_offload_fraction,
                cpu_optimizer_cls=cpu_optimizer_cls,
                gpu_optimizer_cls=gpu_optimizer_cls,
                overlap_cpu_optimizer_d2h_h2d=config.overlap_cpu_optimizer_d2h_h2d,
                pin_cpu_grads=config.pin_cpu_grads,
                pin_cpu_params=config.pin_cpu_params,
                param_update_in_fp32=True,
                **optimizer_defaults,
            )
            init_state_fn = None
        elif config.optimizer == 'adam':
            kwargs = {
                "params": param_groups,
                "lr": config.lr,
                "weight_decay": config.weight_decay,
                "betas": (config.adam_beta1, config.adam_beta2),
                "eps": config.adam_eps,
            }

            if config.use_precision_aware_optimizer:
                kwargs.update(
                    {
                        "master_weights": True,
                        "use_decoupled_grad": True,
                        "master_weight_dtype": config.main_params_dtype,
                        "exp_avg_dtype": config.exp_avg_dtype,
                        "exp_avg_sq_dtype": config.exp_avg_sq_dtype,
                    }
                )

                if is_te_min_version("2.1.0.dev0"):
                    kwargs.update({"store_param_remainders": True})

            optimizer = Adam(**kwargs)

            def init_state_fn(opt, config=None):
                for group in opt.param_groups:
                    for p in group['params']:
                        if len(opt.state[p]) == 0:
                            if config is None or not config.use_precision_aware_optimizer:
                                opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                                opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
                            else:
                                opt.initialize_state(p)

        elif config.optimizer == 'sgd':
            optimizer = SGD(
                param_groups,
                lr=config.lr,
                weight_decay=config.weight_decay,
                momentum=config.sgd_momentum,
            )
            init_state_fn = None

        elif config.optimizer == 'muon':
            args = get_args()
            optimizer = Muon(
                param_groups,
                lr=config.lr,
                weight_decay=config.weight_decay,
                matched_adamw_rms=args.matched_adamw_rms,
                momentum=args.muon_momentum,
                ns_steps=args.ns_steps,
                adamw_betas=(config.adam_beta1, config.adam_beta2),
                adamw_eps=config.adam_eps,
            )

            def init_state_fn(opt, config=None):
                for group in opt.param_groups:
                    for p in group['params']:
                        if len(opt.state[p]) == 0:
                            if config is None or not config.use_precision_aware_optimizer:
                                opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                                opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
                            else:
                                opt.initialize_state(p)
        else:
            raise Exception('{} optimizer is not supported.'.format(config.optimizer))
    else:
        optimizer = None
        init_state_fn = None

    # Mixed precision optimizer.
    # - Note: both the Float16Optimizer and the DistributedOptimizer inherit
    #   from the MixedPrecisionOptimizer, which manages any optimizer where
    #   the model params and main params are distinct.
    if config.fp16 or config.bf16 or config.use_distributed_optimizer:

        # Grad scaler:
        #    if loss-scale is provided, instantiate the constant scaler.
        #    if we are using fp16 and loss-scale is not present, use a
        #       dynamic scaler.
        #    otherwise we are running in bf16 with no loss-scale so
        #       leave it as None.
        grad_scaler = None

        # Constant loss scale.
        if config.loss_scale:
            grad_scaler = ConstantGradScaler(config.loss_scale)

        # Dynamic loss scale.
        else:
            if config.fp16:
                grad_scaler = DynamicGradScaler(
                    initial_scale=config.initial_loss_scale,
                    min_scale=config.min_loss_scale,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=config.loss_scale_window,
                    hysteresis=config.hysteresis,
                )

        optimizer_args = [optimizer, config, grad_scaler, init_state_fn]
        if config.use_distributed_optimizer:
            optimizer = DistributedOptimizer(
                *optimizer_args,
                model_chunks=model_chunks,
                per_model_buffers=per_model_buffers,
                data_parallel_group=data_parallel_group,
                data_parallel_group_gloo=data_parallel_group_gloo,
                data_parallel_group_idx=data_parallel_group_idx,
                distributed_optimizer_instance_id=distributed_optimizer_instance_id,
            )
        else:
            optimizer = Float16OptimizerWithFloat16Params(*optimizer_args)
            setattr(optimizer, 'grad_stats_parallel_group', model_parallel_group)
    else:
        # FP32 optimizer.
        optimizer = FP32Optimizer(optimizer, config, init_state_fn)
        setattr(optimizer, 'grad_stats_parallel_group', model_parallel_group)

    return optimizer
