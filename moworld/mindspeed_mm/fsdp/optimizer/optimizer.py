# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from collections import ChainMap
import logging

import torch
import torch.nn as nn
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer

from ..distributed.parallel_state import get_parallel_state
from ...optimizer.muon import Muon


logger = logging.getLogger(__name__)


class MultiOptimizer(Optimizer, Stateful):
    """
    A container that handles multiple optimizers (for ep and non-ep parameters when ep+fsdp2 is enabled)

    Mapping of name -> torch.optim.Optimizer with convenience methods.
    Compatible with torch.distributed.checkpoint optimizer APIs that accept a Mapping.

    This class is needed for EP+FSDP2 case because EP and non-EP param have different FSDP sharding dimension (dim-0 vs. dim-1)
    For comparison, EP+FSDP1 also shards EP parameters along dim-0 for FSDP, so it can use the default optimizer class.
    """

    def __init__(
        self,
        root_model: nn.Module,
        optimizers: dict,  # {"ep": opt1, "non_ep": opt2}
        key_names: list[str],
    ):
        self.model = root_model
        self.optimizers_dict = optimizers
        self._is_multi_optimizer: bool = True
        self.key_names = key_names

    @property
    def state(self):
        """
        Returns a read-only aggregated view of the states from all sub-optimizers.
        Uses collections.ChainMap to combine the state dictionaries without copying,
        providing efficient and unified access while preserving immutability at this level.
        """
        state_dicts = [opt.state for opt in self.optimizers_dict.values()]
        return ChainMap(*state_dicts)

    @property
    def param_groups(self):
        """
        Returns a flat list aggregating all parameter groups from every sub-optimizer.
        This allows the composite optimizer to expose a unified interface compatible
        with standard PyTorch optimizer expectations (e.g., for learning rate schedulers).
        """
        all_groups = []
        for opt in self.optimizers_dict.values():
            all_groups.extend(opt.param_groups)
        return all_groups

    def step(self) -> None:
        for opt in self.optimizers_dict.values():
            opt.step()

    def zero_grad(self) -> None:
        for opt in self.optimizers_dict.values():
            opt.zero_grad()

    def state_dict(
        self,
    ) -> Dict[str, Any]:
        # get the flatten state dict for multi-optimizer
        merged: Dict[str, Any] = {}
        for name in self.key_names:
            opt = self.optimizers_dict.get(name)
            sd = get_optimizer_state_dict(self.model, opt, options=StateDictOptions(flatten_optimizer_state_dict=True))
            # check for key clashes before merging
            overlap = set(merged.keys()) & set(sd.keys())
            if overlap:
                raise KeyError(
                    f"Key clash detected while merging state dict for optimizer '{name}': {', '.join(sorted(overlap))}"
                )
            else:
                logger.info("No clashes when merging MultiOptimizer state dicts")
            merged.update(sd)

        return merged

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # Feed the same merged flattened dict to each sub-optimizer; PyTorch will
        # pick out only the entries for parameters that belong to that optimizer.
        for name in self.key_names:
            opt = self.optimizers_dict.get(name)
            set_optimizer_state_dict(
                self.model,
                opt,
                optim_state_dict=state_dict,
                options=StateDictOptions(flatten_optimizer_state_dict=True),
            )

    def register_step_pre_hook(self, hook):
        return [opt.register_step_pre_hook(hook) for opt in self.optimizers_dict.values()]

    def __len__(self) -> int:
        return len(self.optimizers_dict)

    def __repr__(self) -> str:
        return self.optimizers_dict.__repr__()


def _make_param_groups_for_subset(
    model: "nn.Module",
    params: Iterable[torch.nn.Parameter],
    weight_decay: float,
    no_decay_modules: Optional[List[str]] = None,
    no_decay_params: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    decay_param_names = set(get_parameter_names(model, no_decay_modules, no_decay_params))
    name_by_param = {p: n for n, p in model.named_parameters()}
    params = [p for p in params if p.requires_grad]
    decayed = [p for p in params if name_by_param.get(p) in decay_param_names]
    undecayed = [p for p in params if name_by_param.get(p) not in decay_param_names]
    groups: List[Dict[str, Any]] = []
    if decayed:
        groups.append({"params": decayed, "weight_decay": weight_decay})
    if undecayed:
        groups.append({"params": undecayed, "weight_decay": 0.0})
    return groups


# Check if a parameter is eligible for Muon optimization.
def _is_muon_eligible(name: str, param: torch.nn.Parameter) -> bool:
    is_2d_matrix = len(param.shape) == 2
    return (
        not name.endswith(".bias")
        and "embedding" not in name
        and "output_layer" not in name
        and is_2d_matrix
    )


def _mark_muon_param_groups(
    model: "nn.Module",
    param_groups: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    name_by_param = {p: n for n, p in model.named_parameters()}
    marked_groups: List[Dict[str, Any]] = []

    for group in param_groups:
        params = group.get("params", [])
        muon_params = []
        fallback_params = []

        for p in params:
            if not p.requires_grad:
                continue
            param_name = name_by_param.get(p, "")
            if _is_muon_eligible(param_name, p):
                muon_params.append(p)
            else:
                fallback_params.append(p)

        group_base = {k: v for k, v in group.items() if k != "params"}
        if muon_params:
            marked_groups.append({**group_base, "params": muon_params, "use_muon": True})
        if fallback_params:
            marked_groups.append({**group_base, "params": fallback_params, "use_muon": False})

    return marked_groups


# adapted from https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer_pt_utils.py#L1123
def get_parameter_names(model, forbidden_layer_types, forbidden_param_names):
    forbidden_layer_types = [] if forbidden_layer_types is None else forbidden_layer_types
    forbidden_param_names = [] if forbidden_param_names is None else forbidden_param_names
    result = []
    for name, child in model.named_children():
        child_params = get_parameter_names(child, forbidden_layer_types, forbidden_param_names)
        result += [
            f"{name}.{n}"
            for n in child_params
            if child.__class__.__name__ not in forbidden_layer_types
            and not any(forbidden in f"{name}.{n}".lower() for forbidden in forbidden_param_names)
        ]

    result += [
        k for k in model._parameters.keys() if not any(forbidden in k.lower() for forbidden in forbidden_param_names)
    ]
    return result


def build_optimizer(
    model: "nn.Module",
    lr: float = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
    fused: bool = False,
    optimizer_type: str = "adamw",
    param_groups: Optional[Sequence[Dict[str, Any]]] = None,
    no_decay_modules: Optional[List[str]] = None,
    no_decay_params: Optional[List[str]] = None,
    matched_adamw_rms: float = 0.2,
    muon_momentum: float = 0.95,
    ns_steps: int = 5,
    lr_scaling_plan: Optional[List] = None,
) -> "torch.optim.Optimizer":
    # EP-aware routing: for FSDP2+EP, split params into EP and non-EP groups and build two optimizers.
    ps = get_parallel_state()
    if ps.get_ep_group_size() > 1:
        logger.info("Building EP+FSDP2 optimizer")
        return build_ep_fsdp2_optimizer(
            model,
            lr,
            betas,
            eps,
            weight_decay,
            fused,
            optimizer_type,
            param_groups,
            no_decay_modules,
            no_decay_params,
            matched_adamw_rms=matched_adamw_rms,
            muon_momentum=muon_momentum,
            ns_steps=ns_steps,
        )
    # Other cases remain the same
    if lr_scaling_plan:
        decay_param_names = get_parameter_names(model, no_decay_modules, no_decay_params)
        param_groups = get_param_groups(model, weight_decay, lr, lr_scaling_plan, decay_param_names)

    elif param_groups is None:
        decay_param_names = get_parameter_names(model, no_decay_modules, no_decay_params)
        param_groups = [
            {
                "params": [p for n, p in model.named_parameters() if n in decay_param_names and p.requires_grad],
                "weight_decay": weight_decay,
            },
        ]
        no_decay_parameters, no_decay_parameter_names = [], []
        for n, p in model.named_parameters():
            if n not in decay_param_names and p.requires_grad:
                no_decay_parameter_names.append(n)
                no_decay_parameters.append(p)

        if len(no_decay_parameters) > 0:
            logger.info(f"Parameters without weight decay: {no_decay_parameter_names}")
            param_groups.append({"params": no_decay_parameters, "weight_decay": 0.0})

    if optimizer_type == "muon":
        param_groups = _mark_muon_param_groups(model, param_groups)
        logger.info(f"Muon parameter groups: {param_groups}")
        optim = Muon(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            matched_adamw_rms=matched_adamw_rms,
            momentum=muon_momentum,
            ns_steps=ns_steps,
            adamw_betas=betas,
            adamw_eps=eps,
        )
    elif optimizer_type == "adamw":
        foreach = not fused
        fused = fused
        optim = AdamW(param_groups, lr, betas, eps, weight_decay, fused=fused, foreach=foreach)
    else:
        raise ValueError("Only adamw and muon are supported as optimizers.")

    return optim


def group_params_by_lr_ratio(param_names: List[str], params: List[torch.Tensor],
                             lr_scaling_plan: List[Dict], base_lr: float) -> List[Dict]:
    """按lr_scaling_plan对参数进行分组"""
    groups_dict = {}
    if not lr_scaling_plan:
        lr_scaling_plan = []

    for name, param in zip(param_names, params):
        lr_ratio = 1.0
        for pattern_config in lr_scaling_plan:
            if re.match(pattern_config.match, name):
                lr_ratio = pattern_config.scale
                break

        lr = base_lr * lr_ratio

        if lr not in groups_dict:
            groups_dict[lr] = {"params": [], "lr": lr}

        groups_dict[lr]["params"].append(param)

    return list(groups_dict.values())


def get_param_groups(
        model: torch.nn.Module,
        weight_decay: float,
        base_lr: float,
        lr_scaling_plan: List[Dict] = None,
        decay_param_names: List[str] = None,
) -> List[Dict]:
    """按正则表达式分组参数，先按weight_decay，再按lr_ratio分组"""

    no_decay_parameters, no_decay_parameter_names = [], []
    decay_parameters, decay_parameter_names = [], []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if n in decay_param_names:
            decay_parameter_names.append(n)
            decay_parameters.append(p)
        else:
            no_decay_parameter_names.append(n)
            no_decay_parameters.append(p)

    if len(no_decay_parameters) > 0:
        logger.info(f"Parameters without weight decay: {no_decay_parameter_names}")

    decay_groups = group_params_by_lr_ratio(decay_parameter_names, decay_parameters, lr_scaling_plan, base_lr)
    no_decay_groups = group_params_by_lr_ratio(no_decay_parameter_names, no_decay_parameters, lr_scaling_plan, base_lr)

    param_groups = []
    for group in decay_groups:
        group["weight_decay"] = weight_decay
        param_groups.append(group)
    for group in no_decay_groups:
        group["weight_decay"] = 0.0
        param_groups.append(group)

    return param_groups


def build_ep_fsdp2_optimizer(
    model: "nn.Module",
    lr: float = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
    fused: bool = False,
    optimizer_type: str = "adamw",
    param_groups: Optional[List[Dict[str, Any]]] = None,
    no_decay_modules: Optional[List[str]] = None,
    no_decay_params: Optional[List[str]] = None,
    matched_adamw_rms: float = 0.2,
    muon_momentum: float = 0.95,
    ns_steps: int = 5,
):
    """
    Build a MultiOptimizer instance when model is parallelized with EP+FSDP2

    If param_groups provided, it can be a list of dicts with arbitrary parameter groups:
    - Example: [{"params": params1, "lr": lr1},
                {"params": params2, "lr": lr2},
                {"params": params3, "lr": lr3}]
    - Each group's params are automatically split into EP and non-EP based on DTensor mesh
    - Custom learning rates and other optimizer settings are preserved per group
    """
    # Collect all EP and non-EP parameters across all groups
    ep_groups: List[Dict[str, Any]] = []
    non_ep_groups: List[Dict[str, Any]] = []

    # Process custom param_groups if provided
    if param_groups is not None:
        # Process each parameter group
        for group_config in param_groups:
            # Extract group-specific settings
            group_lr = group_config.get("lr", lr)
            group_params = group_config["params"]

            # Split this group's params into EP and non-EP
            group_ep_params: List[torch.nn.Parameter] = []
            group_non_ep_params: List[torch.nn.Parameter] = []

            for p in group_params:
                if not p.requires_grad:
                    continue
                if DTensor is not None and isinstance(p, DTensor):
                    mesh = getattr(p, "device_mesh", None)
                    names = getattr(mesh, "mesh_dim_names", []) if mesh is not None else []
                    # 根据是否有efsdp mesh才获取参数
                    if "efsdp" in names:
                        group_ep_params.append(p)
                        continue
                group_non_ep_params.append(p)

            # Create subgroups with weight decay handling
            if group_ep_params:
                group_ep_subgroups = _make_param_groups_for_subset(
                    model, group_ep_params, weight_decay, no_decay_modules, no_decay_params
                )
                for subgroup in group_ep_subgroups:
                    subgroup["lr"] = group_lr
                    # Preserve other custom settings from original group
                    for key, value in group_config.items():
                        if key not in ["params", "lr", "weight_decay"]:
                            subgroup[key] = value
                ep_groups.extend(group_ep_subgroups)

            if group_non_ep_params:
                group_non_ep_subgroups = _make_param_groups_for_subset(
                    model, group_non_ep_params, weight_decay, no_decay_modules, no_decay_params
                )
                for subgroup in group_non_ep_subgroups:
                    subgroup["lr"] = group_lr
                    # Preserve other custom settings from original group
                    for key, value in group_config.items():
                        if key not in ["params", "lr", "weight_decay"]:
                            subgroup[key] = value
                non_ep_groups.extend(group_non_ep_subgroups)
    else:
        # Default case (param_groups is None): all model parameters with uniform settings(lr)
        ep_params: List[torch.nn.Parameter] = []
        non_ep_params: List[torch.nn.Parameter] = []

        for _, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if DTensor is not None and isinstance(p, DTensor):
                mesh = getattr(p, "device_mesh", None)
                names = getattr(mesh, "mesh_dim_names", []) if mesh is not None else []
                if "efsdp" in names:
                    ep_params.append(p)
                    continue
            non_ep_params.append(p)

        # Build param groups with weight decay handling
        ep_groups = _make_param_groups_for_subset(model, ep_params, weight_decay, no_decay_modules, no_decay_params)
        non_ep_groups = _make_param_groups_for_subset(
            model, non_ep_params, weight_decay, no_decay_modules, no_decay_params
        )

    def _build(groups: Sequence[Dict[str, Any]]) -> Optimizer:
        foreach = not fused
        fused_ = fused
        if optimizer_type == "muon":
            groups = _mark_muon_param_groups(model, groups)
            return Muon(
                groups,
                lr=lr,
                weight_decay=weight_decay,
                matched_adamw_rms=matched_adamw_rms,
                momentum=muon_momentum,
                ns_steps=ns_steps,
                adamw_betas=betas,
                adamw_eps=eps,
            )
        elif optimizer_type == "adamw":
            return AdamW(groups, lr, betas, eps, weight_decay, fused=fused_, foreach=foreach)
        else:
            raise ValueError("Only adamw and muon are supported as optimizers.")

    optimizer_dict: Dict[str, Optimizer] = {}
    if ep_groups:
        optimizer_dict["ep"] = _build(ep_groups)
    if non_ep_groups:
        optimizer_dict["non_ep"] = _build(non_ep_groups)

    # cache for EP-aware grad clipping helpers
    model._ep_param_groups = {
        "ep": [p for g in ep_groups for p in g.get("params", [])] if ep_groups else [],
        "non_ep": [p for g in non_ep_groups for p in g.get("params", [])] if non_ep_groups else [],
    }

    key_names = list(optimizer_dict.keys())

    # Build MultiOptimizer and attach a pre-step hook to sanitize DTensor states
    multi_opt = MultiOptimizer(model, optimizer_dict, key_names=key_names)

    return multi_opt
