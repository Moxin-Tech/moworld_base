# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Any
from functools import partial
from dataclasses import replace
import torch
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.metadata import Metadata, TensorStorageMetadata
from torch.distributed.checkpoint.planner import LoadPlan
from torch.distributed.checkpoint.planner_helpers import _create_read_items
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner


class EPLoadPlanner(DefaultLoadPlanner):
    """ Expert Parallel Load Planner

    This class extends the DefaultLoadPlanner to handle expert parallelism (EP) during checkpoint loading.
    It customizes the load plan creation to account for MoE layers and their distribution across expert parallel ranks.
    """

    def __init__(self, ep_rank: int = 0, ep_size: int = 1, check_moe_fn=None, **kwargs):
        super().__init__(**kwargs)
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.check_moe_fn = check_moe_fn if check_moe_fn is not None else lambda x: False

    def create_local_plan(self) -> LoadPlan:
        create_default_local_load_plan = partial(create_default_local_load_plan_with_moe, self.check_moe_fn, self.ep_rank)
        torch.distributed.checkpoint.default_planner.create_default_local_load_plan = create_default_local_load_plan
        return super().create_local_plan()


def create_default_local_load_plan_with_moe(
    check_moe_fn, ep_rank, state_dict: dict[str, Any], metadata: Metadata, strict: bool = True
) -> LoadPlan:
    requests = []
    """
    Create the ``LoadPlan`` used by DefaultLoadPlanner.

    It produces one read item per value in ``state_dict`` using the metadata in ``metadata``.

    The default behavior is to match key exactly between state_dict and metadata.
    It handles resharding by issuing multiple read requests against storage in order to match
    load requirements.
    """

    for fqn, obj in state_dict.items():
        # ignore state_dict keys which do not exist in `state_dict` if strict=False
        if fqn not in metadata.state_dict_metadata:
            if strict:
                raise RuntimeError(f"Missing key in checkpoint state_dict: {fqn}.")
            else:
                continue

        md = metadata.state_dict_metadata[fqn]
        if not check_moe_fn(fqn): # keep non-MoE layers unchanged
            if (
                isinstance(md, TensorStorageMetadata)
                and getattr(obj, "size", None) is not None
                and md.size != obj.size()
            ):
                raise ValueError(
                    f"Size mismatch between saved {md.size} and current: {obj.size()} for {fqn}",
                )
            # Since DTensor supports submesh, adding extra check to ensure _create_read_items()
            # gets called only when the current rank is part of the mesh for the corresponding DTensor.
            if isinstance(obj, DTensor):
                if obj.device_mesh.get_coordinate() is not None:
                    requests += _create_read_items(fqn, md, obj)
            else:
                requests += _create_read_items(fqn, md, obj)
        else: # MoE layers need to be chunked according to expert parallel rank
            if isinstance(obj, DTensor):
                if obj.device_mesh.get_coordinate() is not None:
                    moe_req = _create_read_items(fqn, md, obj)
            else:
                moe_req = _create_read_items(fqn, md, obj)
            requests += [get_chunk_readitem(req, ep_rank) for req in moe_req]

    return LoadPlan(requests)


def get_chunk_readitem(readitem, ep_rank, operate_dim=0):
    """Get the chunk read item for expert parallelism.

    Args:
        readitem (ReadItem): The original read item.
        ep_rank (int): The expert parallel rank.
        operate_dim (int): The dimension along which to chunk the tensor. Default is 0.

    Returns:
        ReadItem: The chunked read item.
    """
    storage_offsets = readitem.storage_offsets
    lengths = readitem.lengths
    if len(storage_offsets) != len(lengths):
        raise ValueError("storage_offsets and lengths must have the same size.")
    offset_list = []
    for i, (a, b) in enumerate(zip(storage_offsets, lengths)):
        if i == operate_dim:
            offset_list.append(a + b * ep_rank)
        else:
            offset_list.append(a)
    new_storage_offsets = torch.Size(offset_list)
    return replace(readitem, storage_offsets=new_storage_offsets)


def get_check_moe_func(model):

    ep_params = set()
    recompute_prefix = "_checkpoint_wrapped_module."
    for name, param in model.named_parameters():
        if isinstance(param, DTensor):
            mesh = getattr(param, "device_mesh", None)
            names = getattr(mesh, "mesh_dim_names", []) if mesh is not None else []
            if "efsdp" in names:
                ep_params.add(name.replace(recompute_prefix, ""))

    def check_moe_fn(param_name):
        nonlocal ep_params
        nonlocal recompute_prefix
        if recompute_prefix in param_name:
            param_name = param_name.replace(recompute_prefix, "")
        return any([param_name.endswith(moe_param) for moe_param in ep_params])

    return check_moe_fn
