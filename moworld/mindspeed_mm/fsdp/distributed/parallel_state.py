# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import os
import types
import logging
from dataclasses import dataclass, asdict, fields
from functools import reduce
from typing import Optional

import torch
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
from torch.distributed import ProcessGroup

from mindspeed_mm.fsdp.utils.device import get_device_type
from mindspeed_mm.fsdp.utils.decorators import Singleton


logger = logging.getLogger(__name__)


def get_last_mesh_dim(mesh_shape):
    last_mesh = torch.distributed.get_world_size()

    for shape in mesh_shape:
        if last_mesh % shape != 0:
            raise AssertionError("World size is not divisible by mesh group {}".format(mesh_shape))
        last_mesh //= shape
    return last_mesh


@dataclass
class ParallelState(metaclass=Singleton):
    data_parallel_size: int = 1
    fully_shard_parallel_size: int = 1
    tensor_parallel_size: int = 1
    ring_attention_size: int = 1
    ulysses_parallel_size: int = 1

    expert_parallel_size: int = 1
    expert_fully_shard_parallel_size: int = 1
    expert_data_parallel_size: int = 1

    device_mesh_map: dict[str, DeviceMesh] = None

    def __post_init__(self):
        """Initialize device meshes and parallel groups after dataclass instantiation."""
        if self.device_mesh_map is None:
            self.device_mesh_map = dict()

        # create DP/CP/Ulysses/TP groups
        dp_shard_size = self.fully_shard_parallel_size // self.ring_attention_size // self.ulysses_parallel_size
        dp_replicate_size = self.data_parallel_size // dp_shard_size
        # Define mesh dimensions and their sizes
        mesh_dim_names = ("dp_replicate", "dp_shard", "ulysses", "ring", "tp")
        mesh_shape = (
            dp_replicate_size,
            dp_shard_size,
            self.ulysses_parallel_size,
            self.ring_attention_size,
            self.tensor_parallel_size,
        )
        self.device_mesh = init_device_mesh(device_type=get_device_type(), mesh_shape=mesh_shape, mesh_dim_names=mesh_dim_names)

        # Flatten mesh dimensions to create hierarchical groups
        # Combine dp_replicate and dp_shard into dp (data parallel) group
        self.device_mesh[("dp_replicate", "dp_shard")]._flatten(mesh_dim_name="dp")
        # Combine ulysses and ring into cp group
        self.device_mesh[("ulysses", "ring")]._flatten(mesh_dim_name="cp")
        # Combine dp_shard, ulysses, ring into dp_shard_cp group
        self.device_mesh[("dp_shard", "ulysses", "ring")]._flatten(mesh_dim_name="dp_shard_cp")
        # Combine all dp and cp dimensions into dp_cp group
        self.device_mesh[("dp_replicate", "dp_shard", "ulysses", "ring")]._flatten(mesh_dim_name="dp_cp")

        # Register helper functions for all mesh dimensions
        self.register_funcs(self.device_mesh, ["dp", "cp", "ulysses", "ring", "tp"])


        # create EP_DP/EP groups
        mesh_dim_names = ('edp', 'efsdp', 'ep')
        mesh_shape = (self.expert_fully_shard_parallel_size, self.expert_parallel_size,)
        self.expert_data_parallel_size = get_last_mesh_dim(mesh_shape)
        mesh_shape = (self.expert_data_parallel_size,) + mesh_shape

        self.ep_fsdp_device_mesh = init_device_mesh(device_type=get_device_type(), mesh_shape=mesh_shape, mesh_dim_names=mesh_dim_names)
        self.register_funcs(self.ep_fsdp_device_mesh, mesh_dim_names)

        if torch.distributed.get_rank() == 0:
            logger.info(f'Parallel state initialized:\n {self.__str__()}')

    def __str__(self):
        info = ''
        for name, _ in self.device_mesh_map.items():
            enable = self.is_group_enable(name)
            size = self.get_group_size(name)
            mesh = self.get_device_mesh(name)
            info += f'[{name}] = {enable} | Group size: {size} | device mesh:{mesh} \n'
        info += f'[fsdp] = {True} | Group size: {self.get_fsdp_group_size()} | device mesh:{self.get_fsdp_device_mesh()} \n'
        return info

    # ----------------------------- FSDP ----------------------------- #
    def get_fsdp_group(self) -> Optional["ProcessGroup"]:
        return self.device_mesh.get_group("dp_cp")

    def get_fsdp_group_size(self) -> Optional["ProcessGroup"]:
        return self.device_mesh.get_group("dp_cp").size()

    def get_fsdp_device_mesh(self) -> "DeviceMesh":
        if self.device_mesh.get_group("dp_replicate").size() > 1:
            return self.device_mesh["dp_replicate", "dp_shard_cp"]
        else:
            return self.device_mesh["dp_shard_cp"]

    def get_fsdp_rank(self) -> int:
        return self.device_mesh.get_local_rank("dp_cp")

    def is_group_enable(self, mesh_name: str) -> bool:
        if mesh_name in self.device_mesh_map:
            return self.get_group_size(mesh_name) > 1
        else:
            return False

    def get_group(self, mesh_name: str):
        if mesh_name in self.device_mesh_map:
            return self.device_mesh_map[mesh_name].get_group(mesh_name)
        else:
            raise RuntimeError(f"Mesh group {mesh_name} not found.")

    def get_group_size(self, mesh_name: str):
        if mesh_name in self.device_mesh_map:
            return torch.distributed.get_world_size(self.device_mesh_map[mesh_name].get_group(mesh_name))
        else:
            raise RuntimeError(f"Mesh group {mesh_name} not found.")

    def get_rank(self, mesh_name: str):
        if mesh_name in self.device_mesh_map:
            return self.device_mesh_map[mesh_name].get_local_rank(mesh_name)
        else:
            raise RuntimeError(f"Mesh group {mesh_name} not found.")

    def get_device_mesh(self, mesh_name: str):
        if mesh_name in self.device_mesh_map:
            return self.device_mesh_map[mesh_name][mesh_name]
        else:
            raise RuntimeError(f"Mesh group {mesh_name} not found.")

    def register_funcs(self, device_mesh, mesh_names):
        """
        Dynamically register helper methods for each mesh dimension.

        For each mesh dimension, creates methods like:
        - is_{mesh_name}_enable()
        - get_{mesh_name}_group()
        - get_{mesh_name}_group_size()
        - get_{mesh_name}_rank()
        - get_{mesh_name}_device_mesh()
        """
        def get_methods(name):
            def is_enable_method(self):
                return self.is_group_enable(name)

            def get_group_method(self):
                return self.get_group(name)

            def get_size_method(self):
                return self.get_group_size(name)

            def get_rank_method(self):
                return self.get_rank(name)

            def get_mesh_method(self):
                return self.get_device_mesh(name)

            return is_enable_method, get_group_method, get_size_method, get_rank_method, get_mesh_method

        for mesh_name in mesh_names:
            self.device_mesh_map[mesh_name] = device_mesh
            is_enable, get_group, get_size, get_rank, get_mesh = get_methods(mesh_name)
            setattr(self, 'is_{}_enable'.format(mesh_name), types.MethodType(is_enable, self))
            setattr(self, 'get_{}_group'.format(mesh_name), types.MethodType(get_group, self))
            setattr(self, 'get_{}_group_size'.format(mesh_name), types.MethodType(get_size, self))
            setattr(self, 'get_{}_rank'.format(mesh_name), types.MethodType(get_rank, self))
            setattr(self, 'get_{}_device_mesh'.format(mesh_name), types.MethodType(get_mesh, self))

_PARALLEL_STATE: "ParallelState" = None


def init_parallel_state(
    data_parallel_size: int = 1,
    fully_shard_parallel_size: int = 1,
    tensor_parallel_size: int = 1,
    ring_attention_size: int = 1,
    ulysses_parallel_size: int = 1,
    expert_parallel_size: int = 1,
    expert_fully_shard_parallel_size: int = 1,
    expert_data_parallel_size: int = 1,
    **kwargs
):
    global _PARALLEL_STATE
    _PARALLEL_STATE = ParallelState(
        data_parallel_size=data_parallel_size,
        fully_shard_parallel_size=fully_shard_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        ring_attention_size=ring_attention_size,
        ulysses_parallel_size=ulysses_parallel_size,
        expert_parallel_size=expert_parallel_size,
        expert_fully_shard_parallel_size=expert_fully_shard_parallel_size,
        expert_data_parallel_size=expert_data_parallel_size
    )

    return _PARALLEL_STATE


def get_parallel_state() -> ParallelState:
    """
    Get the global ParallelState singleton instance.

    Returns:
        The ParallelState instance.

    Note:
        If ParallelState has not been initialized, returns a default single-process state.
    """
    global _PARALLEL_STATE
    if _PARALLEL_STATE is None:
        logger.warning_once("Parallel state has not been initialized. returning default Single-process state.")
        return ParallelState()
    return _PARALLEL_STATE


def is_parallel_state_initialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    global _PARALLEL_STATE
    return _PARALLEL_STATE is not None
