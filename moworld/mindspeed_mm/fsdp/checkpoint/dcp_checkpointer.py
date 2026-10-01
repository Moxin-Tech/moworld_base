# Copyright 2025 Bytedance Ltd. and/or its affiliates
# pylint: skip-file
import gc
import os
from typing import Any, Dict, Optional
import logging

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
)
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.stateful import Stateful

from mindspeed.fsdp.utils.log import print_rank
from mindspeed_mm.fsdp.distributed.parallel_state import get_parallel_state
from mindspeed_mm.fsdp.utils.device import empty_cache, synchronize
from mindspeed_mm.fsdp.checkpoint.checkpointer import CheckpointerBase
from mindspeed_mm.fsdp.checkpoint.utils import (
    get_checkpoint_name,
    read_metadata,
    get_checkpoint_tracker_filename,
    remove_base_layer_keys,
    restore_base_layer_keys,
)
from mindspeed_mm.fsdp.checkpoint.load_utils import rank0_load_and_broadcast_weights


logger = logging.getLogger(__name__)

_EXTRA_STATE_FORMAT = "extra_state_rank_{}.pt"
_EXTRA_STATE_DIR = "extra_state"


class ModelState(Stateful):
    """
    A wrapper around a model to make it stateful.
    Args:
        model (Model): model to wrap.
    """

    def __init__(self, model):
        self.model = model

        # Determine whether this is EP+FSDP2 case
        # If so, we need to restore EP-dim before saving to DCP
        # For FSDP1, it is implemented by FSDPExtension and state_dict hooks
        # which is aumatically triggered by get_model_state_dict
        self.parallel_state = get_parallel_state()

    @torch.no_grad()
    def state_dict(self):
        model_state_dict = get_model_state_dict(model=self.model)
        return model_state_dict

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        """
        perform the reverse operation for state_dict()
        need to drop EP-dim when loading from DCP checkpoints
        so that EP-FSDP would not be confused
        """
        set_model_state_dict(model=self.model, model_state_dict=state_dict)


class LoraModelState(ModelState):
    """
    A wrapper around a lora model to make it stateful.
    Args:
        model (Model): lora model to wrap.
    """

    def __init__(self, model):
        super().__init__(model)
        self.key_mapping = None

    @torch.no_grad()
    def state_dict(self):
        model_state_dict = get_model_state_dict(model=self.model)
        self.key_mapping = remove_base_layer_keys(model_state_dict)
        return model_state_dict

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        restore_base_layer_keys(state_dict, self.key_mapping)
        set_model_state_dict(model=self.model, model_state_dict=state_dict)


class OptimizerState(Stateful):
    """
    A wrapper around an optimizer to make it stateful.

    Args:
        optimizer (Optimizer): optimizer to wrap.
    """

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        # Similar to ModelState, OptimizerState also need to be EP+FSDP2 aware
        self.parallel_state = get_parallel_state()
        self.should_ep_aware = getattr(self.optimizer, "_is_multi_optimizer", False)

    def state_dict(self):
        if self.should_ep_aware:
            # MultiOptimizer is only used for EP+FSDP2 case for now,
            # and it knows how to produce a merged, flattened dict already
            if not self.optimizer._is_multi_optimizer:
                raise ValueError("EP is enabled but optimizer is not a MultiOptimizer instance")
            return self.optimizer.state_dict()

        # Single torch optimizer
        sd = get_optimizer_state_dict(model=self.model, optimizers=self.optimizer)
        return sd

    def load_state_dict(self, state_dict):
        optim_state_from_dcp_load = state_dict
        if self.should_ep_aware:
            # Delegate to MultiOptimizer (it will split/filter correctly)
            self.optimizer.load_state_dict(optim_state_from_dcp_load)
            return

        # Single torch optimizer
        set_optimizer_state_dict(
            model=self.model,
            optimizers=self.optimizer,
            optim_state_dict=optim_state_from_dcp_load,
        )


class DistributedCheckpointer(CheckpointerBase):
    """
    Distributed checkpointer for torch.distributed.checkpoint
    """

    dcp_save_future: Optional[Any] = None
    # Dedicated process group for async saves (created on first use)
    _async_process_group: Optional[Any] = None

    @classmethod
    def save(
        cls,
        path: str,
        state: Dict[str, Any],
        save_async: bool = False,
        iteration: int = None,
        storage_writer: Optional[FileSystemWriter] = None,
    ) -> None:
        """
        save training state to distributed checkpoint

        args:
            path: path to save checkpoint
            state: state to save
            save_async: whether to save asynchronously
            iteration: global steps
            storage_writer: storage writer backend for dcp.save and dcp.async_save. If None, will use FileSystemWriter
        return:
            None
        """
        if "model" not in state:
            raise ValueError("Model must be provided to save a distributed checkpoint.")
        checkpoint_dir = get_checkpoint_name(path, iteration, release=False)
        cls._create_checkpoint_dir(checkpoint_dir)

        # saving extra_state first to gurantee that every saved model/optimizer ckpts have their extra_state saved before them
        cls._save_extra_state(checkpoint_dir=checkpoint_dir, state=state)

        save_state = {"model": ModelState(state["model"])}
        if "optimizer" in state:
            save_state["optimizer"] = OptimizerState(model=state["model"], optimizer=state["optimizer"])  # type: ignore[index]

        if storage_writer is None:
            storage_writer = cls._create_storage_writer(checkpoint_dir)

        cls.execute_save(save_state=save_state, storage_writer=storage_writer, save_async=save_async)

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            tracker_filename = get_checkpoint_tracker_filename(path)
            with open(tracker_filename, 'w', encoding='utf-8') as f:
                f.write(str(iteration))

        print_rank(logger.info, f"Saved checkpoint to {checkpoint_dir}")

    @classmethod
    def load(
        cls,
        path: str,
        state: Dict[str, Any],
        process_group=None,
        storage_reader: Optional[FileSystemReader] = None,
        load_rank0_and_broadcast: bool = False,
        load_strict: bool = False,
        enable_lora: bool = False,
    ) -> bool:
        """
        load training state from distributed checkpoint
        args:
            path: path to load checkpoint
            state: state to load, "model" are required,  "optimizer" and "extra_state" are optional
            process_group: process group for loading checkpoint
            storage_reader: storage reader backend for dcp.load. If None, will use FileSystemReader
            enable_lora: whether to enable LoRA checkpoint loading logic
        return:
            release (bool): whether the loaded checkpoint is a "release" checkpoint
        """
        checkpoint_dir = path

        iteration, release = -1, False
        tracker_filename = get_checkpoint_tracker_filename(checkpoint_dir)
        if os.path.isfile(tracker_filename):
            iteration, release = read_metadata(tracker_filename)

        checkpoint_dir = get_checkpoint_name(checkpoint_dir, iteration, release)

        if state is None:
            raise ValueError("State dict must be provided to load a distributed checkpoint.")

        if "model" not in state:
            raise ValueError("Model must be provided to load a distributed checkpoint.")

        load_state = {"model": LoraModelState(state["model"])} if enable_lora else {"model": ModelState(state["model"])}
        if not release and "optimizer" in state:
            load_state["optimizer"] = OptimizerState(model=state["model"], optimizer=state["optimizer"])  # type: ignore[index]

        if storage_reader is None:
            storage_reader = cls._create_storage_reader(checkpoint_dir)

        if load_rank0_and_broadcast:
            rank0_load_and_broadcast_weights(
                load_state=load_state,
                storage_reader=storage_reader,
            )
        else:
            dcp.load(
                state_dict=load_state,
                storage_reader=storage_reader,
                process_group=process_group,
                planner=DefaultLoadPlanner(allow_partial_load=not load_strict),
            )
        # Note: further per-param DTensor alignment and device fixes happen inside OptimizerState.load_state_dict

        if not release:
            cls._load_extra_state(checkpoint_dir=checkpoint_dir, state=state)

        print_rank(logger.info, f"Loaded checkpoint from {checkpoint_dir}")

        return release

    @classmethod
    def execute_save(
        cls,
        save_state: Dict[str, Any],
        storage_writer: FileSystemWriter,
        save_async: bool,
    ) -> None:
        """Execute DCP save with optional async support."""
        if save_async:
            # Lazily create a dedicated Gloo process group for async DCP saves
            if cls._async_process_group is None:
                cls._async_process_group = dist.new_group(backend="gloo")

            if cls.dcp_save_future is not None:
                logger.info("[RANK %s] waiting for previous DCP saving session to end...", dist.get_rank())
                cls.dcp_save_future.result()
                cls.dcp_save_future = None
                # block until all the ranks resolve their previous dcp async saving
                dist.barrier()

            cls.dcp_save_future = dcp.async_save(
                state_dict=save_state,
                storage_writer=storage_writer,
                process_group=cls._async_process_group,
            )
        else:
            dcp.save(
                state_dict=save_state,
                storage_writer=storage_writer,
            )
            if dist.is_initialized():
                dist.barrier()
            gc.collect()
            empty_cache()
            synchronize()

    # Private helper methods
    @classmethod
    def _create_checkpoint_dir(cls, checkpoint_dir: str) -> None:
        """Create checkpoint directory."""
        os.makedirs(checkpoint_dir, exist_ok=True)

    @classmethod
    def _create_storage_reader(cls, checkpoint_dir: str) -> FileSystemReader:
        """Create storage reader for DCP."""
        return FileSystemReader(checkpoint_dir)

    @classmethod
    def _create_storage_writer(cls, checkpoint_dir: str) -> FileSystemWriter:
        """Create storage writer for DCP."""
        return FileSystemWriter(
            checkpoint_dir,
            thread_count=16,
            single_file_per_rank=True,
            sync_files=False,
        )

    @classmethod
    def _save_extra_state(cls, checkpoint_dir: str, state: Dict[str, Any]) -> None:
        """Save extra_state to checkpoint directory."""
        if "extra_state" not in state:
            logger.warning("extra_state not found in state, skipping extra_state save")
            return

        extra_state_dir = os.path.join(checkpoint_dir, _EXTRA_STATE_DIR)
        os.makedirs(extra_state_dir, exist_ok=True)
        extra_state_path = os.path.join(extra_state_dir, _EXTRA_STATE_FORMAT.format(dist.get_rank()))
        torch.save(
            state["extra_state"],
            extra_state_path,
        )

    @classmethod
    def _load_extra_state(cls, checkpoint_dir: str, state: Dict[str, Any]) -> None:
        """Load extra_state from checkpoint directory."""
        if "extra_state" not in state:
            logger.warning("extra_state not found in state, skipping extra_state load")
            return

        extra_state_dir = os.path.join(checkpoint_dir, _EXTRA_STATE_DIR)
        os.makedirs(extra_state_dir, exist_ok=True)
        extra_state_path = os.path.join(extra_state_dir, _EXTRA_STATE_FORMAT.format(dist.get_rank()))
        state["extra_state"] = torch.load(extra_state_path, weights_only=False)
