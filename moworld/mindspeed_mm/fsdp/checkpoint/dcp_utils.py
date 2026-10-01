from typing import Optional, List, cast
import inspect
import warnings
import dataclasses
import os
from functools import reduce

import torch
from torch import Tensor
from torch.distributed.checkpoint.metadata import Metadata, STATE_DICT_TYPE
from torch.distributed.checkpoint.storage import StorageWriter, StorageReader
from torch.distributed.checkpoint.planner import SavePlanner, LoadPlanner, ReadItem, LoadItemType, LoadPlan
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner, _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.logger import _dcp_method_logger
from torch.distributed.checkpoint.filesystem import _StoragePrefix, _StorageInfo
from torch.distributed.checkpoint import FileSystemReader


def partial_save_dcp_state_dict(
    state_dict: STATE_DICT_TYPE,
    storage_writer: StorageWriter,
    planner: Optional[SavePlanner] = None,
    part_idx: int = 0
):
    """
    Save a partial shard of a Distributed Checkpoint (DCP) state_dict.

    This function enables a single process (e.g., in a single-machine or testing context)
    to save its portion of model weights as torch_dcp format. It coordinates between
    a SavePlanner (which decides how tensors are laid out) and a StorageWriter (which
    handles actual I/O). The function returns global metadata (typically populated only
    by a coordinator rank in multi-process settings) and the results of write operations.

    Args:
        state_dict (STATE_DICT_TYPE): The local subset of the model state dict to save.
        storage_writer (StorageWriter): Handles writing data to persistent storage.
        planner (Optional[SavePlanner]): Custom save planner; uses DefaultSavePlanner if None.
        part_idx (int): Offset index used to generate unique storage prefixes (e.g., "__2_").

    Returns:
        Tuple[Optional[Metadata], List[Any]]:
            - global_metadata: checkpoint metadata.
            - all_writes: Results of the write operations.
    """

    # Use default planner if none is provided
    if planner is None:
        planner = DefaultSavePlanner()

    # Initialize global metadata; will be set during global planning phase
    global_metadata = None

    ckpt_kwargs = {}
    ckpt_id = getattr(storage_writer, "checkpoint_id", None)
    if ckpt_id is not None:
        ckpt_kwargs["checkpoint_id"] = ckpt_id

    @_dcp_method_logger(**ckpt_kwargs)
    def local_step():
        storage_meta = storage_writer.storage_meta()
        if "storage_meta" not in inspect.signature(planner.set_up_planner).parameters:
            warnings.warn(
                "The function definition for SavePlanner.set_up_planner has been updated"
                " to include the storage_meta argument. Please update your implementation"
                " to include this parameter."
            )
            planner.set_up_planner(state_dict, True)
        else:
            planner.set_up_planner(
                state_dict=state_dict,
                storage_meta=storage_meta,
                is_coordinator=True
            )
        storage_writer.set_up_storage_writer(True)

        local_plan = planner.create_local_plan()
        local_plan = storage_writer.prepare_local_plan(local_plan)
        return local_plan

    @_dcp_method_logger(**ckpt_kwargs)
    def global_step(all_local_plans):
        nonlocal global_metadata

        all_local_plans, global_metadata = planner.create_global_plan(all_local_plans)
        all_local_plans = storage_writer.prepare_global_plan(all_local_plans)
        all_local_plans = [
            dataclasses.replace(plan, storage_data=_StoragePrefix(f"__{i+part_idx}_"))
            for i, plan in enumerate(all_local_plans)
        ]
        return all_local_plans

    local_plan = local_step()
    all_local_plan = global_step([local_plan])[0]

    @_dcp_method_logger(**ckpt_kwargs)
    def write_data():
        final_local_plan = planner.finish_plan(all_local_plan)
        all_writes = storage_writer.write_data(final_local_plan, planner)
        all_writes.wait()
        return all_writes.value()

    all_writes = write_data()

    # Return metadata and write results for potential post-processing or validation
    return global_metadata, all_writes


def save_metadata(
    global_metadata,
    all_writes,
    storage_writer: StorageWriter
):
    """
    Finalize a Distributed Checkpoint (DCP) by saving its global metadata.

    Args:
        global_metadata: The consolidated metadata describing the entire checkpoint.
                         Typically generated during the global planning phase.
        all_writes: Results from previous tensor write operations (e.g., list of
                    written file names or async write futures). Used by the writer
                    to finalize references in the metadata.
        storage_writer (StorageWriter): The writer responsible for persisting data
                                       and metadata to the underlying storage backend.
    """
    ckpt_kwargs = {}
    ckpt_id = getattr(storage_writer, "checkpoint_id", None)
    if ckpt_id is not None:
        ckpt_kwargs["checkpoint_id"] = ckpt_id

    @_dcp_method_logger(**ckpt_kwargs)
    def finish_checkpoint():
        storage_writer.finish(metadata=global_metadata, results=all_writes)
        return global_metadata

    return finish_checkpoint()


def merge_meta_info(
    global_meta_infos: List[Metadata],
):
    """
    Merge multiple DCP (Distributed Checkpoint) metadata objects into a single unified Metadata instance.

    This function is typically used when a checkpoint has been saved in multiple shards or parts
    (e.g., via partial saves), each producing its own Metadata object. The merge combines:
      - `state_dict_metadata`: mapping of tensor names to their storage/sharding info,
      - `planner_data`: auxiliary data used by the SavePlanner (e.g., layout hints, version info).

    It assumes that keys across shards are disjoint (i.e., no overlapping tensor names),
    so simple dictionary merging via `**` is safe.

    Args:
        global_meta_infos (List[Metadata]): A list of Metadata objects from individual shards.
                                            Must be non-empty to produce a valid result.

    Returns:
        Metadata: A merged Metadata object containing the union of all input metadata.
                  Returns None if the input list is empty.
    """

    # Use functools.reduce to iteratively merge all Metadata instances.
    # Start with the first element as the accumulator, then merge in the rest one by one.
    merged_data = reduce(
        lambda acc, x: acc.__class__(
            state_dict_metadata={**acc.state_dict_metadata, **x.state_dict_metadata},
            planner_data={**acc.planner_data, **x.planner_data}
        ),
        global_meta_infos[1:],  # All elements after the first
        global_meta_infos[0]    # Initial accumulator
    ) if global_meta_infos else None    # Only merge if the list is non-empty

    return merged_data


def load_metadata(
    storage_reader: StorageReader
):
    """
    Load the global metadata of a Distributed Checkpoint (DCP) from persistent storage.

    This function uses a `StorageReader` to read the checkpoint's central metadata file,
    which typically contains:
      - `state_dict_metadata`
      - `planner_data`

    The metadata is essential for correctly reconstructing the full state dict during loading.

    Args:
        storage_reader (StorageReader): An object capable of reading checkpoint data from
                                        the underlying storage backend (e.g., filesystem, cloud).

    Returns:
        Metadata: The deserialized global metadata object describing the entire checkpoint.
    """

    ckpt_kwargs = {}
    ckpt_id = getattr(storage_reader, "checkpoint_id", None)
    if ckpt_id is not None:
        ckpt_kwargs["checkpoint_id"] = ckpt_id

    @_dcp_method_logger(**ckpt_kwargs)
    def read_metadata():
        metadata = storage_reader.read_metadata()
        return metadata

    return read_metadata()


def partial_load_dcp_state_dict(
    metadata: Metadata,
    storage_reader: StorageReader,
    planner: Optional[LoadPlanner] = None,
):
    """
    Load a partial subset of a Distributed Checkpoint (DCP) state dictionary.

    This function is designed for scenarios where only a portion of the full model
    needs to be loaded (e.g., loading specific layers or shards). The input `metadata`
    describes only the relevant tensors to load—not the entire checkpoint.

    It uses a LoadPlanner and StorageReader to:
      - Plan which tensors to load based on the provided metadata,
      - Read the corresponding data from storage,
      - Populate a local `state_dict` in-place.

    Note: The resulting `state_dict` will contain only the keys covered by the input `metadata`.

    Args:
        metadata (Metadata): Partial metadata describing the subset of tensors to load.
                             Must include entries in `state_dict_metadata` for the desired keys.
        storage_reader (StorageReader): Handles reading tensor data from persistent storage.
        planner (Optional[LoadPlanner]): Custom load planner. If not provided, uses
                                        `_EmptyStateDictLoadPlanner`, which initializes an
                                        empty state dict and populates it during loading.

    Returns:
        STATE_DICT_TYPE: A state dictionary containing only the tensors specified in `metadata`.
    """
    # Initialize an empty state dict; it will be populated in-place by the planner during loading
    state_dict: STATE_DICT_TYPE = {}

    if planner is None:
        planner = _EmptyStateDictLoadPlanner()

    ckpt_kwargs = {}
    ckpt_id = getattr(storage_reader, "checkpoint_id", None)
    if ckpt_id is not None:
        ckpt_kwargs["checkpoint_id"] = ckpt_id

    @_dcp_method_logger(**ckpt_kwargs)
    def local_step():
        planner.set_up_planner(state_dict, metadata, True)
        storage_reader.set_up_storage_reader(metadata, True)

        local_plan = planner.create_local_plan()
        local_plan = storage_reader.prepare_local_plan(local_plan)
        return local_plan

    @_dcp_method_logger(**ckpt_kwargs)
    def global_step(all_local_plans):
        all_local_plans = planner.create_global_plan(all_local_plans)
        all_local_plans = storage_reader.prepare_global_plan(all_local_plans)
        return all_local_plans

    local_plan = local_step()
    central_plan = global_step([local_plan])[0]

    @_dcp_method_logger(**ckpt_kwargs)
    def read_data():
        final_local_plan = planner.finish_plan(central_plan)
        all_reads = storage_reader.read_data(final_local_plan, planner)
        all_reads.wait()

    read_data()
    return state_dict


def extract_metadata(
    selected_keys: List[str],
    metadata: Metadata
):
    """
    Extract a partial Metadata object containing only the entries corresponding to the given keys.

    This function filters a full DCP (Distributed Checkpoint) Metadata instance to produce a
    reduced version that includes only the tensors (or state dict keys) specified in `selected_keys`.
    It selectively subsets three core components of the metadata:
      - `state_dict_metadata`
      - `storage_data`
      - `planner_data`

    The resulting partial metadata can be used to load or save only a subset of the checkpoint,
    enabling efficient partial operations (e.g., loading specific layers of a large model).

    Args:
        selected_keys (List[str]): A list of fully qualified names (FQNs) of tensors to retain.
                                   Only metadata entries matching these keys will be included.
        metadata (Metadata): The complete metadata object from a DCP checkpoint.

    Returns:
        Metadata: A new Metadata instance containing only the entries associated with `selected_keys`.
    """

    # select metadata
    partial_state_dict_metadata = {}
    partial_storage_data_metadata = {}
    partial_planner_data = {}

    # Filter meta_items include only entries whose keys are in selected_keys
    for dcp_key, tensor_storage_metadata in metadata.state_dict_metadata.items():
        if dcp_key in selected_keys:
            partial_state_dict_metadata.update({dcp_key: tensor_storage_metadata})
    for metadataindex, storage_info in metadata.storage_data.items():
        if metadataindex.fqn in selected_keys:
            partial_storage_data_metadata.update({metadataindex: storage_info})
    for dcp_key, state_dict_key_tuple in metadata.planner_data.items():
        if dcp_key in selected_keys:
            partial_planner_data.update({dcp_key: state_dict_key_tuple})

    # Construct and return a new Metadata object with the filtered components
    partial_metadata = Metadata(
        state_dict_metadata=partial_state_dict_metadata,
        storage_data=partial_storage_data_metadata,
        planner_data=partial_planner_data
    )

    return partial_metadata
