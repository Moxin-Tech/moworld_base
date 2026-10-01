import os
import logging
import gc
from glob import glob
from typing import Optional
from dataclasses import dataclass
from tqdm import tqdm

import torch
from torch.distributed.tensor import DTensor

from mindspeed.fsdp.utils.log import print_rank
from mindspeed_mm.fsdp.checkpoint.dcp_utils import load_metadata, extract_metadata, partial_load_dcp_state_dict
from mindspeed_mm.fsdp.utils.utils import to_empty_if_needed, tensor_to_dtensor
from mindspeed_mm.fsdp.utils.device import get_device_type, empty_cache


logger = logging.getLogger(__name__)


@dataclass
class ParamInfo:
    """
    Metadata for broadcasting rank 0 checkpoint to all ranks.
    """
    name: Optional[str] = None
    shape: Optional[torch.Size] = None
    dtype: Optional[torch.dtype] = None
    prefix: Optional[str] = None


def chunk_list(lst, chunk_size):
    """Yield successive chunk_size-sized chunks from lst."""
    k, m = divmod(len(lst), chunk_size)

    return [lst[i * k + min(i, m): (i + 1) * k + min(i + 1, m)]
            for i in range(chunk_size)]


@torch.no_grad()
def rank0_load_and_broadcast_weights(load_state, storage_reader):
    MODEL = "model"
    OPTIMIZER = "optimizer"

    model = load_state[MODEL].model

    model_state_dict = load_state[MODEL].state_dict()
    params_to_load = set(model_state_dict.keys())
    dcp_keys = [f"{MODEL}.{key}" for key in model_state_dict.keys()]

    if OPTIMIZER in load_state:
        optim_state_dict = load_state[OPTIMIZER].state_dict()
        params_to_load.update(optim_state_dict.keys())
        dcp_keys.extend([f"{OPTIMIZER}.{key}" for key in optim_state_dict.keys()])
    else:
        optim_state_dict = None

    torch_device = torch.device(get_device_type())
    global_rank = torch.distributed.get_rank()

    shard_info_list = []
    if global_rank == 0:
        metadata = load_metadata(storage_reader)

        fqn2file = {}
        for key, value in metadata.storage_data.items():
            fqn = key.fqn
            if fqn not in fqn2file:
                fqn2file[fqn] = set()
            fqn2file[fqn].add(value.relative_path)

        shard_dict = {k: v for k, v in fqn2file.items() if len(v) > 1}
        unshard_dict = {k: v for k, v in fqn2file.items() if len(v) == 1 and not k.startswith(f"{OPTIMIZER}.")}
        optim_unshard_dict = {k: v for k, v in fqn2file.items() if len(v) == 1 and k.startswith(f"{OPTIMIZER}.")}

        shard_info_list = []
        if len(optim_unshard_dict) > 0:
            fqn2file_list = sorted(optim_unshard_dict.items(), key=lambda x: x[0])
            selected_keys = [fqn[0] for fqn in fqn2file_list if fqn[0] in dcp_keys]
            shard_info_list.append(selected_keys)

        def register_shard_info(info_dict):
            nonlocal dcp_keys
            nonlocal shard_info_list
            file2fqn = {}
            for key, value in info_dict.items():
                files_tuple = tuple(sorted(value))
                if files_tuple not in file2fqn:
                    file2fqn[files_tuple] = set()
                file2fqn[files_tuple].add(key)
            file2fqn_list = sorted(file2fqn.items(), key=lambda x: x[0])
            for files_tuple, fqn_set in file2fqn_list:
                selected_keys = [fqn for fqn in fqn_set if fqn in dcp_keys]
                shard_info_list.append(selected_keys)
            return len(file2fqn_list)

        if len(shard_dict) == 0:
            register_shard_info(unshard_dict)
        else:
            shard_info_count = register_shard_info(shard_dict)
            if shard_info_count == 1:
                shard_info_list = shard_info_list[:-shard_info_count]
            if shard_info_count <= 1:
                fqn2file_list = sorted(shard_dict.items(), key=lambda x: x[0])
                shard_size = len(fqn2file_list[0][1])
                file_num = len(glob(os.path.join(storage_reader.path, "*.distcp")))
                shard_count = max(file_num // shard_size, 1)
                for fqn2file_elem in chunk_list(fqn2file_list, shard_count):
                    selected_keys = [fqn[0] for fqn in fqn2file_elem if fqn[0] in dcp_keys]
                    shard_info_list.append(selected_keys)
    else:
        shard_info_list = []

    shard_count = len(shard_info_list)
    shard_count_tensor = torch.tensor(shard_count, dtype=torch.int64, device=torch_device)
    torch.distributed.broadcast(shard_count_tensor, src=0)
    shard_count = int(shard_count_tensor.item())

    shard_iterable = tqdm(
        range(shard_count),
        desc="Loading checkpoint shards",
        disable=int(os.getenv("LOCAL_RANK", "-1")) > 0,
    )

    for shard_id in shard_iterable:
        if shard_id == 0 and optim_state_dict is not None:
            if global_rank == 0:
                shard_metadata = extract_metadata(shard_info_list[shard_id], metadata)
                shard_state_dict = partial_load_dcp_state_dict(shard_metadata, storage_reader)
            else:
                shard_state_dict = {}

            broadcast_list = [shard_state_dict]
            torch.distributed.broadcast_object_list(broadcast_list, src=0)
            shard_state_dict = broadcast_list[0]

            for key, value in shard_state_dict[OPTIMIZER].items():
                if key in optim_state_dict:
                    optim_state_dict[key] = shard_state_dict[OPTIMIZER][key]
                    params_to_load.discard(key)

            load_state[OPTIMIZER].load_state_dict(optim_state_dict)
            continue

        if global_rank == 0:
            shard_metadata = extract_metadata(shard_info_list[shard_id], metadata)
            shard_state_dict = partial_load_dcp_state_dict(shard_metadata, storage_reader)

            param_info_list = []
            if MODEL in shard_state_dict or OPTIMIZER in shard_state_dict:
                for prefix in shard_state_dict:
                    prefix_state_dict = shard_state_dict[prefix]
                    param_info_list.extend([
                        ParamInfo(name=k, shape=v.shape, dtype=v.dtype, prefix=prefix)
                        for k, v in prefix_state_dict.items()
                    ])
            else:
                param_info_list.extend([
                    ParamInfo(name=k, shape=v.shape, dtype=v.dtype)
                    for k, v in shard_state_dict.items()
                ])
        else:
            param_info_list = []

        broadcast_list = [param_info_list]
        torch.distributed.broadcast_object_list(broadcast_list, src=0)
        param_info_list = broadcast_list[0]

        for param_info in param_info_list:
            param_name = param_info.name
            if param_name not in params_to_load:
                continue

            tensor = None
            if global_rank != 0:
                tensor = torch.empty(param_info.shape, dtype=param_info.dtype, device=torch_device)
            else:
                if param_info.prefix is None:
                    tensor = shard_state_dict[param_name].to(torch_device, non_blocking=True)
                else:
                    tensor = shard_state_dict[param_info.prefix][param_name].to(torch_device, non_blocking=True)

            torch.distributed.broadcast(tensor, src=0)

            params_to_load.discard(param_name)
            if param_info.prefix == OPTIMIZER:
                target_state_dict = optim_state_dict
            else:
                target_state_dict = model_state_dict
            target_tensor = target_state_dict[param_name]

            if isinstance(target_tensor, DTensor):
                device_mesh = getattr(target_tensor, "device_mesh", None)
                placements = getattr(target_tensor, "placements", None)
                target_state_dict[param_name].copy_(tensor_to_dtensor(tensor, device_mesh, placements))
            else:
                target_state_dict[param_name].copy_(tensor)

            del tensor

        gc.collect()
        empty_cache()

    if len(params_to_load) > 0:
        print_rank(logger.warning, f"These weights were not loaded from the checkpoint, param keys: {params_to_load}.")
    print_rank(logger.info, f"Finished loading and broadcasting checkpoint tensors from rank 0.")
