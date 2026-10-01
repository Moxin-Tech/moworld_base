import os
import glob
import re
import json
from pathlib import Path
import stat
import shutil
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from checkpoint.common.permissions import set_directory_permissions
from checkpoint.common.constant import FILE_MODE
MEGATRON_LASTEST_ITERATION_FILE_NAME = "latest_checkpointed_iteration.txt"
MEGATRON_MODEL_KEY = "model"
MEGATRON_CKPT_NAME = "model_optim_rng.pt"


def load_from_mm(load_dir):
    flags = os.O_RDONLY
    mode = stat.S_IRUSR

    iteration_path = os.path.join(load_dir, MEGATRON_LASTEST_ITERATION_FILE_NAME)
    with os.fdopen(os.open(iteration_path, flags, mode)) as f:
        latest_checkpointed_iteration = f.readline()

    if latest_checkpointed_iteration == "release":
        directory = "release"
    else:
        directory = "iter_{:07d}".format(int(latest_checkpointed_iteration))

    pp_tp_state_dicts = {}
    sub_dirs = sorted(os.listdir(os.path.join(load_dir, directory)))
    enable_pp = len(sub_dirs[0].split('_')) == 4

    print(
        f"[mm-load] loading {len(sub_dirs)} checkpoint shard(s) from "
        f"{os.path.join(load_dir, directory)}",
        flush=True,
    )
    for sub_dir in tqdm(sub_dirs, desc="Load MM shards", unit="shard"):
        state_dict_path = os.path.join(load_dir, directory, sub_dir, MEGATRON_CKPT_NAME)
        state_dict = torch.load(state_dict_path, map_location='cpu', weights_only=False)
        if enable_pp:
            tp_rank, pp_rank = map(int, (sub_dir.split('_')[2:4]))
            vpp_state_dicts = []
            for key, vpp_state_dict in state_dict.items():
                match = re.match(r'model(\d)', key)
                if match:
                    number = int(match.group(1))
                    vpp_state_dicts.append((number, vpp_state_dict))
            vpp_state_dicts.sort(key=lambda x: x[0])
            state_dict = [vpp_state_dict for _, vpp_state_dict in vpp_state_dicts]
        else:
            pp_rank = 0
            tp_rank = int(sub_dir.split('_')[2])
            state_dict = state_dict[MEGATRON_MODEL_KEY]
        pp_tp_state_dicts[(pp_rank, tp_rank)] = state_dict

    pp_size = max([pp_tp_rank[0] for pp_tp_rank in pp_tp_state_dicts.keys()]) + 1
    tp_size = max([pp_tp_rank[1] for pp_tp_rank in pp_tp_state_dicts.keys()]) + 1

    state_dicts = []
    for pp_rank in range(pp_size):
        tp_state_dicts = []
        for tp_rank in range(tp_size):
            tp_state_dicts.append(pp_tp_state_dicts[((pp_rank, tp_rank))])
        state_dicts.append(tp_state_dicts)

    return state_dicts


def save_as_mm(save_dir, state_dicts, latest_checkpointed_iteration="release"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    flags = os.O_WRONLY | os.O_CREAT
    mode = stat.S_IWUSR | stat.S_IRUSR

    iteration_path = os.path.join(save_dir, MEGATRON_LASTEST_ITERATION_FILE_NAME)
    with os.fdopen(os.open(iteration_path, flags, mode), 'w') as fout:
        fout.write(latest_checkpointed_iteration)

    if latest_checkpointed_iteration == "release":
        directory = "release"
    else:
        directory = "iter_{:07d}".format(latest_checkpointed_iteration)

    enable_pp = len(state_dicts) > 1
    shard_count = sum(len(tp_state_dicts) for tp_state_dicts in state_dicts)
    print(f"[mm-save] writing {shard_count} checkpoint shard(s) to {save_dir}", flush=True)
    with tqdm(total=shard_count, desc="Write MM shards", unit="shard") as progress:
        for pp_rank, tp_state_dicts in enumerate(state_dicts):
            for tp_rank, state_dict in enumerate(tp_state_dicts):
                if enable_pp:
                    state_dict_save_dir = os.path.join(save_dir, directory, f"mp_rank_{tp_rank:02d}_{pp_rank:03d}")
                else:
                    state_dict_save_dir = os.path.join(save_dir, directory, f"mp_rank_{tp_rank:02d}")
                os.makedirs(state_dict_save_dir)
                save_path = os.path.join(state_dict_save_dir, MEGATRON_CKPT_NAME)
                save_dict = {}

                if isinstance(state_dict, list):
                    # enable vpp
                    vpp_size = len(state_dict)
                    save_dict = {f"model{vpp_rank}": state_dict[vpp_rank] for vpp_rank in range(vpp_size)}
                    save_dict['checkpoint_version'] = 3.0
                else:
                    save_dict[MEGATRON_MODEL_KEY] = state_dict
                torch.save(save_dict, save_path)
                progress.update(1)

    set_directory_permissions(Path(save_dir))


def load_from_hf(hf_dir):
    if not os.path.exists(hf_dir):
        raise FileNotFoundError(f"Directory not found: {hf_dir}")

    search_pattern = os.path.join(hf_dir, '**', '*.safetensors')
    files = sorted(glob.glob(search_pattern, recursive=True))

    if not files or len(files) == 0:
        raise FileNotFoundError(f"No .safetensors files found in directory: {hf_dir}")

    print(f"[hf-load] loading {len(files)} safetensors file(s) from {hf_dir}", flush=True)
    state_dict = {}
    for safe_path in tqdm(files, desc="Load safetensors", unit="file"):
        state_dict.update(load_file(str(safe_path), device='cpu'))

    print(f"[hf-load] loaded {len(state_dict)} tensor(s)", flush=True)

    return state_dict


def save_as_hf(state_dict, hf_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # copy diffusion_pytorch_model.safetensors.index.json and config.json from hf_dir to save_dir
    shutil.copy2(os.path.join(hf_dir, "diffusion_pytorch_model.safetensors.index.json"), save_dir)
    shutil.copy2(os.path.join(hf_dir, "config.json"), save_dir)

    # split state_dict by index.json
    index_json_path = os.path.join(save_dir, "diffusion_pytorch_model.safetensors.index.json")
    with open(index_json_path, "r", encoding="utf-8") as file:
        weight_map = json.load(file)["weight_map"]

    state_dicts = []
    for key, value in weight_map.items():
        index = int(value.split("-")[1])
        while index > len(state_dicts):
            state_dicts.append({})
        state_dicts[index - 1][key] = state_dict[key]

    metadata = {"format": "pt"}

    for index, state_dict in enumerate(state_dicts, start=1):
        name = f'model-{index:05}-of-{len(state_dicts):05}.safetensors'
        save_file(state_dict, os.path.join(save_dir, name), metadata=metadata)

    set_directory_permissions(Path(save_dir))


def load_pt(source_path, module_name=None):
    state_dict = torch.load(source_path, map_location='cpu')
    if module_name:
        state_dict = state_dict[module_name]
    return state_dict


def save_as_pt(state_dict, target_path):
    torch.save(state_dict, target_path)
    os.chmod(target_path, FILE_MODE)


def load_from_layerzero(source_path, iteration=None, prefix=None, ema_model=False, for_release=True):
    import mindspeed.megatron_adaptor
    from mindspeed.core.distributed.layerzero.state.scripts.layerzero_checkpointer import LayerzeroCheckpoint
    from mindspeed.core.distributed.layerzero.state.scripts.convert_to_megatron import _create_rank_checkpoint

    if ema_model:
        from mindspeed.core.distributed.layerzero.state.scripts.layerzero_checkpointer import remove_model_prefix
        remove_model_prefix(prefix)

    # If iter is none, the result of the last save will be converted by default.
    if iteration is None:
        iteration = _get_latest_iter_number(source_path)

    source_path = os.path.join(source_path, 'iter_{:07d}'.format(iteration))
    layerzero_checkpoint = LayerzeroCheckpoint(source_path)

    # Notice: no support TP and PP
    layerzero_checkpoint = _create_rank_checkpoint(
        layerzero_checkpoint,
        tp_index=0,
        pp_index=0,
        tp_degree=1,
        pp_degree=1,
        for_release=for_release
    )
    return layerzero_checkpoint


def _get_latest_iter_number(directory):
    dir_path = Path(directory)

    # Traverse all subdirectories under the directory,
    # and try to match the directory names in the format of iter_xxxx.
    pattern = re.compile(r'^iter_(\d+)$')
    max_num = -1

    for subdir in dir_path.iterdir():
        if subdir.is_dir():
            match = pattern.match(subdir.name)
            if match:
                num = int(match.group(1))
                if num > max_num:
                    max_num = num

    if max_num == -1:
        raise ValueError("No iter_xxxx directories found.")

    return max_num
