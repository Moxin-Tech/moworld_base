import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm

from pydantic import validate_arguments, DirectoryPath, FilePath
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner

from transformers import AutoConfig, AutoProcessor
from safetensors.torch import load_file, save_file

from checkpoint.common.permissions import set_directory_permissions
from checkpoint.common.dcp_utils import load_metadata, extract_metadata, partial_load_dcp_state_dict


@validate_arguments
def load_dcp_state_dict(dcp_checkpoint_dir: DirectoryPath) -> STATE_DICT_TYPE:
    sd: STATE_DICT_TYPE = {}
    _load_state_dict(
        sd,
        storage_reader=FileSystemReader(str(dcp_checkpoint_dir)),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    return sd['model'] if 'model' in sd else sd


def find_safetensors_index(directory: Path) -> Optional[FilePath]:
    """Find the .safetensors.index.json file in the given directory."""
    if not directory.is_dir():
        return None
    for file in directory.iterdir():
        if file.is_file() and file.name.endswith(".safetensors.index.json"):
            return file
    return None


@validate_arguments
def save_hf_weights(
    save_path: Path,
    model_assets_dir: DirectoryPath,
    state_dict: Dict,
    prefix: str = "",
):
    save_path.mkdir(parents=True, exist_ok=True)

    index_file: Optional[FilePath] = find_safetensors_index(Path(model_assets_dir))
    if index_file is None:
        raise FileNotFoundError(f"Could not find safetensors index file in directory {model_assets_dir}")

    # Copy index file
    shutil.copy2(index_file, save_path)

    with open(index_file, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    state_dicts = []
    for key, value in weight_map.items():
        index = int(value.split("-")[1])
        while index > len(state_dicts):
            state_dicts.append({})
        full_key = f"{prefix}{key}"
        if full_key in state_dict:
            state_dicts[index - 1][key] = state_dict[full_key]
        else:
            print(f"Missing key: '{full_key}' in state_dict")

    metadata = {"format": "pt"}
    for idx, sd in enumerate(state_dicts, start=1):
        name = f"model-{idx:05d}-of-{len(state_dicts):05d}.safetensors"
        save_file(sd, save_path / name, metadata=metadata)

    set_directory_permissions(save_path)


def update_safetensors_files(
    save_dir: Path,
    state_dict: Dict,
    weight_map: Dict[str, str],
):
    """
    Update existing safetensors files with new state_dict entries.

    This function groups new entries by their target file (from weight_map),
    loads each file, merges in the new entries, and saves back.

    Args:
        save_dir (Path): Directory containing existing HF safetensors checkpoint.
        state_dict (Dict): New state dict entries to merge into existing files.
        weight_map (Dict[str, str]): Mapping from key name to target safetensors filename.
    """
    file_to_new_weights: Dict[str, Dict] = {}
    for key, tensor in state_dict.items():
        if key in weight_map:
            target_file = weight_map[key]
            file_to_new_weights.setdefault(target_file, {})[key] = tensor

    for target_file, new_weights in file_to_new_weights.items():
        file_path = Path(save_dir) / target_file
        existing_state_dict = load_file(file_path)
        existing_state_dict.update(new_weights)
        save_file(existing_state_dict, file_path, metadata={"format": "pt"})


@validate_arguments
def merge_dcp_to_hf(
    load_dir: DirectoryPath,
    save_dir: str | Path,
    model_assets_dir: DirectoryPath,
    prefix: str = "",
):
    """
    Load model in torch DCP format and save in Hugging Face format.
    """
    state_dict = load_dcp_state_dict(load_dir)

    config = AutoConfig.from_pretrained(str(model_assets_dir))
    processor = AutoProcessor.from_pretrained(str(model_assets_dir), trust_remote_code=True)

    save_path = Path(save_dir)
    config.save_pretrained(save_path)
    processor.save_pretrained(save_path)

    save_hf_weights(
        save_path=save_path,
        model_assets_dir=str(model_assets_dir),
        state_dict=state_dict,
        prefix=prefix,
    )


def merge_dcp_to_hf_sharded(
    load_dir: DirectoryPath,
    save_dir: str | Path,
    model_assets_dir: DirectoryPath,
    select_key_convert_func: Optional[callable],
    state_dict_convert_func: Optional[callable],
    trust_remote_code: bool = True
):
    """
    Load DCP weights in shards and save them as sharded checkpoints in Hugging Face (HF) format.
    """

    config = AutoConfig.from_pretrained(model_assets_dir, trust_remote_code=trust_remote_code)
    processor = AutoProcessor.from_pretrained(model_assets_dir, trust_remote_code=trust_remote_code)
    config.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)

    index_file: Optional[FilePath] = find_safetensors_index(Path(model_assets_dir))
    if index_file is None:
        raise FileNotFoundError(f"Could not find safetensors index file in directory {model_assets_dir}")

    shutil.copy2(index_file, save_dir)
    with open(index_file, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    storage_reader = FileSystemReader(load_dir)
    metadata = load_metadata(storage_reader)
    hf_metadata = {"format": "pt"}

    safetensor_files = set(weight_map.values())
    for safetensor_file in tqdm(safetensor_files, desc="Processing files"):
        selected_keys = [
            select_key_convert_func(k) if select_key_convert_func else k
            for k, v in weight_map.items()
            if v == safetensor_file
        ]

        partial_metadata = extract_metadata(selected_keys, metadata)
        partial_state_dict = partial_load_dcp_state_dict(partial_metadata, storage_reader)
        partial_state_dict = partial_state_dict["model"] if "model" in partial_state_dict else partial_state_dict

        partial_state_dict = state_dict_convert_func(partial_state_dict) if state_dict_convert_func else partial_state_dict

        save_file(partial_state_dict, save_dir / safetensor_file, metadata=hf_metadata)

    set_directory_permissions(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-dir", type=str, required=True, help="Path to DCP checkpoint directory")
    parser.add_argument("--save-dir", type=str, required=True, help="Path to save HF format model")
    parser.add_argument("--model-assets-dir", type=str, required=True, help="Path to model assets (config, tokenizer, etc.)")
    parser.add_argument("--prefix", type=str, default="", help="Key prefix for state dict (e.g., 'model.')")
    parser.add_argument("--sharded", action="store_true", help="Enable sharded conversion to reduce memory usage (process one shard at a time)")

    args = parser.parse_args()

    print(f"Merge Args: {args}")
    if args.sharded:
        merge_dcp_to_hf_sharded(
            load_dir=args.load_dir,
            save_dir=args.save_dir,
            model_assets_dir=args.model_assets_dir,
            select_key_convert_func=lambda key: f"model.{args.prefix}" + key,
            state_dict_convert_func=lambda sd: {
                (k[len(args.prefix):] if k.startswith(args.prefix) else k): v
                for k, v in sd.items()
            }
        )
    else:
        merge_dcp_to_hf(
            load_dir=args.load_dir,
            save_dir=args.save_dir,
            model_assets_dir=args.model_assets_dir,
            prefix=args.prefix,
        )
    print(f"Merge to HF format success! Saved to: {args.save_dir}")
