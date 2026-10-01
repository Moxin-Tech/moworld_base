import argparse
from typing import Optional
from pathlib import Path
from tqdm import tqdm
from torch.distributed.checkpoint import FileSystemWriter
from safetensors.torch import load_file

from checkpoint.common.constant import LATEST_TXT
from checkpoint.common.dcp_utils import partial_save_dcp_state_dict, merge_meta_info, save_metadata
from checkpoint.vlm_model.hf_to_mm import load_from_hf, save_by_dcp
from checkpoint.common.permissions import set_directory_permissions


def hf_to_dcp(
    hf_dir: str,
    dcp_dir: str,
    prefix: Optional[str]
):
    state_dict = load_from_hf(Path(hf_dir))
    state_dict = {f"{prefix}{k}": v for k, v in state_dict.items()}
    save_by_dcp(state_dict, Path(dcp_dir))


def hf_to_dcp_sharded(
    hf_dir: str,
    dcp_dir: str,
    state_dict_convert_func: Optional[callable],
):
    """
    By default, DCP shards are split following the same sharding logic as the original Hugging Face (HF) checkpoint weights.
    """
    iter_name = "release"
    save_root_dir = Path(dcp_dir)
    save_path = save_root_dir.joinpath(iter_name)
    save_path.mkdir(exist_ok=True, parents=True)
    save_root_dir.joinpath(LATEST_TXT).write_text("release")

    storage_writer = FileSystemWriter(save_path)
    files = sorted(list(Path(hf_dir).glob("*.safetensors")))

    meta_infos = []
    all_writes = []
    for i, safe_path in enumerate(tqdm(files, desc="Processing files")):
        state_dict = load_file(str(safe_path), device="cpu")
        state_dict = state_dict_convert_func(state_dict) if state_dict_convert_func else state_dict

        save_dict = {
            "model": state_dict
        }

        if i == 0:
            save_dict["checkpoint_version"] = 3.0

        global_meta, all_write = partial_save_dcp_state_dict(save_dict, storage_writer, part_idx=i)
        meta_infos.append(global_meta)
        all_writes.append(all_write)

    merged_meta = merge_meta_info(meta_infos)
    save_metadata(merged_meta, all_writes, storage_writer)
    set_directory_permissions(Path(dcp_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", type=str, required=True, help="Path to HF format checkpoint directory")
    parser.add_argument("--dcp-dir", type=str, required=True, help="Path to save torch_dcp format model")
    parser.add_argument("--prefix", type=str, default="", help="Key prefix for state dict (e.g., 'model.')")
    parser.add_argument("--sharded", action="store_true", help="Enable sharded conversion to reduce memory usage (process one shard at a time)")

    args = parser.parse_args()
    if args.sharded:
        hf_to_dcp_sharded(
            args.hf_dir,
            args.dcp_dir,
            state_dict_convert_func=lambda sd: {f"{args.prefix}{k}": v for k, v in sd.items()}
        )
    else:
        hf_to_dcp(
            args.hf_dir,
            args.dcp_dir,
            prefix=args.prefix
        )
