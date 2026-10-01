#!/usr/bin/env python
"""
Merge LoRA safetensors weights with base HF model and save as HF format.
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import shutil
import torch
import torch_npu
from safetensors import safe_open
from safetensors.torch import save_file


def get_args():
    parser = argparse.ArgumentParser(description="Merge LoRA safetensors weights with base HF model")
    parser.add_argument("--base_hf_dir", type=str, required=True, help="Path to the base HF model directory")
    parser.add_argument("--lora_safetensors", type=str, required=True, help="Path to the LoRA safetensors file")
    parser.add_argument("--save_merged_hf_dir", type=str, required=True, help="Path to save the merged HF model")
    parser.add_argument("--lora_target_modules", type=str, nargs='+', default=None, help="LoRA target modules (auto-detect if not specified)")
    parser.add_argument("--lora_alpha", type=int, default=16, help="The lora_alpha config value")
    parser.add_argument("--lora_r", type=int, default=8, help="The lora_r config value")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "npu"], help="Device to use for LoRA merging computation (default: cpu)")
    return parser.parse_args()


def merge_lora_to_base(base_state_dict, lora_state_dict, target_layers, scaling, device="npu"):
    """Merge LoRA weights into base model weights."""
    for target_layer in target_layers:
        base_key = f"{target_layer}.weight"
        lora_a_key = f"{target_layer}.lora_A.default.weight"
        lora_b_key = f"{target_layer}.lora_B.default.weight"

        if base_key in base_state_dict:
            lora_a = lora_state_dict.get(lora_a_key)
            lora_b = lora_state_dict.get(lora_b_key)

            if lora_a is not None and lora_b is not None:
                base_weight = base_state_dict[base_key].to(device=device, dtype=torch.float32)
                lora_a = lora_a.to(device=device, dtype=torch.float32)
                lora_b = lora_b.to(device=device, dtype=torch.float32)
                merged_weight = base_weight + scaling * (lora_b @ lora_a)
                base_state_dict[base_key] = merged_weight.to(device="cpu", dtype=torch.bfloat16)
    return base_state_dict


def main():
    args = get_args()

    base_hf_dir = Path(args.base_hf_dir)
    lora_safetensors = Path(args.lora_safetensors)
    save_merged_hf_dir = Path(args.save_merged_hf_dir)
    lora_target_modules = args.lora_target_modules
    lora_alpha = args.lora_alpha
    lora_r = args.lora_r
    device = args.device
    scaling = lora_alpha / lora_r

    print("=" * 60)
    print("Loading base HF model weights...")
    print("=" * 60)

    index_file = base_hf_dir / "model.safetensors.index.json"
    with open(index_file, "r") as f:
        weight_map = json.load(f)["weight_map"]

    shard_keys = {}
    for key, shard_file in weight_map.items():
        shard_keys.setdefault(shard_file, []).append(key)

    print(f"Total weights: {len(weight_map)}")
    print(f"Shard files: {len(shard_keys)}")

    print(f"\nLoading LoRA weights from: {lora_safetensors}")
    lora_state_dict = {}
    with safe_open(lora_safetensors, framework="pt", device="cpu") as f:
        for key in f.keys():
            lora_state_dict[key] = f.get_tensor(key)
    print(f"LoRA keys: {len(lora_state_dict)}")

    target_layers = set()
    for name in lora_state_dict.keys():
        if ".lora_A.default.weight" in name:
            layer_name = name.split(".lora_")[0]
            if lora_target_modules is None or any(mod in layer_name for mod in lora_target_modules):
                target_layers.add(layer_name)
    print(f"LoRA target layers: {len(target_layers)}")

    save_merged_hf_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCopying config and tokenizer...")
    for item in base_hf_dir.iterdir():
        if item.name.startswith("model.safetensors"):
            continue
        if item.is_file():
            shutil.copy2(item, save_merged_hf_dir / item.name)
        elif item.is_dir():
            shutil.copytree(item, save_merged_hf_dir / item.name, dirs_exist_ok=True)

    shutil.copy2(index_file, save_merged_hf_dir / "model.safetensors.index.json")

    print(f"\nMerging and saving shards...")
    for shard_file, keys in shard_keys.items():
        print(f"  Processing {shard_file}...")

        shard_path = base_hf_dir / shard_file
        shard_state_dict = {}
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                shard_state_dict[key] = f.get_tensor(key)

        shard_state_dict = merge_lora_to_base(shard_state_dict, lora_state_dict, target_layers, scaling, device)
        merged_count = sum(1 for tl in target_layers if f"{tl}.weight" in shard_state_dict)

        if merged_count > 0:
            print(f"    Merged {merged_count} layers")

        save_file(shard_state_dict, save_merged_hf_dir / shard_file, metadata={"format": "pt"})
        del shard_state_dict
        if device == "npu":
            torch.npu.empty_cache()

    print(f"\nMerge complete! Saved to {save_merged_hf_dir}")


if __name__ == "__main__":
    main()
