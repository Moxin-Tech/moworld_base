"""
Convert between MindSpeed-MM DCP checkpoint and DiffSynth safetensors.

Supports two directions:
  1. DCP → DiffSynth  (default): convert trained DCP checkpoint for DiffSynth inference
  2. DiffSynth → DCP  (--reverse): convert original base model to DCP for MM training init

Usage:
    # DCP → DiffSynth (after training, for inference)
    python checkpoint/convert_dcp_to_diffsynth.py \
        --load_dir /path/to/dcp/release/ \
        --save_path /path/to/output.safetensors \
        --reference /path/to/base_model/diffusion_pytorch_model.safetensors

    # DiffSynth → DCP (before training, to create initial checkpoint)
    python checkpoint/convert_dcp_to_diffsynth.py \
        --reverse \
        --hf_path /path/to/base_model/diffusion_pytorch_model.safetensors \
        --dcp_dir /path/to/output_dcp/

Key naming differences (MM DCP ↔ DiffSynth native):
  MM:   predictor.blocks.X.self_attn.proj_q.weight
  DS:   blocks.X.self_attn.q.weight

  MM:   predictor.blocks.X.self_attn.q_norm.weight
  DS:   blocks.X.self_attn.norm_q.weight

  MM:   predictor.blocks.X.cross_attn.proj_out.weight
  DS:   blocks.X.cross_attn.o.weight

  MM:   predictor.text_embedding.linear_1.weight
  DS:   text_embedding.0.weight
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors.torch import save_file, load_file
from checkpoint.common.merge_dcp_to_hf import load_dcp_state_dict


# ═══════════════════════════════════════════════════════════════
# DCP → DiffSynth key renaming (MM format → DiffSynth native)
# ═══════════════════════════════════════════════════════════════

DCP_TO_DIFFSYNTH_RENAMES = [
    # self_attn
    (r"\.self_attn\.proj_q\.",   ".self_attn.q."),
    (r"\.self_attn\.proj_k\.",   ".self_attn.k."),
    (r"\.self_attn\.proj_v\.",   ".self_attn.v."),
    (r"\.self_attn\.proj_out\.", ".self_attn.o."),
    (r"\.self_attn\.q_norm\.",   ".self_attn.norm_q."),
    (r"\.self_attn\.k_norm\.",   ".self_attn.norm_k."),
    # cross_attn
    (r"\.cross_attn\.proj_q\.",   ".cross_attn.q."),
    (r"\.cross_attn\.proj_k\.",   ".cross_attn.k."),
    (r"\.cross_attn\.proj_v\.",   ".cross_attn.v."),
    (r"\.cross_attn\.proj_out\.", ".cross_attn.o."),
    (r"\.cross_attn\.q_norm\.",   ".cross_attn.norm_q."),
    (r"\.cross_attn\.k_norm\.",   ".cross_attn.norm_k."),
    # cross_attn image
    (r"\.cross_attn\.proj_k_img\.", ".cross_attn.k_img."),
    (r"\.cross_attn\.proj_v_img\.", ".cross_attn.v_img."),
    (r"\.cross_attn\.norm_k_img\.", ".cross_attn.norm_k_img."),
    # Megatron alternate naming
    (r"\.self_attn\.linear_q\.",    ".self_attn.q."),
    (r"\.self_attn\.linear_k\.",    ".self_attn.k."),
    (r"\.self_attn\.linear_v\.",    ".self_attn.v."),
    (r"\.self_attn\.linear_proj\.", ".self_attn.o."),
    (r"\.cross_attn\.linear_q\.",   ".cross_attn.q."),
    (r"\.cross_attn\.linear_k\.",   ".cross_attn.k."),
    (r"\.cross_attn\.linear_v\.",   ".cross_attn.v."),
    (r"\.cross_attn\.linear_proj\.", ".cross_attn.o."),
    # text_embedding: TextProjection (linear_1/linear_2) → nn.Sequential (0/2)
    (r"^text_embedding\.linear_1\.", "text_embedding.0."),
    (r"^text_embedding\.linear_2\.", "text_embedding.2."),
]

DCP_STRIP_PREFIXES = ["predictor.", "model.", "module."]


# ═══════════════════════════════════════════════════════════════
# DiffSynth → DCP key renaming (DiffSynth native → MM format)
# ═══════════════════════════════════════════════════════════════

DIFFSYNTH_TO_DCP_RENAMES = [
    # self_attn
    (r"\.self_attn\.q\.",        ".self_attn.proj_q."),
    (r"\.self_attn\.k\.",        ".self_attn.proj_k."),
    (r"\.self_attn\.v\.",        ".self_attn.proj_v."),
    (r"\.self_attn\.o\.",        ".self_attn.proj_out."),
    (r"\.self_attn\.norm_q\.",   ".self_attn.q_norm."),
    (r"\.self_attn\.norm_k\.",   ".self_attn.k_norm."),
    # cross_attn
    (r"\.cross_attn\.q\.",       ".cross_attn.proj_q."),
    (r"\.cross_attn\.k\.",       ".cross_attn.proj_k."),
    (r"\.cross_attn\.v\.",       ".cross_attn.proj_v."),
    (r"\.cross_attn\.o\.",       ".cross_attn.proj_out."),
    (r"\.cross_attn\.norm_q\.",  ".cross_attn.q_norm."),
    (r"\.cross_attn\.norm_k\.",  ".cross_attn.k_norm."),
    # cross_attn image
    (r"\.cross_attn\.k_img\.",   ".cross_attn.proj_k_img."),
    (r"\.cross_attn\.v_img\.",   ".cross_attn.proj_v_img."),
    (r"\.cross_attn\.norm_k_img\.", ".cross_attn.norm_k_img."),
    # text_embedding: nn.Sequential (0/2) → TextProjection (linear_1/linear_2)
    (r"^text_embedding\.0\.", "text_embedding.linear_1."),
    (r"^text_embedding\.2\.", "text_embedding.linear_2."),
]

DCP_ADD_PREFIX = "predictor."


# ═══════════════════════════════════════════════════════════════
# Shared validation patterns
# ═══════════════════════════════════════════════════════════════

EXPECTED_DIFFSYNTH_PATTERNS = [
    r"^blocks\.\d+\.self_attn\.[a-z_]+\.(weight|bias)$",
    r"^blocks\.\d+\.cross_attn\.[a-z_]+\.(weight|bias)$",
    r"^blocks\.\d+\.norm\d+\.(weight|bias)$",
    r"^blocks\.\d+\.modulation$",
    r"^blocks\.\d+\.ffn\.\d+\.(weight|bias)$",
    r"^head\.head\.(weight|bias)$",
    r"^head\.modulation$",
    r"^patch_embedding\.(weight|bias)$",
    r"^text_embedding\.\d+\.(weight|bias)$",
    r"^time_embedding\.\d+\.(weight|bias)$",
    r"^time_projection\.\d+\.(weight|bias)$",
    r"^control_adapter\..+$",
]


def dcp_key_to_diffsynth(key: str) -> str:
    """Convert a DCP state_dict key to DiffSynth WanModel format."""
    if "_extra_state" in key:
        return None
    new_key = key
    for prefix in DCP_STRIP_PREFIXES:
        if new_key.startswith(prefix):
            new_key = new_key[len(prefix):]
            break
    for pattern, replacement in DCP_TO_DIFFSYNTH_RENAMES:
        new_key = re.sub(pattern, replacement, new_key)
    return new_key


def diffsynth_key_to_dcp(key: str) -> str:
    """Convert a DiffSynth native key to MM DCP format."""
    new_key = key
    for pattern, replacement in DIFFSYNTH_TO_DCP_RENAMES:
        new_key = re.sub(pattern, replacement, new_key)
    new_key = DCP_ADD_PREFIX + new_key
    return new_key


def validate_diffsynth_keys(keys):
    valid, unexpected = [], []
    for k in keys:
        matched = any(re.match(pat, k) for pat in EXPECTED_DIFFSYNTH_PATTERNS)
        (valid if matched else unexpected).append(k)
    return valid, unexpected


# ═══════════════════════════════════════════════════════════════
# Verification against reference
# ═══════════════════════════════════════════════════════════════

def verify_against_reference(converted, ref_path):
    print(f"\n{'='*65}")
    print(f" Verification against reference: {ref_path}")
    print(f"{'='*65}")

    ref = load_file(ref_path)

    keys_conv = set(converted.keys())
    keys_ref = set(ref.keys())
    common = sorted(keys_conv & keys_ref)
    only_conv = sorted(keys_conv - keys_ref)
    only_ref = sorted(keys_ref - keys_conv)

    print(f"\n  Key coverage:")
    print(f"    Converted:  {len(keys_conv)}")
    print(f"    Reference:  {len(keys_ref)}")
    print(f"    Matched:    {len(common)}")
    print(f"    Only in converted: {len(only_conv)}")
    print(f"    Only in reference: {len(only_ref)}")

    if only_conv:
        print(f"\n  Keys only in converted:")
        for k in only_conv[:15]:
            print(f"    {k}: {tuple(converted[k].shape)}")
        if len(only_conv) > 15:
            print(f"    ... and {len(only_conv) - 15} more")

    if only_ref:
        print(f"\n  Keys MISSING from converted (not in reference):")
        for k in only_ref[:15]:
            print(f"    {k}: {tuple(ref[k].shape)}")
        if len(only_ref) > 15:
            print(f"    ... and {len(only_ref) - 15} more")

    # Shape check
    shape_ok = 0
    shape_mismatch = []
    for k in common:
        if converted[k].shape == ref[k].shape:
            shape_ok += 1
        else:
            shape_mismatch.append((k, converted[k].shape, ref[k].shape))

    print(f"\n  Shape check: {shape_ok}/{len(common)} matching")
    if shape_mismatch:
        print(f"    Mismatched: {len(shape_mismatch)}")
        for k, s1, s2 in shape_mismatch[:10]:
            print(f"      {k}: {s1} vs {s2}")

    # Value comparison
    match_exact = 0
    match_transposed = 0
    value_diff = []

    for k in common:
        t1 = converted[k].float()
        t2 = ref[k].float()
        if t1.shape != t2.shape:
            continue
        if t1.isnan().any() or t2.isnan().any():
            value_diff.append((k, float('nan'), float('nan'), 0.0, False))
            continue
        if torch.allclose(t1, t2, atol=1e-5):
            match_exact += 1
            continue
        # Check transposed
        if t1.ndim == 2 and t1.shape == t2.T.shape and torch.allclose(t1, t2.T, atol=1e-5):
            match_transposed += 1
            continue
        max_diff = (t1 - t2).abs().max().item()
        mean_diff = (t1 - t2).abs().mean().item()
        value_diff.append((k, max_diff, mean_diff, t2.std().item(), False))

    print(f"\n  Value comparison:")
    print(f"    Exact match:      {match_exact}")
    print(f"    Transposed match: {match_transposed}")
    print(f"    Value differ:     {len(value_diff)}")

    if value_diff:
        value_diff.sort(key=lambda x: x[1] if x[1] == x[1] else float('inf'), reverse=True)
        print(f"\n  Top-20 value differences:")
        print(f"    {'key':<55} {'max_diff':>10} {'mean_diff':>10} {'ref_std':>8}")
        for k, mx, mn, std, _ in value_diff[:20]:
            mx_s = f"{mx:.2e}" if mx == mx else "NaN"
            mn_s = f"{mn:.2e}" if mn == mn else "NaN"
            print(f"    {k:<55} {mx_s:>10} {mn_s:>10} {std:>8.4f}")

        # Per-group
        groups = {}
        for k, mx, mn, std, _ in value_diff:
            parts = k.split(".")
            group = parts[0] if parts[0] != "blocks" else f"{parts[0]}.{parts[1]}.{parts[2]}" if len(parts) >= 3 else parts[0]
            groups.setdefault(group, []).append((k, mx, mn))
        print(f"\n  Per-group diff summary:")
        print(f"    {'group':<35} {'count':>6} {'max_max_diff':>14} {'avg_mean_diff':>14}")
        for g in sorted(groups.keys()):
            items = groups[g]
            max_of_max = max(x[1] for x in items if x[1] == x[1])
            avg_of_mean = sum(x[2] for x in items if x[2] == x[2]) / len(items)
            print(f"    {g:<35} {len(items):>6} {max_of_max:>14.2e} {avg_of_mean:>14.2e}")

    # Verdict
    print(f"\n  Verdict:")
    if not only_conv and not only_ref and not shape_mismatch and match_exact == len(common):
        print(f"    PERFECT MATCH")
    elif only_ref:
        print(f"    PROBLEM — {len(only_ref)} keys missing from converted")
    elif shape_mismatch:
        print(f"    PROBLEM — {len(shape_mismatch)} shape mismatches")
    elif match_transposed > 0 and match_exact + match_transposed == len(common):
        print(f"    ALL KEYS MATCH — {match_transposed} need transpose fix")
    else:
        print(f"    {len(value_diff)} keys differ in value")
        print(f"    If comparing TRAINED weights vs base model, diff is expected.")
        print(f"    If comparing UNTRAINED base→DCP→base roundtrip, diff means conversion error.")
    print(f"{'='*65}\n")


# ═══════════════════════════════════════════════════════════════
# Direction 1: DCP → DiffSynth
# ═══════════════════════════════════════════════════════════════

def convert_dcp_to_diffsynth(args):
    total_steps = 5 if args.reference else 4

    # 1. Load DCP
    print(f"[1/{total_steps}] Loading DCP from: {args.load_dir}")
    state_dict = load_dcp_state_dict(args.load_dir)
    print(f"       Loaded {len(state_dict)} keys")

    # 2. Convert keys
    print(f"[2/{total_steps}] Converting key names (DCP → DiffSynth)...")
    converted = {}
    skipped = []
    for key, tensor in state_dict.items():
        new_key = dcp_key_to_diffsynth(key)
        if new_key is None or tensor is None:
            skipped.append(key)
            continue
        converted[new_key] = tensor.contiguous()

    if skipped:
        print(f"       Skipped {len(skipped)} keys (None / _extra_state)")
    print(f"       Converted: {len(converted)} keys")

    # 3. Validate
    print(f"[3/{total_steps}] Validating keys...")
    valid, unexpected = validate_diffsynth_keys(sorted(converted.keys()))
    print(f"       Matching DiffSynth patterns: {len(valid)}")
    if unexpected:
        print(f"       Unexpected: {len(unexpected)}")
        for k in unexpected[:10]:
            print(f"         {k}: {tuple(converted[k].shape)}")

    # 4. Verify against reference
    if args.reference:
        print(f"[4/{total_steps}] Verifying against reference...")
        verify_against_reference(converted, args.reference)
    else:
        print(f"[4/{total_steps}] No --reference, skipping verification")

    # 5. Save
    print(f"[{total_steps}/{total_steps}] Saving to: {args.save_path}")
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    save_file(converted, args.save_path)
    print(f"       Done! {len(converted)} keys saved")


# ═══════════════════════════════════════════════════════════════
# Direction 2: DiffSynth → DCP (for training initialization)
# ═══════════════════════════════════════════════════════════════

def convert_diffsynth_to_dcp(args):
    """
    Convert a DiffSynth native safetensors to DCP format for MM training init.

    This creates a DCP checkpoint with correct MM key names so that
    the training framework can properly load the base model weights.
    """
    from torch.distributed.checkpoint import FileSystemWriter
    from checkpoint.common.dcp_utils import partial_save_dcp_state_dict, merge_meta_info, save_metadata
    from checkpoint.common.permissions import set_directory_permissions
    from checkpoint.common.constant import LATEST_TXT

    # 1. Load DiffSynth safetensors
    print(f"[1/3] Loading DiffSynth weights from: {args.hf_path}")
    sd = load_file(args.hf_path)
    print(f"       Loaded {len(sd)} keys")

    # 2. Convert keys (DiffSynth native → MM DCP)
    print(f"[2/3] Converting key names (DiffSynth → DCP)...")
    converted = {}
    for key, tensor in sd.items():
        new_key = diffsynth_key_to_dcp(key)
        converted[new_key] = tensor.contiguous()

    # Print mapping sample
    sample = sorted(converted.keys())[:8]
    print(f"       Sample mappings:")
    for k in sample:
        orig = k[len(DCP_ADD_PREFIX):]
        print(f"         {orig} → {k}")

    # Verify key names match MM model expectations
    # MM expects keys like: predictor.blocks.X.self_attn.proj_q.weight
    mm_patterns = [
        r"^predictor\.blocks\.\d+\.self_attn\.proj_[qkv]\.(weight|bias)$",
        r"^predictor\.blocks\.\d+\.self_attn\.proj_out\.(weight|bias)$",
        r"^predictor\.blocks\.\d+\.self_attn\.[qk]_norm\.(weight|bias)$",
        r"^predictor\.blocks\.\d+\.cross_attn\.proj_[qkv]\.(weight|bias)$",
        r"^predictor\.blocks\.\d+\.cross_attn\.proj_out\.(weight|bias)$",
        r"^predictor\.blocks\.\d+\.cross_attn\.[qk]_norm\.(weight|bias)$",
        r"^predictor\.blocks\.\d+\.norm\d+\.(weight|bias)$",
        r"^predictor\.blocks\.\d+\.modulation$",
        r"^predictor\.blocks\.\d+\.ffn\.\d+\.(weight|bias)$",
        r"^predictor\.head\.head\.(weight|bias)$",
        r"^predictor\.head\.modulation$",
        r"^predictor\.patch_embedding\.(weight|bias)$",
        r"^predictor\.text_embedding\.linear_[12]\.(weight|bias)$",
        r"^predictor\.time_embedding\.\d+\.(weight|bias)$",
        r"^predictor\.time_projection\.\d+\.(weight|bias)$",
        r"^predictor\.control_adapter\..+$",
    ]

    mm_valid, mm_unexpected = [], []
    for k in converted.keys():
        matched = any(re.match(pat, k) for pat in mm_patterns)
        (mm_valid if matched else mm_unexpected).append(k)

    print(f"\n       MM format validation: {len(mm_valid)} matching")
    if mm_unexpected:
        print(f"       Unexpected: {len(mm_unexpected)}")
        for k in mm_unexpected[:10]:
            print(f"         {k}: {tuple(converted[k].shape)}")
    else:
        print(f"       All keys match MM model expectations!")

    # 3. Save as DCP
    print(f"\n[3/3] Saving DCP to: {args.dcp_dir}")
    iter_name = "release"
    save_root = Path(args.dcp_dir)
    save_path = save_root / iter_name
    save_path.mkdir(exist_ok=True, parents=True)
    save_root.joinpath(LATEST_TXT).write_text("release")

    storage_writer = FileSystemWriter(save_path)
    save_dict = {"model": converted, "checkpoint_version": 3.0}

    global_meta, all_writes = partial_save_dcp_state_dict(save_dict, storage_writer, part_idx=0)
    merged_meta = merge_meta_info([global_meta])
    save_metadata(merged_meta, [all_writes], storage_writer)
    set_directory_permissions(save_root)

    print(f"       Done! DCP checkpoint saved to {save_path}")

    # Verification: roundtrip check — load back and compare
    print(f"\n       Roundtrip verification...")
    reloaded = load_dcp_state_dict(save_path)
    reloaded_converted = {}
    for key, tensor in reloaded.items():
        new_key = dcp_key_to_diffsynth(key)
        if new_key is None or tensor is None:
            continue
        reloaded_converted[new_key] = tensor.float()

    exact = 0
    diff_keys = []
    for k in sorted(reloaded_converted.keys()):
        if k in sd:
            if torch.allclose(reloaded_converted[k], sd[k].float(), atol=1e-5):
                exact += 1
            else:
                diff_keys.append(k)

    total = len(reloaded_converted)
    print(f"       Roundtrip: {exact}/{total} keys exact match after save→load→convert")
    if diff_keys:
        print(f"       Differing keys (first 10):")
        for k in diff_keys[:10]:
            print(f"         {k}")
    else:
        print(f"       PERFECT ROUNDTRIP — conversion is lossless!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert between MindSpeed-MM DCP and DiffSynth safetensors"
    )
    parser.add_argument("--reverse", action="store_true",
                        help="Reverse direction: DiffSynth → DCP (for training init)")

    # DCP → DiffSynth args
    parser.add_argument("--load_dir", type=str, default=None,
                        help="[DCP→DS] Path to DCP checkpoint directory")
    parser.add_argument("--save_path", type=str, default=None,
                        help="[DCP→DS] Output safetensors file path")
    parser.add_argument("--reference", type=str, default=None,
                        help="[DCP→DS] Reference safetensors for verification")

    # DiffSynth → DCP args
    parser.add_argument("--hf_path", type=str, default=None,
                        help="[DS→DCP] Input DiffSynth safetensors file path")
    parser.add_argument("--dcp_dir", type=str, default=None,
                        help="[DS→DCP] Output DCP directory path")

    args = parser.parse_args()

    if args.reverse:
        if not args.hf_path or not args.dcp_dir:
            parser.error("--reverse requires --hf_path and --dcp_dir")
        convert_diffsynth_to_dcp(args)
    else:
        if not args.load_dir or not args.save_path:
            parser.error("DCP→DiffSynth requires --load_dir and --save_path")
        convert_dcp_to_diffsynth(args)
