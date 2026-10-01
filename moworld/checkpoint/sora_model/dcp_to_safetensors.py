"""Export a MoWorld Wan expert from torch DCP to DiffSynth safetensors."""

import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict


INVERSE_NAME_REPLACEMENTS = (
    ("self_attn.proj_q.", "self_attn.q."),
    ("self_attn.proj_k.", "self_attn.k."),
    ("self_attn.proj_v.", "self_attn.v."),
    ("self_attn.proj_out.", "self_attn.o."),
    ("cross_attn.proj_q.", "cross_attn.q."),
    ("cross_attn.proj_k.", "cross_attn.k."),
    ("cross_attn.proj_v.", "cross_attn.v."),
    ("cross_attn.proj_out.", "cross_attn.o."),
    ("self_attn.q_norm", "self_attn.norm_q"),
    ("self_attn.k_norm", "self_attn.norm_k"),
    ("cross_attn.q_norm", "cross_attn.norm_q"),
    ("cross_attn.k_norm", "cross_attn.norm_k"),
    ("text_embedding.linear_1.", "text_embedding.0."),
    ("text_embedding.linear_2.", "text_embedding.2."),
)

DTYPES = {
    "preserve": None,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def resolve_dcp_dir(path):
    path = Path(path).resolve()
    candidates = [path, path / "release"]
    tracker = path / "latest_checkpointed_iteration.txt"
    if tracker.is_file():
        latest = tracker.read_text(encoding="utf-8").strip()
        if latest == "release":
            candidates.append(path / "release")
        elif latest.isdigit():
            candidates.append(path / f"iter_{int(latest):07d}")
    for candidate in candidates:
        if (candidate / ".metadata").is_file():
            return candidate
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"No DCP .metadata found; checked: {checked}")


def convert_name(name):
    name = name.removeprefix("model.")
    for source, target in INVERSE_NAME_REPLACEMENTS:
        name = name.replace(source, target)
    return name


def load_dcp(path):
    state_dict = {}
    _load_state_dict(
        state_dict,
        storage_reader=FileSystemReader(str(path)),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    return state_dict.get("model", state_dict)


def validate_reference(state_dict, reference_path):
    with safe_open(str(reference_path), framework="pt", device="cpu") as reference:
        reference_keys = set(reference.keys())
        output_keys = set(state_dict)
        if reference_keys != output_keys:
            missing = sorted(reference_keys - output_keys)
            extra = sorted(output_keys - reference_keys)
            raise ValueError(f"Reference key mismatch: missing={missing[:10]}, extra={extra[:10]}")
        for name, tensor in state_dict.items():
            expected_shape = tuple(reference.get_slice(name).get_shape())
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"Shape mismatch for {name}: DCP={tuple(tensor.shape)}, reference={expected_shape}"
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="DCP expert directory, optionally containing release/")
    parser.add_argument("--output", required=True, help="Output diffusion_pytorch_model.safetensors")
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--reference-safetensors", help="Optional base expert used to validate names and shapes")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file")
    args = parser.parse_args()

    dcp_dir = resolve_dcp_dir(args.input)
    output_path = Path(args.output).resolve()
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[dcp-export] loading {dcp_dir}", flush=True)
    source_state = load_dcp(dcp_dir)
    print(f"[dcp-export] loaded {len(source_state)} entries", flush=True)

    target_dtype = DTYPES[args.dtype]
    converted = {}
    skipped = []
    for source_name in list(source_state):
        tensor = source_state.pop(source_name)
        if not isinstance(tensor, torch.Tensor):
            skipped.append(source_name)
            continue
        target_name = convert_name(source_name)
        if target_name in converted:
            raise KeyError(f"Duplicate converted tensor name: {target_name}")
        tensor = tensor.detach().cpu()
        if target_dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(target_dtype)
        converted[target_name] = tensor.contiguous()

    if args.reference_safetensors:
        print(f"[dcp-export] validating against {args.reference_safetensors}", flush=True)
        validate_reference(converted, Path(args.reference_safetensors))

    print(
        f"[dcp-export] saving {len(converted)} tensors as {args.dtype}; skipped={skipped}",
        flush=True,
    )
    save_file(converted, str(output_path), metadata={"format": "pt"})
    size_gib = output_path.stat().st_size / (1024 ** 3)
    print(f"[dcp-export] complete: {output_path} ({size_gib:.2f} GiB)", flush=True)


if __name__ == "__main__":
    main()
