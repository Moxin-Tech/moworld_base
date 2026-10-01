import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Set, Tuple

import torch
import torch.distributed as dist
from safetensors import safe_open
from tqdm import tqdm
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

from mindspeed.fsdp.utils.log import print_rank
from mindspeed_mm.fsdp.checkpoint.load_utils import ParamInfo
from mindspeed_mm.fsdp.checkpoint.utils import remove_base_layer_keys
from mindspeed_mm.fsdp.utils.device import empty_cache, get_device_type
from mindspeed_mm.fsdp.utils.utils import tensor_to_dtensor

logger = logging.getLogger(__name__)


# ===========================================================================
# stream one safetensors file tensor by tensor
# ===========================================================================
@dataclass
class HFWeightFileStream:
    """Yields ``(key, full_tensor)`` one tensor at a time from a safetensors file.

    Uses ``safe_open`` so only the current tensor is materialized in host
    memory; the consumer copies it into the model and drops it before the next
    is read. Peak host memory is therefore about one tensor, not the whole
    shard file.
    """

    filepath: str

    def __iter__(self) -> Generator[Tuple[str, torch.Tensor], None, None]:
        with safe_open(self.filepath, framework="pt", device="cpu") as f:
            for key in f.keys():
                yield key, f.get_tensor(key)


# ===========================================================================
# locate weight files / detect an HF directory
# ===========================================================================
def locate_hf_weight_files(weights_path: str) -> List[HFWeightFileStream]:
    """Resolve the safetensors files under *weights_path* into stream readers.

    Supports both standard HF layouts:
      - single ``model.safetensors``
      - sharded ``model-*-of-*.safetensors`` described by ``model.safetensors.index.json``
    """
    single = os.path.join(weights_path, SAFE_WEIGHTS_NAME)
    if os.path.isfile(single):
        return [HFWeightFileStream(single)]

    index = os.path.join(weights_path, SAFE_WEIGHTS_INDEX_NAME)
    if os.path.isfile(index):
        with open(index, "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        streams = []
        for name in sorted(set(weight_map.values())):
            path = os.path.join(weights_path, name)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Shard '{name}' referenced by {index} is missing: {path}"
                )
            streams.append(HFWeightFileStream(path))
        return streams

    raise ValueError(f"No HF safetensors weights found under {weights_path}.")


def looks_like_hf_weight_dir(path: Optional[str]) -> bool:
    """Whether *path* is a directory holding HF safetensors weights.

    Used by ``load_format='auto'`` to tell an HF directory apart from a DCP
    checkpoint directory -- the latter has ``latest_checkpointed_iteration.txt``
    and ``.distcp`` files instead, never the standard HF weight filenames.
    """
    if not path or not os.path.isdir(path):
        return False
    return os.path.isfile(os.path.join(path, SAFE_WEIGHTS_NAME)) or os.path.isfile(
        os.path.join(path, SAFE_WEIGHTS_INDEX_NAME)
    )


# ===========================================================================
# write one tensor into the (sharded) model
# ===========================================================================
def _resolve_leaf(model: torch.nn.Module, name: str) -> Tuple[torch.nn.Module, str]:
    """Walk the dotted FQN *name* down to the leaf module that owns it.

    For ``model.language_model.layers.0.self_attn.q_proj.weight`` it returns
    the ``q_proj`` Linear module and ``"weight"``.
    """
    module = model
    pieces = name.split(".")
    for piece in pieces[:-1]:
        if not hasattr(module, piece):
            raise ValueError(f"Cannot resolve '{name}': submodule '{piece}' not found.")
        module = getattr(module, piece)
    return module, pieces[-1]


def convert_weight_key(key: str, model: torch.nn.Module) -> str:
    mapping = getattr(model, "_checkpoint_conversion_mapping", None)
    if not mapping:
        return key
    for pattern, replacement in mapping.items():
        replacement = re.sub(r"\(.*\)", "", replacement.lstrip("^"))
        new_key, n_subs = re.subn(pattern, replacement, key)
        if n_subs > 0:
            return new_key
    return key


def write_full_tensor(model: torch.nn.Module, name: str, full_tensor: torch.Tensor) -> None:
    """Write *full_tensor* into the model at *name* (parameter or buffer).

    For a sharded parameter (a DTensor created by ``fully_shard``), the target
    already carries its ``device_mesh`` / ``placements``; we move the full
    tensor onto the mesh device and let ``tensor_to_dtensor`` carve out this
    rank's shard (a local Replicate -> Shard redistribute, no comm). For a
    plain tensor (replicated parameter or buffer), we just move dtype/device
    and copy. The write is always in place since FSDP2 holds the parameter
    object.
    """
    leaf_module, local_name = _resolve_leaf(model, name)

    if local_name in leaf_module._parameters:
        target = leaf_module._parameters[local_name].data
    elif local_name in leaf_module._buffers:
        target = leaf_module._buffers[local_name]
    else:
        raise ValueError(f"'{name}' is neither a parameter nor a buffer of the model.")

    if hasattr(target, "device_mesh"):  # sharded -> a DTensor
        full_tensor = full_tensor.to(device=target.to_local().device, dtype=target.dtype)
        shard = tensor_to_dtensor(full_tensor, target.device_mesh, target.placements)
        target.copy_(shard)
    else:  # plain tensor (replicated on every rank)
        target.copy_(full_tensor.to(device=target.device, dtype=target.dtype))


def _lora_base_key_map(param_names: Set[str]) -> Dict[str, str]:
    """Ref DCP lora base key map
    """
    base_to_bare = remove_base_layer_keys({name: None for name in param_names})
    return {bare: base for base, bare in base_to_bare.items()}


def _retie_embeddings(model: torch.nn.Module) -> None:
    """Re-tie input/output embeddings when the config requests it.

    - ``to_empty_if_needed`` broke any shared storage; this restores it.
    - AND across ``model.config`` and ``text_config`` covers nested multimodal cases.
    - Object-reference assignment so both modules share one nn.Parameter.
    """
    config = getattr(model, "config", None)
    if config is None:
        return
    text_config = (
        config.get_text_config(decoder=True) if hasattr(config, "get_text_config") else config
    )
    should_tie = (
        (hasattr(config, "tie_word_embeddings") or hasattr(text_config, "tie_word_embeddings"))
        and getattr(config, "tie_word_embeddings", True)
        and getattr(text_config, "tie_word_embeddings", True)
    )
    if not should_tie:
        return

    try:
        input_embeddings = model.get_input_embeddings()
        output_embeddings = model.get_output_embeddings()
        if input_embeddings is None or output_embeddings is None:
            return
        # Object-reference assignment -- after this both modules share the same
        # nn.Parameter, so gradients accumulate into one storage.
        output_embeddings._parameters["weight"] = input_embeddings._parameters["weight"]
    except Exception as e:
        raise RuntimeError("Failed to tie input/output embeddings after HF load") from e


def _log_unexpected_keys(unexpected_keys: Set[str]) -> None:
    if not unexpected_keys:
        return
    samples = sorted(unexpected_keys)[:5]
    suffix = "" if len(unexpected_keys) <= 5 else f" (showing 5 of {len(unexpected_keys)})"
    print_rank(
        logger.info,
        f"HF checkpoint had {len(unexpected_keys)} key(s) not present in the model. Examples{suffix}: {samples}",
    )


# ===========================================================================
# Post-load fixups -- tied embeddings and missing-key reporting
# ===========================================================================
def post_process_after_load(
    model: torch.nn.Module,
    missing_param_keys: set,
    load_strict: bool = False,
) -> None:
    # filter LoRA: lora_A/lora_B are intentionally absent from a base checkpoint
    missing = {k for k in missing_param_keys if "lora_" not in k}
    if missing:
        # load_strict : true -> error on any model key the tensor did not provide with raise.
        if load_strict:
            raise RuntimeError(
                f"{len(missing)} parameter key(s) absent from the HF checkpoint "
                f"(load_strict=True): {sorted(missing)}"
            )
        logger.warning(
            "%d parameter key(s) absent from the HF checkpoint and left "
            "uninitialized: %s. Training will likely produce NaNs. "
            "Pre-load these via the HF file or extend post_process_after_load "
            "to call _init_weights on them.",
            len(missing),
            sorted(missing),
        )

    _retie_embeddings(model)


@torch.no_grad()
def load_hf_weights(
    model: torch.nn.Module, hf_dir: str, enable_lora: bool = False, load_strict: bool = False
) -> None:
    """Load HF safetensors weights into *model* directly, no offline conversion.

    Every rank opens the same files and slices each full tensor into its own
    DTensor shard locally -- no cross-rank communication. Suitable for small/
    medium models or shared filesystems; for very large models use the
    rank0-read-and-broadcast variant (added in a follow-up).

    Assumes ``model`` has already been laid out by ``fully_shard`` and brought
    out of meta via ``to_empty_if_needed`` -- i.e. parameter slots are empty
    DTensors on the real device and buffer values have been preserved.
    """
    param_names = {name for name, _ in model.named_parameters()}
    buffer_names = {name for name, _ in model.named_buffers()}
    lora_base_map = _lora_base_key_map(param_names) if enable_lora else {}
    unexpected_keys: Set[str] = set()

    for shard_stream in locate_hf_weight_files(hf_dir):
        for raw_key, full_tensor in shard_stream:
            key = convert_weight_key(raw_key, model)
            key = lora_base_map.get(key, key)  # bare -> base_layer for LoRA targets
            if key in param_names:
                param_names.discard(key)
                write_full_tensor(model, key, full_tensor)
            elif key in buffer_names:
                # Persistent buffer override from the file. Non-persistent
                # buffers (e.g. RoPE inv_freq) never appear here; they were
                # built in model construction and preserved by to_empty_if_needed.
                write_full_tensor(model, key, full_tensor)
            else:
                unexpected_keys.add(key)

    _log_unexpected_keys(unexpected_keys)
    post_process_after_load(model, missing_param_keys=param_names, load_strict=load_strict)


def load_hf_checkpoint(
    model: torch.nn.Module,
    hf_dir: str,
    *,
    load_rank0_and_broadcast: bool = False,
    enable_lora: bool = False,
    load_strict: bool = False,
) -> bool:
    cfg = {}
    cfg_path = os.path.join(hf_dir, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    tcfg = cfg.get("text_config", cfg)
    method = "rank0-broadcast" if load_rank0_and_broadcast else "every-rank-read"
    print_rank(
        logger.info,
        f"Loading HF safetensors -> FSDP DTensors via {method} (online): "
        f"dir={hf_dir} "
        f"arch={(cfg.get('architectures') or ['?'])[0]} "
        f"layers={tcfg.get('num_hidden_layers', '?')} "
        f"tie_emb={cfg.get('tie_word_embeddings', tcfg.get('tie_word_embeddings', '?'))}",
    )

    if load_rank0_and_broadcast:
        rank0_load_and_broadcast_hf_weights(model, hf_dir, enable_lora=enable_lora, load_strict=load_strict)
    else:
        load_hf_weights(model, hf_dir, enable_lora=enable_lora, load_strict=load_strict)
    return True


@torch.no_grad()
def rank0_load_and_broadcast_hf_weights(
    model: torch.nn.Module, hf_dir: str, enable_lora: bool = False, load_strict: bool = False
) -> None:
    """Load HF safetensors weights via rank0 read + ``dist.broadcast``.

    Mirrors the structure of the project's DCP rank0 broadcast loader
    (``load_utils.py``): rank0 opens each safetensors shard, broadcasts the
    per-shard ``param_info_list`` in one shot, then per-tensor broadcasts the
    tensor data. Every rank then runs the same dispatch as ``load_hf_weights``
    once it holds the full tensor. Total disk I/O is one read of the HF
    weights (vs ``world_size`` reads for ``load_hf_weights``), at the cost of
    cross-rank communication.

    Assumes ``model`` has been laid out by ``fully_shard`` and brought out of
    meta via ``to_empty_if_needed`` in ``get_model``.
    """
    rank0 = dist.get_rank() == 0
    torch_device = torch.device(get_device_type())

    param_names = {name for name, _ in model.named_parameters()}
    buffer_names = {name for name, _ in model.named_buffers()}
    lora_base_map = _lora_base_key_map(param_names) if enable_lora else {}
    unexpected_keys: Set[str] = set()

    if rank0:
        shard_paths = [s.filepath for s in locate_hf_weight_files(hf_dir)]
    else:
        shard_paths = []
    shard_count_tensor = torch.tensor(
        [len(shard_paths)] if rank0 else [0],
        dtype=torch.int64,
        device=torch_device,
    )
    dist.broadcast(shard_count_tensor, src=0)
    shard_count = int(shard_count_tensor.item())

    shard_iterable = tqdm(
        range(shard_count),
        desc="Loading HF checkpoint shards",
        disable=int(os.getenv("LOCAL_RANK", "-1")) > 0,
    )

    for shard_id in shard_iterable:
        # rank0 reads the whole shard into a dict and builds the per-shard param_info_list;
        # other ranks receive that list via broadcast.
        if rank0:
            shard_state: Dict[str, torch.Tensor] = {}
            with safe_open(shard_paths[shard_id], framework="pt", device="cpu") as f:
                for key in f.keys():
                    shard_state[key] = f.get_tensor(key)
            param_info_list = [
                ParamInfo(name=k, shape=v.shape, dtype=v.dtype)
                for k, v in shard_state.items()
            ]
        else:
            shard_state = {}
            param_info_list = []

        broadcast_list = [param_info_list]
        dist.broadcast_object_list(broadcast_list, src=0)
        param_info_list = broadcast_list[0]

        # Per-tensor broadcast + dispatch.
        for info in param_info_list:
            key = convert_weight_key(info.name, model)
            key = lora_base_map.get(key, key)  # bare -> base_layer for LoRA targets
            if rank0:
                tensor = shard_state[info.name].to(torch_device, non_blocking=True)
            else:
                tensor = torch.empty(info.shape, dtype=info.dtype, device=torch_device)
            dist.broadcast(tensor, src=0)

            if key in param_names:
                param_names.discard(key)
                write_full_tensor(model, key, tensor)
            elif key in buffer_names:
                write_full_tensor(model, key, tensor)
            else:
                unexpected_keys.add(key)
            del tensor

        del shard_state
        empty_cache()

    _log_unexpected_keys(unexpected_keys)
    post_process_after_load(model, missing_param_keys=param_names, load_strict=load_strict)
