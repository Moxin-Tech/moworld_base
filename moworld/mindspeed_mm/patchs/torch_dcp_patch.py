# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
import os
import sys
import torch
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

from megatron.training.checkpointing import (
    _NON_PERSISTENT_CKPT_SUBDIR,
    _get_non_persistent_iteration,
    get_checkpoint_tracker_filename,
    read_metadata,
    _load_non_persistent_base_checkpoint,
    get_checkpoint_name,
    _get_checkpoint_format,
    _load_global_dist_base_checkpoint,
    CheckpointType,
    find_checkpoint_rank_0
)
from megatron.training.utils import print_rank_0
from mindspeed.megatron_adaptor import get_mindspeed_args
from mindspeed.patch_utils import MindSpeedPatchesManager as pm
from mindspeed_mm.models.transformers.global_vars import get_check_moe_func, get_ep_size, get_ep_rank
from mindspeed_mm.models.transformers.moe_utils import EPLoadPlanner
from mindspeed_mm.tasks.finetune.lora.utils import is_enable_lora, remove_base_layer_keys, restore_base_layer_keys


def _load_base_checkpoint(
    load_dir,
    args,
    rank0=False,
    sharded_state_dict=None,
    checkpointing_context=None,
):
    """ Load the base state_dict from the given directory

    If rank0 is true, just loads rank 0 checkpoint, ignoring arguments.
    """
    # Try to load non-persistent checkpoint first
    non_persistent_global_dir = (
        args.non_persistent_global_ckpt_dir
        if args.non_persistent_global_ckpt_dir or load_dir is None
        else os.path.join(load_dir, _NON_PERSISTENT_CKPT_SUBDIR)
    )
    non_persistent_iteration = _get_non_persistent_iteration(
        non_persistent_global_dir, args, checkpointing_context
    )
    iteration, release = -1, False
    tracker_filename = 'because load directory is not defined'
    if load_dir is not None:
        tracker_filename = get_checkpoint_tracker_filename(load_dir)
        if os.path.isfile(tracker_filename):
            iteration, release = read_metadata(tracker_filename)

    # Allow user to specify the loaded iteration.
    if getattr(args, "ckpt_step", None):
        iteration = args.ckpt_step

    if non_persistent_iteration != -1:  # there is a non-persistent checkpoint
        if non_persistent_iteration >= iteration:
            return _load_non_persistent_base_checkpoint(
                non_persistent_global_dir,
                args,
                rank0,
                sharded_state_dict,
                non_persistent_iteration,
                checkpointing_context,
            )
        else:
            print_rank_0('WARNING: non-persistent checkpoints are older than persistent checkpoint')

    # Otherwise we are dealing with global checkpoints
    # If no tracker file, return nothing
    if iteration == -1:
        if not rank0:
            print_rank_0('WARNING: could not find the metadata file {}'.format(tracker_filename))
            print_rank_0('    will not load any checkpoints and will start from random')
        # Conditionally exit if checkpoint not found.
        if args.exit_on_missing_checkpoint:
            print_rank_0(">> '--exit-on-missing-checkpoint' set ... exiting. <<")
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            sys.exit()

        return None, "", False, None

    # Determine the type of the checkpoint on disk.
    checkpoint_name = get_checkpoint_name(load_dir, iteration, release, return_base_dir=True)
    ckpt_format = _get_checkpoint_format(checkpoint_name)

    if not rank0:
        dist_infix = "distributed " if ckpt_format == "torch_dist" else ""
        if release:
            print_rank_0(f' loading release {dist_infix}checkpoint from {load_dir}')
        else:
            print_rank_0(
                f' loading {dist_infix}checkpoint from {load_dir} at iteration {iteration}'
            )

    ckpt_type = None

    # Handle global distributed checkpoint
    if ckpt_format == "torch_dist":
        return _load_global_dist_base_checkpoint(
            load_dir, args, rank0, sharded_state_dict, iteration, release, checkpointing_context=checkpointing_context
        )
    elif ckpt_format == "torch":
        ckpt_type = CheckpointType.LEGACY
        # Handle global legacy checkpoint
        if rank0:
            checkpoint_name = find_checkpoint_rank_0(load_dir, iteration, release)
        else:
            checkpoint_name = get_checkpoint_name(load_dir, iteration, release, return_base_dir=False)
        try:
            state_dict = torch.load(checkpoint_name, map_location='cpu', weights_only=False)
        except ModuleNotFoundError:
            from megatron.legacy.fp16_deprecated import loss_scaler

            # For backward compatibility.
            if not rank0:
                print_rank_0(' > deserializing using the old code structure ...')
            sys.modules['fp16.loss_scaler'] = sys.modules['megatron.legacy.fp16_deprecated.loss_scaler']
            sys.modules['megatron.fp16.loss_scaler'] = sys.modules[
                'megatron.legacy.fp16_deprecated.loss_scaler'
            ]
            sys.modules['megatron.model'] = sys.modules['megatron.legacy.model']
            state_dict = torch.load(checkpoint_name, map_location='cpu', weights_only=False)
            sys.modules.pop('fp16.loss_scaler', None)
            sys.modules.pop('megatron.fp16.loss_scaler', None)
            sys.modules.pop('megatron.model', None)
        except Exception as e:
            print('could not load the checkpoint')
            print(e)
            sys.exit()
    elif ckpt_format == "torch_dcp":
        ckpt_type = CheckpointType.TORCH_DCP

        if rank0:
            # _load_base_checkpoint is called from load_args_from_checkpoint. torch.distributed is not initialized.
            # Load only metadata.
            state_dict = {"args": None, "iteration": None}
            torch.distributed.checkpoint.load(
                state_dict=state_dict,
                checkpoint_id=checkpoint_name,
            )
        else:
            # _load_base_checkpoint is called from load_checkpoint with a proper state dict.
            state_dict = sharded_state_dict

            if release:
                state_dict.pop('args', None)
                state_dict.pop('rerun_state_machine', None)
                state_dict.pop('num_floating_point_operations_so_far', None)

            # remove '_extra_state'
            _extra_state_list = [i for i in state_dict.get('model', {}).keys() if '_extra_state' in i]
            for _extra_state in _extra_state_list:
                if _extra_state in state_dict.get('model', {}):
                    state_dict['model'].pop(_extra_state)

            ep_size = get_ep_size()
            ep_rank = get_ep_rank()
            check_moe_func = get_check_moe_func()
            if ep_size <= 1:
                load_planner = DefaultLoadPlanner(allow_partial_load=True)
            else:
                load_planner = EPLoadPlanner(allow_partial_load=True, ep_rank=ep_rank, ep_size=ep_size, check_moe_fn=check_moe_func)

            if is_enable_lora():
                key_mapping = remove_base_layer_keys(state_dict.get('model', None))

            torch.distributed.checkpoint.load(
                state_dict=state_dict,
                checkpoint_id=checkpoint_name,
                planner=load_planner
            )

            if is_enable_lora():
                restore_base_layer_keys(state_dict.get('model', None), key_mapping)

            curr_keys = load_planner.state_dict.keys()
            load_keys = load_planner.metadata.state_dict_metadata.keys()
            unexpected_keys = set(load_keys) - set(curr_keys)
            missing_keys = set(curr_keys) - set(load_keys)
            model_unexpected_keys = [item[len("model."):] for item in unexpected_keys if item.startswith("model.")]
            model_missing_keys = [item[len("model."):] for item in missing_keys if item.startswith("model.")]

            print_rank_0(f"Missing keys: {model_missing_keys}")
            print_rank_0(f"Unexpected keys: {model_unexpected_keys}")
    else:
        raise NotImplementedError(f"checkpoint format {ckpt_format} not supported")
    # By default FSDP2 mixed precision keeps the original weight precision.
    # To match the numerical behavior of  --bf16 , we added the  --downcast-to-bf16  option.
    if args.downcast_to_bf16 and isinstance(state_dict, dict) and 'model' in state_dict:
        for k in state_dict['model']:
            if state_dict['model'][k].dtype == torch.float:
                state_dict['model'][k].data.copy_(state_dict['model'][k].bfloat16().float())

    return state_dict, checkpoint_name, release, ckpt_type


mindspeed_args = get_mindspeed_args()
if hasattr(mindspeed_args, 'ckpt_format') and mindspeed_args.ckpt_format == 'torch_dcp' and \
        not hasattr(mindspeed_args, 'lora_target_modules'):
    pm.register_patch('megatron.training.checkpointing._load_base_checkpoint', _load_base_checkpoint, force_patch=True)
