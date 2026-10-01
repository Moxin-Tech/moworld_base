# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import os
import logging

logger = logging.getLogger(__name__)


def get_checkpoint_name(checkpoints_path, iteration, release=False):
    """Determine the directory name for this rank's checkpoint."""
    if release:
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(iteration)

    common_path = os.path.join(checkpoints_path, directory)
    return common_path


def get_checkpoint_tracker_filename(checkpoints_path):
    """Tracker file rescords the latest chckpoint during training to restart from."""
    return os.path.join(checkpoints_path, 'latest_checkpointed_iteration.txt')


def read_metadata(tracker_filename):
    # Read the tracker file and either set the iteration or
    # mark it as a release checkpoint.
    iteration = 0
    release = False
    with open(tracker_filename, 'r', encoding='utf-8') as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError as e:
            release = metastring == 'release'
            if not release:
                raise ValueError('ERROR: Invalid metadata file {}.'.format(tracker_filename)) from e
    if not (iteration > 0 or release):
        print('error parsing metadata file {}'.format(tracker_filename))

    return iteration, release


def remove_base_layer_keys(state_dict):
    if state_dict is None or not isinstance(state_dict, dict):
        return {}

    key_mapping = {}
    original_keys = list(state_dict.keys())

    for old_key in original_keys:
        if '.base_layer' in old_key:
            new_key = old_key.replace('.base_layer', '')
            key_mapping[old_key] = new_key
            state_dict[new_key] = state_dict.pop(old_key)

    return key_mapping


def restore_base_layer_keys(modified_state_dict, key_mapping):
    if modified_state_dict is None or not isinstance(modified_state_dict, dict):
        return
    if key_mapping is None or not isinstance(key_mapping, dict):
        return

    reverse_mapping = {new_key: orig_key for orig_key, new_key in key_mapping.items()}
    modified_keys = list(modified_state_dict.keys())

    for key in modified_keys:
        original_key = reverse_mapping.get(key, key)
        if original_key != key:
            modified_state_dict[original_key] = modified_state_dict.pop(key)
