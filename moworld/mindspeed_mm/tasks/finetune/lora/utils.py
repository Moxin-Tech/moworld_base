# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from megatron.training import get_args


def is_enable_lora():
    args = get_args()
    if hasattr(args, 'lora_target_modules') and args.lora_target_modules:
        return True
    return False


def is_save_full_weight():
    args = get_args()
    mm_model = args.mm_model
    save_full_weight = getattr(mm_model, 'lora_save_full_weight', False)
    return save_full_weight


def merge_dicts(statedict1, statedict2):
    result = statedict1
    for key, value in statedict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def modify_keys_with_dict(dictionary, exclude_words):
    args = get_args()
    modified_dict = {}
    for key, value in dictionary.items():
        key_str = str(key)
        not_exclude_word = not any(exclude_word in key_str for exclude_word in exclude_words)
        is_target_module_bias = any(key_str in target_module + '.bias' for target_module in args.lora_trainable_target_modules)
        is_target_module_weight = any(key_str in target_module + '.weight' for target_module in args.lora_trainable_target_modules)

        new_key = key_str
        if not_exclude_word and (is_target_module_bias or is_target_module_weight):
            if 'weight' in key_str:
                new_key = key_str.replace('weight', 'base_layer.weight')
            elif 'bias' in key_str:
                new_key = key_str.replace('bias', 'base_layer.bias')
        modified_dict[new_key] = value

    return modified_dict


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
