import sys
from copy import deepcopy
import yaml

# TO CHECK these args when used, the default type for these args are not right.
# _BOOL_TYPE_ARGS = ['lora-mixed-training', 'lazy-mpu-init', 'onnx-safe']


def get_sys_args_from_yaml():
    def flatten_config(d):
        """
        Recursively expand nested dictionaries, retaining only the key-value pairs at the lowest level.
        Underscores `_` in key names will be converted to hyphens `-`.
        """
        items = []

        def _flatten(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, dict):
                        _flatten(v)
                    else:
                        key_flat = k.replace('_', '-')
                        items.append((key_flat, v))
            else:
                pass

        _flatten(d)
        return dict(items)

    first_arg = sys.argv[1]
    if not first_arg.endswith('.yaml'):
        return

    config_path = first_arg
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError as e:
        print(f"Warning: Config file not found: {config_path} - {e}")
    except PermissionError as e:
        print(f"Error: Permission denied when reading: {config_path} - {e}")
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML format in {config_path} - {e}")
    except Exception as e:
        print(f"Error: Unexpected error loading {config_path} - {e}")

    original_argv = deepcopy(sys.argv)
    new_argv = [original_argv[0]]  # script name
    flat_args = flatten_config(config['gpt_args'])
    if 'gpt_args' not in config:
        raise KeyError("The required keyword 'gpt_args' is missing from the configuration file.")
    if 'lora_args' in config:
        flat_args.update(flatten_config(config['lora_args']))
    for key, value in flat_args.items():
        if isinstance(value, bool) and value:
            new_argv.append(f'--{key}')
        elif isinstance(value, list):
            for v in value:
                new_argv.append(f'--{key}')
                new_argv.append(str(v))
        else:
            new_argv.append(f'--{key}')
            new_argv.append(str(value))

    if 'MM_TOOL_PATH' not in config:
        raise KeyError("The required keyword 'MM_TOOL_PATH' is missing from the configuration file.")
    new_argv.append(f'--mm-data')
    new_argv.append(config_path)
    new_argv.append(f'--mm-model')
    new_argv.append(config_path)
    new_argv.append(f'--mm-tool')
    new_argv.append(config['MM_TOOL_PATH'])

    sys.argv = new_argv
    return


get_sys_args_from_yaml()
