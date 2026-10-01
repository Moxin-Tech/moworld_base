import os
from functools import wraps
import json
import yaml


from mindspeed_mm.utils.utils import get_dtype


class ConfigReader:
    """
    read_config read json file dict processed by MMconfig
    and convert to class attributes, besides, read_config
    support to convert dict for specific purposes.
    """

    def __init__(self, config_dict: dict) -> None:
        for k, v in config_dict.items():
            if k == "dtype":
                v = get_dtype(v)
            if isinstance(v, dict):
                self.__dict__[k] = ConfigReader(v)
            else:
                self.__dict__[k] = v

    def to_dict(self) -> dict:
        ret = {}
        for k, v in self.__dict__.items():
            if isinstance(v, self.__class__):
                ret[k] = v.to_dict()
            else:
                ret[k] = v
        return ret

    def __repr__(self) -> str:
        for k, v in self.__dict__.items():
            if isinstance(v, self.__class__):
                print(">>>>> {}".format(k))
                print(v)
            else:
                print("{}: {}".format(k, v))
        return ""

    def __str__(self) -> str:
        try:
            self.__repr__()
        except Exception as e:
            print(f"An error occurred: {e}")
        return ""

    def get(self, key, default):
        return self.__dict__.get(key, default)

    def update_unuse(self, **kwargs):

        to_remove = []
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                to_remove.append(key)

        # remove all the attributes that were updated, without modifying the input dict
        unused_kwargs = {key: value for key, value in kwargs.items() if key not in to_remove}
        return unused_kwargs


class MMConfig:
    """
    MMconfig
        input: a dict of json path
    """

    def __init__(self, files: dict) -> None:
        errors = []
        for name, path in files.items():
            try:
                if path == "":
                    continue
                real_path = os.path.realpath(path)
                config_dict = None
                if real_path.endswith('.json'):
                    with open(real_path, 'r') as f:
                        config_dict = self.read_json(real_path)
                elif real_path.endswith('.yaml'):
                    with open(real_path, 'r') as f:
                        config_dict = yaml.safe_load(f)[name]
                setattr(self, name, ConfigReader(config_dict))
            except FileNotFoundError as e:
                errors.append(f"Warning: Config file not found: {path} - {e}")
            except PermissionError as e:
                errors.append(f"Error: Permission denied when reading: {path} - {e}")
            except json.JSONDecodeError as e:
                errors.append(f"Error: Invalid JSON format in {path}: {e}")
            except yaml.YAMLError as e:
                errors.append(f"Error: Invalid YAML format in {path}: {e}")
            except Exception as e:
                errors.append(f"Error: Unexpected error loading {path}: {e}")

        if errors:
            for error in errors:
                print(error)
            raise RuntimeError("One or more config files failed to load. See above errors.")

    @staticmethod
    def read_json(json_path):
        with open(json_path, mode="r") as f:
            json_file = f.read()
        config_dict = json.loads(json_file)
        return config_dict


def _add_mm_args(parser):
    group = parser.add_argument_group(title="multimodel")
    group.add_argument("--mm-data", type=str, default="")
    group.add_argument("--mm-model", type=str, default="")
    group.add_argument("--mm-tool", type=str, default="")
    return parser


def mm_extra_args_provider(parser):
    parser = _add_mm_args(parser)
    return parser


def merge_mm_args_decorator(func):
    called = False

    @wraps(func)
    def wrapper(args):
        func(args)
        nonlocal called
        if not called:
            args_external_path_checker(args)
            called = True

    return wrapper


@merge_mm_args_decorator
def merge_mm_args(args):
    if not hasattr(args, "mm"):
        setattr(args, "mm", object)
        json_files = {"model": args.mm_model, "data": args.mm_data, "tool": args.mm_tool}
        args.mm = MMConfig(json_files)


def args_external_path_checker(args):
    """
    Verify the security of all file path parameters in 3 code repositories:mindspeed-mm,mindspeed,megatron
    and 3 json file:mm_data.json,mm_model.json,mm_tool.json
    """
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "validate_params.json"), 'r') as f:
            validation_params = json.load(f)
    except OSError:
        print("WARNING: validata_params.json can't open")

    # args from mindspeed_mm
    if "mindspeed_mm" in validation_params:
        mindspeed_mm_params = validation_params["mindspeed_mm"]["params"]
        for param in mindspeed_mm_params:
            if hasattr(args, param) and getattr(args, param):
                file_legality_checker(getattr(args, param), param)

    # args from mindspeed
    if "mindspeed" in validation_params:
        mindspeed_param = validation_params["mindspeed"]["params"]
        for param in mindspeed_param:
            if hasattr(args, param) and getattr(args, param):
                file_legality_checker(getattr(args, param), param)

    # args from megatron
    if "megatron" in validation_params:
        megatron_param = validation_params["megatron"][0]["params"]
        for param in megatron_param:
            if hasattr(args, param) and getattr(args, param):
                file_legality_checker(getattr(args, param), param)

        # These parameters may have the following format:weight path weight path
        megatron_special_params = validation_params["megatron"][1]["params"]
        for param in megatron_special_params:
            if hasattr(args, param) and getattr(args, param):
                file_list = split_special_megatron_param(param)
                for path in file_list:
                    file_legality_checker(path, param)

    # args from MM_ModeL
    if "mm_model" in validation_params:
        mm_model_params = validation_params["mm_model"]["params"]
        if hasattr(args.mm, "model"):
            for param in mm_model_params:
                values = get_ConfigReader_value(args.mm.model, param)
                for value in values:
                    if value:
                        file_legality_checker(value, param)

    # args from MM_Data
    if "mm_data" in validation_params:
        mm_data_params = validation_params["mm_data"]["params"]
        if hasattr(args.mm, "data"):
            for param in mm_data_params:
                values = get_ConfigReader_value(args.mm.data, param)
                for value in values:
                    if value:
                        file_legality_checker(value, param)

    # args from MM_Tool
    if "mm_tool" in validation_params:
        mm_tool_params = validation_params["mm_tool"]["params"]
        if hasattr(args.mm, "tool"):
            for param in mm_tool_params:
                values = get_ConfigReader_value(args.mm.tool, param)
                for value in values:
                    if value:
                        file_legality_checker(value, param)


def file_legality_checker(file_path, param_name, base_dir=None):
    """
    Perform soft link and path traversal checks on file path
    """
    if not base_dir:
        base_dir = os.getcwd()

    # check file exist
    try:
        if not os.path.exists(file_path):
            return False
    except OSError:
        return False

    # check symbolic link
    from mindspeed_mm.utils.security_utils.validate_path import normalize_path
    try:
        norm_path, is_link = normalize_path(file_path)
        if is_link:
            print(
                "WARNING: [{}] {} is a symbolic link.It's normalize path is {}".format(param_name, file_path,
                                                                                       norm_path))
            return False
    except OSError:
        return False

    # check path crossing
    try:
        # get absolute file path
        norm_path = os.path.realpath(file_path)
        # get absolute base dir path
        base_directory = os.path.abspath(base_dir)
        if not norm_path.startswith(base_directory):
            print("WARNING: [{}] {} attempts to traverse to an disallowed directory".format(param_name, file_path))
            return False
    except OSError:
        return False

    return True


def split_special_megatron_param(param):
    """
    Segment some special parameters in megatron,for example:data_path,train_data_path..
    """

    def is_number(s):
        if isinstance(s, str):
            s = s.strip()
        try:
            float(s)
            return True
        except (ValueError, TypeError):
            return False

    param_list = param.split(" ")
    if len(param_list) == 1:
        return param_list
    else:
        if is_number(param_list[0]):
            return [param_list[2 * i] for i in range(len(param_list) // 2)]
        else:
            return param_list


def get_ConfigReader_value(config, param):
    objs = [config.to_dict()]
    for key in param.split("."):
        new_objs = []
        for obj in objs:
            if not obj:
                continue
            if key in obj:
                if not obj[key]:
                    continue
                if isinstance(obj[key], list):
                    new_objs.extend(obj[key])
                else:
                    new_objs.append(obj[key])
        if new_objs:
            objs = new_objs
        else:
            return []

    return objs
