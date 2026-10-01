import importlib
import pkgutil

from mindspeed.fsdp.utils.log import print_rank


def import_package(package_name, print_info=True):
    if "/" in package_name:
        package_name = package_name.replace("/", ".")
    package_name = package_name.strip(".")
    try:
        package = importlib.import_module(package_name)
        if print_info:
            print_rank(print, f"Successfully imported: {package_name}")
        return package, package_name
    except ModuleNotFoundError as e:
        if print_info:
            print_rank(print, f"Import failed: {package_name}. Error: {e}")
        else:
            raise e
    return None, None


def import_plugin(plugin_list=None):
    if not plugin_list:
        return
    for plugin_path in plugin_list:
        package, package_name = import_package(plugin_path, print_info=True)

        if not hasattr(package, "__path__"):
            continue

        for _, module_name, is_pkg in pkgutil.walk_packages(package.__path__, prefix=f"{package_name}."):
            if not is_pkg:
                _ = import_package(module_name, print_info=False)


class Register:

    def __init__(self):
        self._registry = {}

    def register(self, obj_id):

        def decorator(obj):
            nonlocal obj_id
            if obj_id in self._registry:
                registered = self._registry[obj_id]
                raise KeyError(f"ID '{obj_id}' is already registered to {registered.__name__}.")
            self._registry[obj_id] = obj
            return obj

        return decorator

    def get(self, obj_id):
        if obj_id not in self._registry:
            raise KeyError(f"ID '{obj_id}' is not registered.")
        return self._registry[obj_id]


model_register = Register()
data_register = Register()
