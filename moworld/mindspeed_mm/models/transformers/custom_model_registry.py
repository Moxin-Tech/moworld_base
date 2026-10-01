from typing import Dict, Type, Optional, Set, List
import traceback
import importlib
import pkgutil

# Global registry state
_registry_global: Dict[str, Type] = {}
_failed_registrations: Dict[str, str] = {}
_imported_modules: Set[str] = set()

# Configurable auto-discovery package paths (supports runtime modification)
_AUTO_DISCOVERY_PACKAGES = [
    'mindspeed_mm.models.transformers',
]


# ==================== Configuration Management ====================
def set_auto_discovery_packages(packages: List[str]):
    """
    Set auto-discovery package paths

    Args:
        packages: List of package paths, e.g. ['mindspeed_mm.models.transformers', 'custom.models']
    """
    global _AUTO_DISCOVERY_PACKAGES
    _AUTO_DISCOVERY_PACKAGES = packages.copy()  # Copy to avoid external modification
    print(f"Set auto-discovery package paths: {_AUTO_DISCOVERY_PACKAGES}")


def add_auto_discovery_package(package: str):
    """
    Add a single package path to the auto-discovery list
    """
    global _AUTO_DISCOVERY_PACKAGES
    if package not in _AUTO_DISCOVERY_PACKAGES:
        _AUTO_DISCOVERY_PACKAGES.append(package)
        print(f"Added auto-discovery package path: {package}")


def remove_auto_discovery_package(package: str):
    """
    Remove a package path from the auto-discovery list
    """
    global _AUTO_DISCOVERY_PACKAGES
    if package in _AUTO_DISCOVERY_PACKAGES:
        _AUTO_DISCOVERY_PACKAGES.remove(package)
        print(f"Removed auto-discovery package path: {package}")


def get_auto_discovery_packages() -> List[str]:
    """
    Get currently configured auto-discovery package paths
    """
    return _AUTO_DISCOVERY_PACKAGES.copy()  # Return copy to avoid external modification


def reset_auto_discovery_packages():
    """
    Reset auto-discovery package paths to default values
    """
    global _AUTO_DISCOVERY_PACKAGES
    _AUTO_DISCOVERY_PACKAGES = [
        'mindspeed_mm.models.transformers',
    ]
    print("Reset auto-discovery package paths to default values")


# ==================== Model Registration ====================
def register_model(model_id: str = None, ignore_errors: bool = False):
    """
    Global model registration decorator
    """

    def decorator(model_cls: Type) -> Optional[Type]:
        key = model_id or model_cls.__name__.lower()

        # Check for duplicate registration
        if key in _registry_global:
            existing_cls = _registry_global[key]
            raise ValueError(
                f"Model ID '{key}' is already registered to {existing_cls.__name__}, "
                f"cannot register to {model_cls.__name__} again."
            )

        # Check if previously failed registration
        if key in _failed_registrations:
            if ignore_errors:
                print(
                    f"[WARNING] Model '{key}' previously failed registration: {_failed_registrations[key]}, skipping...")
                return model_cls
            else:
                raise ImportError(f"Model '{key}' previously failed: {_failed_registrations[key]}")

        try:
            # Basic validation
            if not hasattr(model_cls, '__name__'):
                raise ValueError("Model class must have a name")

            # Registration successful
            _registry_global[key] = model_cls
            print(f"Successfully registered model: {key} -> {model_cls.__name__}")
            return model_cls

        except Exception as e:
            # Record failure reason
            error_msg = f"Failed to register model '{key}': {e}"
            _failed_registrations[key] = error_msg

            if ignore_errors:
                print(f"[WARNING] {error_msg}")
                return model_cls
            else:
                print(f"Model registration failed: {error_msg}")
                raise

    return decorator


def get_model_class_global(model_id: str) -> Type:
    """
    Get model class from global registry
    """
    # Check if in failed registrations
    if model_id in _failed_registrations:
        raise ImportError(f"Model '{model_id}' registration failed: {_failed_registrations[model_id]}")

    # Check if registered
    if model_id not in _registry_global:
        # Try auto-discovery
        _auto_discover_model(model_id)

    # Check again after auto-discovery
    if model_id not in _registry_global:
        available = ", ".join(sorted(_registry_global.keys()))
        failed = ", ".join(sorted(_failed_registrations.keys()))

        error_msg = f"Model '{model_id}' not found."
        if available:
            error_msg += f" Available models: [{available}]"
        if failed:
            error_msg += f" Failed models: [{failed}]"

        raise ValueError(error_msg)

    return _registry_global[model_id]


def _auto_discover_model(model_id: str):
    """
    Automatically discover and import modules that may contain the requested model
    """
    print(f"Auto-discovering model: {model_id}")

    for package_path in _AUTO_DISCOVERY_PACKAGES:
        try:
            # Import the package to get its path
            package = importlib.import_module(package_path)

            # Use walk_packages to recursively import all modules
            _import_all_modules_recursive(package, package_path, model_id)

            if model_id in _registry_global:
                return

        except ImportError as e:
            print(f"[WARNING] Failed to scan package {package_path}: {e}")


def _import_all_modules_recursive(package, package_name: str, target_model_id: str):
    """
    Import all modules recursively using walk_packages
    """
    if not hasattr(package, '__path__'):
        return

    # Use walk_packages to get all modules recursively
    for _, module_name, is_pkg in pkgutil.walk_packages(package.__path__, package_name + '.'):
        # Skip packages, only import modules
        if not is_pkg:
            _import_module_if_needed(module_name, target_model_id)

            # Early return if target model found
            if target_model_id and target_model_id in _registry_global:
                return


def _import_module_if_needed(module_name: str, target_model_id: str):
    """
    Import a module if it hasn't been imported yet
    """
    if module_name in _imported_modules:
        return

    try:
        print(f"Importing module: {module_name}")
        importlib.import_module(module_name)
        _imported_modules.add(module_name)

    except ImportError as e:
        print(f"[WARNING] Failed to import module {module_name}: {e}")
        _imported_modules.add(module_name)
    except Exception as e:
        print(f"[WARNING] Error importing module {module_name}: {e}")
        _imported_modules.add(module_name)


def initialize_model_registry():
    """
    Initialize the model registry by scanning configured packages
    """
    print("Initializing model registry...")

    for package_path in _AUTO_DISCOVERY_PACKAGES:
        try:
            package = importlib.import_module(package_path)
            _import_all_modules_recursive(package, package_path, None)
        except ImportError as e:
            print(f"[WARNING] Failed to initialize package {package_path}: {e}")

    available_models = list(_registry_global.keys())
    print(f"Model registry initialized. Available models: {available_models}")
