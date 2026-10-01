import os
import sys
import json
from typing import Dict, Any, Optional, List, Type
from pathlib import Path
import yaml

from mindspeed_mm.config.arguments.base_args import BaseArguments
from mindspeed_mm.fsdp.params.argument import Arguments
from mindspeed_mm.config.exception import ConfigValidationError


class ConfigManager:
    """
    Configuration manager based on BaseArguments
    Design principles:
    1. Only support key=value format command-line arguments
    2. Use BaseArguments structured pattern to ensure type safety
    3. Prohibit adding any undefined fields via command line
    4. Allow limited YAML extensions
    5. Support configuration class inheritance and dynamic fields
    6. Dynamic fields only support basic types and lists
    """

    def __init__(self,
                 config_class: Type[BaseArguments] = Arguments,
                 config_file_path: Optional[str] = None,
                 additional_args: Optional[Dict[str, Any]] = None,
                 allow_yaml_extensions: bool = True,
                 allow_cli_override: bool = True,
                 strict_cli_validation: bool = True,
                 allow_register_yaml_fields: bool = True):
        """
        Initialize configuration manager

        Args:
            config_class: Configuration class, must be a subclass of BaseArguments
            config_file_path: Configuration file path
            additional_args: Additional configuration parameters (passed from code)
            allow_yaml_extensions: Whether to allow YAML configuration extensions
            allow_cli_override: Whether to allow command-line override of existing configurations
            strict_cli_validation: Whether to strictly validate command-line arguments
            allow_register_yaml_fields: Whether to allow automatic registration of newly defined extension fields from YAML, allowed by default.
        """
        if not config_class:
            config_class = Arguments

        if not issubclass(config_class, BaseArguments):
            raise TypeError(f"config_class must be a subclass of BaseArguments: {config_class}")

        self.config_class = config_class
        self.allow_yaml_extensions = allow_yaml_extensions
        self.allow_cli_override = allow_cli_override
        self.strict_cli_validation = strict_cli_validation
        self.allow_register_yaml_fields = allow_register_yaml_fields

        # Additional configuration parameters
        self.additional_args = additional_args or {}

        # Get configuration file path
        self.config_file_path = self.get_config_file_path(config_file_path)

        # Final configuration object
        self._config: Optional[BaseArguments] = None

        # Collect all defined fields (including those from parent classes) and their type information
        self._defined_fields_info: Dict[str, Dict[str, Any]] = {}
        self._collect_defined_fields()

        # Collect all dynamic fields (including those from parent classes)
        self._dynamic_fields = set()
        self._collect_dynamic_fields()

    def get_config_file_path(self, config_file_path: Optional[str]) -> Optional[str]:
        """Get configuration file path"""
        if config_file_path:
            if not Path(config_file_path).exists():
                raise FileNotFoundError(f"Configuration file does not exist: {config_file_path}")
            return config_file_path

        if len(sys.argv) >= 2 and sys.argv[1].endswith(('.yaml', '.yml')):
            file_path = sys.argv[1]
            if Path(file_path).exists():
                return file_path

        return None

    def load_and_parse(self) -> BaseArguments:
        """Load and parse all configuration sources"""
        _print("[INFO] Starting configuration loading...")

        try:
            # 1. Load configuration sources
            yaml_config = self._load_yaml_config()
            cli_config = self._parse_cli_args() if self.allow_cli_override else {}

            # 2. Process YAML extensions
            self._process_yaml_extensions(yaml_config)

            # 3. Validate command-line arguments
            self._validate_cli_args(cli_config)

            # 4. Safely merge configurations
            merged_config = self._safe_merge_configs(yaml_config, cli_config, self.additional_args)

            # 5. Check required fields
            self._check_required_fields(merged_config)

            # 6. Create configuration object
            config_obj = self.config_class(**merged_config)

            # 7. Validate configuration object
            self._validate_config_object(config_obj, merged_config)

            self._config = config_obj

            self.print_summary()

            return config_obj

        except Exception as e:
            _print(f"[ERROR] Configuration loading failed: {e}")
            if isinstance(e, ConfigValidationError):
                raise
            else:
                raise ConfigValidationError(f"Configuration loading failed") from e

    def get_config(self) -> Optional[BaseArguments]:
        """Get current configuration"""
        return self._config

    def get_defined_fields(self) -> List[str]:
        """Get all defined fields (including dynamic fields)"""
        all_fields = (sorted(list(self._defined_fields_info.keys()) +
                             list(self._dynamic_fields)))
        return all_fields

    def register_dynamic_field(self,
                               name: str,
                               value_type: Optional[Type] = None,
                               default: Any = None,
                               description: str = "",
                               required: bool = False):
        """
        Register dynamic field, support nested fields
        This is a wrapper for the configuration class's register_field method

        Args:
            name: Field name, supports dot notation, e.g., "data.dataset.id"
            value_type: Field type
            default: Default value
            description: Field description
            required: Whether required
        """
        self._register_dynamic_field_nested(
            name=name,
            value_type=value_type,
            default=default,
            description=description,
            required=required
        )

    def print_summary(self):
        """Print configuration summary"""
        if not self._config:
            _print("[WARNING] Configuration not loaded")
            return

        # Build output line list
        lines = [
            "=" * 60,
            "BaseArguments Configuration Manager Summary",
            "=" * 60,
            f"\nConfiguration class: {self.config_class.__name__}",
            f"Configuration file: {self.config_file_path or 'None'}",
            f"Allow YAML extensions: {self.allow_yaml_extensions}",
            f"Allow CLI override: {self.allow_cli_override}",
            f"Strict CLI validation: {self.strict_cli_validation}",
            f"Allow registering YAML fields: {self.allow_register_yaml_fields}",
        ]

        defined_count = len(self._defined_fields_info)
        dynamic_count = len(self._dynamic_fields)

        lines.extend([
            f"\nDefined field count: {defined_count}",
            f"Dynamic field count: {dynamic_count}",
            f"Total field count: {defined_count + dynamic_count}",
        ])

        # Add current configuration
        if self._config:
            lines.append("\n\n============ Configuration Details ============")
            lines.append(f"{self._config.to_str()}")

        # Add field details
        if self._dynamic_fields:
            lines.append("\n\n============ Dynamic Fields ============")
            for field_name in sorted(self._dynamic_fields):
                lines.append(f" {field_name}")

        lines.append("=" * 60)

        # Print all at once
        _print("\n".join(lines))

    def save_config(self, file_path: str, include_dynamic: bool = True, include_yaml: bool = True):
        """Save configuration to file"""
        if not self._config:
            raise ValueError("Configuration not loaded")

        config_dict = self._get_config_dict(self._config)

        filtered_dict = {}
        for field_name, value in config_dict.items():
            if field_name in self._dynamic_fields and not include_dynamic:
                continue
            filtered_dict[field_name] = value

        with open(file_path, 'w', encoding='utf-8') as f:
            if file_path.endswith(('.yaml', '.yml')):
                yaml.dump(filtered_dict, f, default_flow_style=False, allow_unicode=True)
            elif file_path.endswith('.json'):
                json.dump(filtered_dict, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported configuration file format: {file_path}")

        print(f"[INFO] Configuration saved to: {file_path}")

    def _collect_defined_fields(self):
        """Collect all fields defined in the class"""
        self._defined_fields_info = {}

        def _collect_fields(cls: Type, prefix: str = ""):
            """Recursively collect fields"""
            # Process class annotated fields
            if hasattr(cls, '__annotations__'):
                for field_name, field_type in cls.__annotations__.items():
                    full_name = f"{prefix}{field_name}"

                    # Check if field type is a BaseArguments subclass
                    if isinstance(field_type, type) and issubclass(field_type, BaseArguments):
                        # BaseArguments subclass, process recursively
                        new_prefix = f"{full_name}."
                        _collect_fields(field_type, new_prefix)
                    else:
                        # Regular field, record directly
                        self._defined_fields_info[full_name] = {
                            'type': field_type,
                            'cls': cls,
                            'field_name': field_name
                        }

            # Recursively process parent classes
            for base_cls in cls.__bases__:
                # Only process BaseArguments subclasses (exclude BaseArguments itself)
                if issubclass(base_cls, BaseArguments) and base_cls is not BaseArguments:
                    _collect_fields(base_cls, prefix)

        _collect_fields(self.config_class)

    def _collect_dynamic_fields(self):
        """Collect all dynamic fields (including inherited from parent classes)"""
        self._dynamic_fields = set()

        def _collect_dynamic_fields_recursive(cls: Type[BaseArguments]):
            # Collect dynamic fields of current class
            if hasattr(cls, '_dynamic_fields_info'):
                for field_name in cls._dynamic_fields_info:
                    self._dynamic_fields.add(field_name)

            # Recursively collect dynamic fields from parent classes
            for base_cls in cls.__bases__:
                if issubclass(base_cls, BaseArguments):
                    _collect_dynamic_fields_recursive(base_cls)

        _collect_dynamic_fields_recursive(self.config_class)

    def _get_nested_attribute(self, obj: Any, attr_path: str, create_missing: bool = False) -> Any:
        """
        Get nested attribute

        Args:
            obj: Object
            attr_path: Attribute path, e.g., "data.dataset.id"
            create_missing: Whether to create missing intermediate attributes

        Returns:
            (parent_obj, last_attr_name) or None
        """
        if not attr_path or not isinstance(obj, (BaseArguments, dict)):
            return None

        parts = attr_path.split('.')
        current = obj

        # Traverse to the second-to-last part
        for i, part in enumerate(parts[:-1]):
            if isinstance(current, BaseArguments):
                if hasattr(current, part):
                    current = getattr(current, part)
                elif create_missing:
                    # Create intermediate object
                    setattr(current, part, {})
                    current = getattr(current, part)
                else:
                    return None
            elif isinstance(current, dict):
                if part in current:
                    current = current[part]
                elif create_missing:
                    current[part] = {}
                    current = current[part]
                else:
                    return None
            else:
                return None

        return current, parts[-1]

    def _set_nested_value(self, config_dict: Dict[str, Any], attr_path: str, value: Any):
        """
        Set value in nested dictionary

        Args:
            config_dict: Configuration dictionary
            attr_path: Attribute path, e.g., "data.dataset.id"
            value: Value to set
        """
        if not attr_path:
            return

        parts = attr_path.split('.')
        current = config_dict

        # Traverse to the second-to-last part
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set value for the last part
        current[parts[-1]] = value

    def _get_nested_value(self, config_dict: Dict[str, Any], attr_path: str, default: Any = None) -> Any:
        """
        Get value from nested dictionary

        Args:
            config_dict: Configuration dictionary
            attr_path: Attribute path, e.g., "data.dataset.id"
            default: Default value
        """
        if not attr_path:
            return default

        parts = attr_path.split('.')
        current = config_dict

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default

        return current

    def _validate_cli_args(self, cli_args: Dict[str, Any]):
        """Validate command-line arguments, support nested fields"""
        if not cli_args or not self.strict_cli_validation:
            return

        for key in cli_args:
            # Check if it's a defined field
            if key in self._defined_fields_info or key in self._dynamic_fields:
                continue

            # Check if it's a nested field
            is_nested_field = '.' in key
            if is_nested_field:
                # Parse nested field path
                parts = key.split('.')
                parent_path = '.'.join(parts[:-1])
                field_name = parts[-1]

                # Get parent class information
                parent_cls = self._get_parent_info(parent_path)
                if parent_cls:
                    # Parent class exists, check if it can be extended
                    if isinstance(parent_cls, type) and issubclass(parent_cls, BaseArguments):
                        # If parent class is BaseArguments or its subclass, new fields can be added
                        continue
                    else:
                        # Other types cannot add new fields
                        raise ConfigValidationError(
                            f"Command-line argument '{key}' parent class type does not support extension: {parent_cls}"
                        )
                else:
                    # Parent path does not exist
                    raise ConfigValidationError(
                        f"Command-line argument '{key}' parent path does not exist\n"
                        f"Allowed fields: {sorted(list(self._defined_fields_info.keys()) + list(self._dynamic_fields))}"
                    )
            else:
                # Non-nested field, but undefined
                raise ConfigValidationError(
                    f"Command-line argument '{key}' is not defined in the configuration class.\n"
                    f"Allowed fields: {sorted(list(self._defined_fields_info.keys()) + list(self._dynamic_fields))}\n"
                    f"Please add this field via YAML configuration file or by inheriting the configuration class."
                )

    def _get_parent_info(self, attr_path: str) -> Optional[type]:
        """
        Get parent class information

        Args:
            attr_path: Attribute path, e.g., "data.dataset"

        Returns:
            Parent class or None
        """
        if not attr_path:
            return None

        # Look up from defined field information
        if attr_path in self._defined_fields_info:
            field_info = self._defined_fields_info[attr_path]
            return field_info.get('cls')

        # Check if it's a parent path of nested attributes
        parts = attr_path.split('.')
        parent_path = '.'.join(parts[:-1])

        if not parent_path:  # Already at top level, but no definition found
            return None

        # Get parent path class information
        if parent_path in self._defined_fields_info:
            parent_field_info = self._defined_fields_info[parent_path]
            return parent_field_info.get('cls')

        return None

    def _process_yaml_extensions(self, yaml_config: Dict[str, Any]):
        """Process extension configurations in YAML, support nested fields"""
        if not self.allow_yaml_extensions or not yaml_config:
            return

        def _process_dict(data: Dict[str, Any], prefix: str = ""):
            for key, value in data.items():
                full_key = f"{prefix}.{key}" if prefix else key

                # Skip defined fields
                if full_key in self._defined_fields_info or full_key in self._dynamic_fields:
                    if isinstance(value, dict):
                        _process_dict(value, full_key)
                    continue

                if isinstance(value, dict):
                    # Recursively process nested dictionaries
                    _process_dict(value, full_key)
                else:
                    # Do not allow defining new extension configuration item fields from YAML
                    if not self.allow_register_yaml_fields:
                        _print(f"[WARNING] Found undefined field '{full_key}' in YAML, ignored")
                        continue

                    # Automatically register fields from YAML file
                    value_type = self._infer_yaml_value_type(value)

                    # Automatically register to configuration class
                    try:
                        self._register_dynamic_field_nested(
                            name=full_key,
                            value_type=value_type,
                            default=value,
                            description=f"Field automatically registered from YAML file {self.config_file_path}"
                        )
                        _print(f"[INFO] Automatically registered YAML field: {full_key} (type: {value_type}, value: {value})")
                    except Exception as e:
                        _print(f"[WARNING] Failed to register YAML field {full_key}: {e}")

        _process_dict(yaml_config)

    def _register_dynamic_field_nested(self,
                                       name: str,
                                       value_type: Optional[Type] = None,
                                       default: Any = None,
                                       description: str = "",
                                       required: bool = False):
        """
        Register nested dynamic field

        Args:
            name: Field name, supports dot notation, e.g., "data.dataset.id"
            value_type: Field type
            default: Default value
            description: Field description
            required: Whether required
        """

        if '.' not in name:
            # Non-nested field, register directly
            self.config_class.register_field(
                name=name,
                value_type=value_type,
                default=default,
                description=description,
                required=required
            )
            self._dynamic_fields.add(name)
            return

        # Nested field, parse path
        parts = name.split('.')
        current_class = self.config_class
        current_path = []

        # Traverse each part of the path (except the last part)
        for part in parts[:-1]:
            current_path.append(part)
            path_str = '.'.join(current_path)

            # Check if current class has this attribute
            if hasattr(current_class, '__annotations__') and part in current_class.__annotations__:
                field_type = current_class.__annotations__[part]

                # Check if it's a BaseArguments subclass
                if isinstance(field_type, type) and issubclass(field_type, BaseArguments):
                    # Recursively go to next level
                    current_class = field_type
                    continue

                # Not a BaseArguments subclass, stop recursion
                _print(f"[WARNING] Field '{part}' exists but is not a BaseArguments subclass: {field_type}")
                # Register full path in top-level class
                self.config_class.register_field(
                    name=name,  # Keep original name
                    value_type=value_type,
                    default=default,
                    description=f"{description} (full path)",
                    required=required
                )
                self._dynamic_fields.add(name)
                return

            # Attribute with specified name doesn't exist, stop recursion
            _print(f"[WARNING] Field path '{path_str}' does not exist")
            # Register full path in top-level class
            self.config_class.register_field(
                name=name,  # Keep original name
                value_type=value_type,
                default=default,
                description=f"{description} (full path)",
                required=required
            )
            self._dynamic_fields.add(name)
            return

        # If successfully traversed to target class
        field_name = parts[-1]

        # Register field in target class
        current_class.register_field(
            name=field_name,
            value_type=value_type,
            default=default,
            description=description,
            required=required
        )

        # Record dynamic field (including full path)
        self._dynamic_fields.add(name)

    def _parse_cli_args(self) -> Dict[str, Any]:
        """Parse command-line arguments, support key=value format for nested fields"""
        cli_args = {}

        for arg in sys.argv[1:]:
            # Skip configuration file arguments
            if arg.endswith(('.yaml', '.yml')):
                continue

            # Process key=value format
            if '=' in arg and not arg.startswith('-'):
                parts = arg.split('=', 1)
                if len(parts) == 2 and parts[0].strip():
                    # Ensure it's not --key=value format
                    if not parts[0].startswith('-'):
                        key = parts[0].strip()
                        value_str = parts[1].strip()

                        # Try to auto-convert type
                        value = self._parse_string_value(value_str)

                        # Process nested fields
                        if '.' in key:
                            self._set_nested_value(cli_args, key, value)
                        else:
                            cli_args[key] = value

        return cli_args

    def _safe_merge_configs(self,
                            yaml_config: Dict[str, Any],
                            cli_config: Dict[str, Any],
                            additional_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Safely merge all configuration sources, support nested fields
        Priority: command line > code input > YAML > default values
        """
        merged = {}

        def _merge_dict(target: Dict, source: Dict, path: str = ""):
            """Recursively merge dictionaries, support nested fields"""
            for key, value in source.items():
                full_key = f"{path}.{key}" if path else key

                if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                    # Recursively merge nested dictionaries
                    _merge_dict(target[key], value, full_key)
                else:
                    # Set value directly
                    if '.' in full_key:
                        self._set_nested_value(target, full_key, value)
                    else:
                        target[key] = value

        # 1. First apply YAML configuration
        if yaml_config:
            _merge_dict(merged, yaml_config)

        # 2. Then apply code input configuration
        if additional_config:
            _merge_dict(merged, additional_config)

        # 3. Finally apply command-line configuration
        if cli_config and self.allow_cli_override:
            for key, value in cli_config.items():
                # Only allow overriding existing fields
                can_override = key in merged or self._is_field_defined(
                    key) or key in self._dynamic_fields or self._is_nested_field_defined(key)
                if can_override:

                    if '.' in key:
                        self._set_nested_value(merged, key, value)
                    else:
                        merged[key] = value
                else:
                    _print(f"[WARNING] Ignoring undefined command-line argument: {key}")

        return merged

    def _is_field_defined(self, field_name: str) -> bool:
        """Check if field is already defined"""
        return (field_name in self._defined_fields_info or
                field_name in self._dynamic_fields)

    def _is_nested_field_defined(self, field_path: str) -> bool:
        """Check if nested field is defined or its parent path exists"""
        if '.' not in field_path:
            return False

        parts = field_path.split('.')
        # Check all prefix paths
        for i in range(1, len(parts)):
            prefix = '.'.join(parts[:i])
            if self._is_field_defined(prefix):
                return True

        return False

    def _check_required_fields(self, config_data: Dict[str, Any]):
        """Check if required fields are provided, support nested fields"""
        # Collect all required fields
        required_fields = set()

        def _collect_required_fields(cls: Type[BaseArguments], prefix: str = ""):
            if hasattr(cls, '_dynamic_fields_info'):
                for field_name, field_info in cls._dynamic_fields_info.items():
                    if field_info.required:
                        full_name = f"{prefix}.{field_name}" if prefix else field_name
                        required_fields.add(full_name)

            # Recursively collect required fields from parent classes
            for base_cls in cls.__bases__:
                if issubclass(base_cls, BaseArguments):
                    _collect_required_fields(base_cls, prefix)

        _collect_required_fields(self.config_class)

        # Check which required fields are missing
        missing_fields = []
        for field_name in required_fields:
            if '.' in field_name:
                # Nested field
                value = self._get_nested_value(config_data, field_name)
                if value is None:
                    missing_fields.append(field_name)
            else:
                # Regular field
                if field_name not in config_data:
                    missing_fields.append(field_name)

        if missing_fields:
            raise ConfigValidationError(
                f"Missing required fields: {', '.join(missing_fields)}\n"
                f"Please provide values for these fields in the configuration file"
            )

    def _infer_yaml_value_type(self, value: Any) -> Optional[Type]:
        """
        Infer YAML value type
        Only supports basic types and lists
        """
        if value is None:
            return None

        if isinstance(value, bool):
            return bool
        elif isinstance(value, int):
            return int
        elif isinstance(value, float):
            return float
        elif isinstance(value, str):
            return str
        elif isinstance(value, list):
            # Infer list element type
            if value:
                elem_type = type(value[0])
                for elem in value[1:]:
                    if not isinstance(elem, elem_type):
                        return list  # Inconsistent type, return generic list
                return List[elem_type]
            return list
        elif isinstance(value, dict):
            return dict
        else:
            return None

    def _parse_string_value(self, value_str: str) -> Any:
        """Parse string value to appropriate type"""
        if not value_str:
            return value_str

        # Try to convert to boolean
        lower_value = value_str.lower()
        if lower_value in ('true', 'false', 'yes', 'no', 'on', 'off'):
            return lower_value in ('true', 'yes', 'on')

        # Try to convert to integer
        if value_str.isdigit():
            return int(value_str)

        # Try to convert to float
        try:
            return float(value_str)
        except ValueError:
            pass

        # Try to convert to list (comma-separated)
        if ',' in value_str:
            parts = [part.strip() for part in value_str.split(',')]
            converted_parts = []
            for part in parts:
                converted_parts.append(self._parse_string_value(part))
            return converted_parts

        return value_str

    def _load_yaml_config(self) -> Dict[str, Any]:
        """Load YAML configuration file"""
        if not self.config_file_path:
            return {}

        try:
            with open(self.config_file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            _print(f"[INFO] Loaded configuration file: {self.config_file_path}")
            return config or {}
        except yaml.YAMLError as e:
            raise ConfigValidationError(f"YAML parsing failed: {self.config_file_path} - {e}") from e
        except Exception as e:
            raise ConfigValidationError(f"Failed to load configuration file: {self.config_file_path} - {e}") from e

    def _validate_config_object(self, config_obj: BaseArguments, merged_config: Dict[str, Any]):
        """Validate configuration object"""
        for field_name in merged_config:
            if '.' in field_name:
                # Nested field, needs special handling
                parts = field_name.split('.')
                current = config_obj
                for part in parts:
                    if hasattr(current, part):
                        current = getattr(current, part)
                    elif isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        _print(f"[WARNING] Nested field {field_name} not set to configuration object")
                        break
            else:
                if not hasattr(config_obj, field_name):
                    _print(f"[WARNING] Field {field_name} not set to configuration object")

    def _get_config_dict(self, config_obj: BaseArguments) -> Dict[str, Any]:
        """Get dictionary representation of configuration object"""
        config_dict = {}

        for attr_name in dir(config_obj):
            if attr_name.startswith('_'):
                continue

            attr_value = getattr(config_obj, attr_name)
            if callable(attr_value):
                continue

            config_dict[attr_name] = attr_value

        return config_dict


def _print(msg: str):
    if int(os.environ.get('RANK', '0')) == 0:
        print(msg)
