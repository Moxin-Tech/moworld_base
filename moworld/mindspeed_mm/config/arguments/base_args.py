from typing import Any, Dict, Optional, Type, Union, get_origin, get_args, ClassVar
from dataclasses import dataclass
from pydantic import BaseModel


@dataclass
class FieldInfo:
    """Field information data class"""
    value_type: Optional[Type] = None
    default: Any = None
    description: str = ""
    required: bool = False


class BaseArguments(BaseModel):
    """
    Class-level dynamic field support
    """

    class Config:
        extra = 'allow'
        validate_assignment = True

    def __init_subclass__(cls, **kwargs):
        """Called when a subclass is created, supports field inheritance, ensures each subclass has its own field info"""
        super().__init_subclass__(**kwargs)

        # Initialize current class's field information
        cls._dynamic_fields_info = {}

        # Only process parent classes that inherit from BaseArguments
        for parent_cls in cls.__bases__:
            if parent_cls is BaseArguments or not issubclass(parent_cls, BaseArguments):
                continue
            if hasattr(parent_cls, '_dynamic_fields_info'):
                # Deep copy parent class's field information
                for field_name, field_info in parent_cls._dynamic_fields_info.items():
                    if field_name not in cls._dynamic_fields_info:
                        cls._dynamic_fields_info[field_name] = field_info

    def __init__(self, **data):
        """
        Initialize configuration object

        Check required fields and type validation during initialization
        """
        # Check required fields
        self._check_required_dynamic_fields_on_init(data)

        # Check types of input data
        self._check_init_data_types(data)

        # Call parent class initialization
        super().__init__(**data)

        # Set default values for all registered fields
        for name, field_info in self._dynamic_fields_info.items():
            if not hasattr(self, name):  # If not set in initialization data
                setattr(self, name, field_info.default)

    @classmethod
    def register_field(cls,
                       name: str,
                       value_type: Optional[Type] = None,
                       default: Any = None,
                       description: str = "",
                       required: bool = False) -> None:
        """
        Register dynamic fields at class level

        Args:
            name: Field name
            value_type: Field type, None means no type validation
            default: Default value
            description: Field description
            required: Whether the field is required

        Raises:
            AttributeError: Raised when field already exists
            TypeError: Raised when default value doesn't match value_type
        """
        # Check if field already exists
        if hasattr(cls, name) and name not in cls.__fields__:
            raise AttributeError(f"Field '{name}' already exists, cannot register again")

        # Validate default value type
        if value_type is not None and default is not None:
            # Use private method for type validation
            # We call _validate_type via instance
            # Create a temporary instance for validation
            try:
                # Create temporary instance for validation
                temp_instance = cls.__new__(cls)
                temp_instance._validate_type(default, value_type, f"{name}.default")
            except TypeError as e:
                raise TypeError(
                    f"Default value type error for field '{name}': {default} is not of type {value_type}\nError details: {e}") from e

        # Store field information in class attribute
        cls._dynamic_fields_info[name] = FieldInfo(
            value_type=value_type,
            default=default,
            description=description,
            required=required
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert instance to dictionary

        Args:
            exclude_none: Whether to exclude fields with None value
            exclude_unset: Whether to exclude unset fields (using default values)
            exclude_defaults: Whether to exclude fields with default values
            exclude_private: Whether to exclude private attributes starting with '_'
            by_alias: Whether to use field aliases

        Returns:
            Dict[str, Any]: Dictionary containing all fields
        """
        # 1. Get Pydantic model's dictionary representation
        result = self.model_dump()

        # 2. Add dynamically registered fields
        for field_name in self._dynamic_fields_info.keys():
            value = getattr(self, field_name)
            result[field_name] = value

        return result

    def to_str(self) -> str:
        """
        Convert instance to string

        Returns:
            str: String representation containing all fields
        """
        lines = []

        def _add_config_lines(cfg: dict, indent: int = 0):
            for key in cfg.to_dict():
                prefix = "  " * indent
                value = getattr(cfg, key)  # Use getattr to get attribute value

                if isinstance(value, BaseArguments):
                    # BaseArguments object, handle recursively
                    lines.append(f"{prefix}{key}:")
                    _add_config_lines(value, indent + 1)
                else:
                    # Other types (including list, dict, etc.) output directly
                    lines.append(f"{prefix}{key}: {value}")

        _add_config_lines(self)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_str()

    def __repr__(self) -> str:
        return self.to_str()

    def _check_required_dynamic_fields_on_init(self, init_data: Dict[str, Any]) -> None:
        """
        Check if required fields are set during initialization

        Args:
            init_data: Initialization data

        Raises:
            ValueError: If required fields are not set
        """
        # Get all required fields
        required_fields = {
            name for name, info in self._dynamic_fields_info.items()
            if info.required
        }

        # Check which required fields are not provided in initialization data
        missing_fields = []
        for field_name in required_fields:
            if field_name not in init_data:
                missing_fields.append(field_name)

        # If there are missing required fields, raise exception
        if missing_fields:
            raise ValueError(
                f"Missing required fields: {', '.join(missing_fields)}\n"
                f"Please provide values for these fields during initialization"
            )

    def _check_init_data_types(self, init_data: Dict[str, Any]) -> None:
        """
        Check types of fields in initialization data

        Args:
            init_data: Initialization data

        Raises:
            TypeError: If field types do not match
        """
        for field_name, value in init_data.items():
            # Check if it's a registered dynamic field
            if field_name in self._dynamic_fields_info:
                field_info = self._dynamic_fields_info[field_name]
                expected_type = field_info.value_type

                # Perform type validation when field type is specified
                if expected_type is not None and value is not None:
                    self._validate_type(value, expected_type, field_name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Override attribute setting to support dynamic field type validation"""
        # Check if it's a registered field
        if name in self._dynamic_fields_info:
            field_info = self._dynamic_fields_info[name]
            expected_type = field_info.value_type

            # Perform type validation when field type is specified
            if expected_type is not None and value is not None:
                self._validate_type(value, expected_type, name)

        # Call parent class
        super().__setattr__(name, value)

    def _validate_type(self, value: Any, expected_type: Type, field_name: str) -> None:
        """
        Type validation
        Supports: Any, int, float, bool, str, list, List[T], Optional[T], Union[T1, T2, ...]
        """
        # 0. Handle Any type
        if expected_type is Any:
            return

        # 1. Handle Union
        origin = get_origin(expected_type)
        if origin is Union:
            args = get_args(expected_type)

            # Union contains Any
            if Any in args:
                return

            # Handle Optional[T] (Union[T, NoneType])
            if type(None) in args and value is None:
                return

            # Try to match each type in Union
            for t in args:
                if t is not type(None):
                    try:
                        self._validate_type(value, t, field_name)
                        return
                    except TypeError:
                        continue

            # No matching type found
            type_str = self._type_to_str(expected_type)
            raise TypeError(f"Field '{field_name}': must be of type {type_str}")

        # 2. Handle None
        if value is None:
            raise TypeError(f"Field '{field_name}': cannot be None")

        # 3. Basic type checking
        if expected_type is int:
            if not isinstance(value, int):
                # Check if it's a numeric string
                if isinstance(value, str) and value.isdigit():
                    return
                raise TypeError(f"Field '{field_name}': must be integer type, got {type(value).__name__}")
            return

        elif expected_type is float:
            if not isinstance(value, (int, float)):
                if isinstance(value, str):
                    try:
                        float(value)
                        return
                    except ValueError:
                        pass
                raise TypeError(f"Field '{field_name}': must be numeric type, got {type(value).__name__}")
            return

        elif expected_type is bool:
            if not isinstance(value, bool):
                if isinstance(value, str) and value.lower() in ('true', 'false', 'yes', 'no', '1', '0'):
                    return
                raise TypeError(f"Field '{field_name}': must be boolean type, got {type(value).__name__}")
            return

        elif expected_type is str:
            if not isinstance(value, str):
                raise TypeError(f"Field '{field_name}': must be string type, got {type(value).__name__}")
            return

        elif expected_type is list:
            if not isinstance(value, list):
                raise TypeError(f"Field '{field_name}': must be list type, got {type(value).__name__}")
            return

        elif expected_type is dict:
            if not isinstance(value, dict):
                raise TypeError(f"Field '{field_name}': must be dictionary type, got {type(value).__name__}")
            return

        # 4. Handle generic List[T]
        if origin is list:
            if not isinstance(value, list):
                raise TypeError(f"Field '{field_name}': must be list type, got {type(value).__name__}")

            args = get_args(expected_type)
            if args:  # Check list element types
                item_type = args[0]
                if item_type is not Any:  # Only check if element type is not Any
                    for i, item in enumerate(value):
                        self._validate_type(item, item_type, f"{field_name}[{i}]")
            return

        # 5. Handle generic Dict[K, V]
        if origin is dict:
            if not isinstance(value, dict):
                raise TypeError(f"Field '{field_name}': must be dictionary type, got {type(value).__name__}")

            args = get_args(expected_type)
            if len(args) >= 2:  # Check key-value types
                key_type, val_type = args[0], args[1]
                if key_type is not Any or val_type is not Any:
                    for k, v in value.items():
                        if key_type is not Any:
                            self._validate_type(k, key_type, f"{field_name}.key")
                        if val_type is not Any:
                            self._validate_type(v, val_type, f"{field_name}['{k}']")
            return

        # 6. Unsupported type
        raise TypeError(
            f"Field '{field_name}' type {self._type_to_str(expected_type)} is not supported.\n"
            f"Supported types: int, float, bool, str, list, dict, List[T], Dict[K, V], Optional[T], Union[T1, T2, ...], Any"
        )

    def _type_to_str(self, t: Type) -> str:
        """Convert type to readable string"""
        if t is Any:
            return "Any"

        # Handle basic types
        if t is int:
            return "int"
        elif t is float:
            return "float"
        elif t is bool:
            return "bool"
        elif t is str:
            return "str"
        elif t is list:
            return "list"
        elif t is dict:
            return "dict"
        elif t is type(None):
            return "None"

        # Handle generic types
        origin = get_origin(t)
        if origin is Union:
            args = get_args(t)
            args_str = []
            for arg in args:
                if arg is not type(None) or len(args) > 1:  # Don't display None alone
                    args_str.append(self._type_to_str(arg))
            if len(args_str) == 1 and type(None) in args:
                return f"Optional[{args_str[0]}]"
            return " | ".join(args_str)
        elif origin is list:
            args = get_args(t)
            if args:
                return f"List[{self._type_to_str(args[0])}]"
            return "List"
        elif origin is dict:
            args = get_args(t)
            if len(args) >= 2:
                return f"Dict[{self._type_to_str(args[0])}, {self._type_to_str(args[1])}]"
            return "Dict"

        # Handle special types
        if hasattr(t, '__name__'):
            return t.__name__

        return str(t)
