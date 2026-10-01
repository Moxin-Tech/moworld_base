# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import fnmatch
import json
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    List,
    Optional,
    Pattern,
    Tuple,
    Union,
    overload,
)

import torch


class StateDict(Mapping[str, torch.Tensor]):
    """
    A state dict accessor that provides a unified interface for querying model
    checkpoints.

    """

    source: "StateSource"

    def __init__(self, source: Dict[str, torch.Tensor] | "StateSource"):
        """
        Initializes the StateDict query accessor.

        Args:
            source: The source of the tensor data. This can be a standard
                Python dictionary mapping tensor names to `torch.Tensor` objects,
                or an instance of a `StateSource` subclass (e.g.,
                `SafetensorsStateSource`) for more advanced, out-of-memory
                access.
        """
        if isinstance(source, dict):
            source = DictStateSource(source)

        if not isinstance(source, StateSource):
            raise TypeError(f"StateDict source must be a dict or a StateSource, got {type(source)}")

        self.source = source

    def _get_all_keys(self) -> List[str]:
        """
        Get all available tensor keys from the underlying source.
        """
        return self.source.get_all_keys()

    def _load_tensors(self, keys_to_load: List[str]) -> Dict[str, torch.Tensor]:
        """
        Load specified tensors from the underlying source.
        """
        return self.source.load_tensors(keys_to_load)

    def _match_keys(self, pattern: Union[str, Pattern]) -> List[str]:
        """Match keys against a glob pattern or regex."""
        all_keys = self._get_all_keys()

        if isinstance(pattern, Pattern):
            # Regex pattern
            return [k for k in all_keys if pattern.search(k)]
        elif "*" in pattern or "?" in pattern or "[" in pattern:
            # Glob pattern
            return [k for k in all_keys if fnmatch.fnmatch(k, pattern)]
        else:
            # Exact match
            return [pattern] if pattern in all_keys else []

    @overload
    def __getitem__(self, key: str) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        ...

    @overload
    def __getitem__(self, key: List[str]) -> Dict[str, torch.Tensor]:
        ...

    @overload
    def __getitem__(self, key: Pattern) -> Dict[str, torch.Tensor]:
        ...

    def __getitem__(self, key: Union[str, List[str], Pattern]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Accesses state dict entries using various key types.

        """
        if isinstance(key, Pattern):
            matched_keys = self._match_keys(key)
            if not matched_keys:
                raise KeyError(f"No keys match regex pattern: {key.pattern}")
            return self._load_tensors(matched_keys)
        elif isinstance(key, str):
            if "*" in key or "?" in key or "[" in key:
                matched_keys = self._match_keys(key)
                if not matched_keys:
                    raise KeyError(f"No keys match pattern: {key}")
                return self._load_tensors(matched_keys)
            else:
                if key not in self._get_all_keys():
                    raise KeyError(f"Key not found: {key}")
                return self._load_tensors([key])[key]
        elif isinstance(key, list):
            all_keys_set = set(self._get_all_keys())
            missing_keys = [k for k in key if k not in all_keys_set]
            if missing_keys:
                raise KeyError(f"Keys not found: {missing_keys}")
            return self._load_tensors(key)
        else:
            raise TypeError(f"Key must be str, list of str, or compiled regex, got {type(key)}")

    def regex(self, pattern: str) -> Dict[str, torch.Tensor]:
        """
        Queries the state dict with a regular expression pattern.

        """
        return self[re.compile(pattern)]

    def glob(self, pattern: str) -> Dict[str, torch.Tensor]:
        """
        Queries the state dict with a glob pattern.

        """
        return self[pattern]

    def __call__(self) -> Dict[str, torch.Tensor]:
        """
        Loads and returns the entire state dict as a dictionary.

        """
        all_keys = self._get_all_keys()
        return self._load_tensors(all_keys)

    def keys(self) -> List[str]:
        """Get all state dict keys."""
        return self._get_all_keys()

    def items(self) -> List[tuple]:
        """Get all state dict items."""
        return list(self().items())

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the state dict."""
        return key in self._get_all_keys()

    def __repr__(self) -> str:
        """String representation."""
        try:
            num_params = len(self)
            return f"<StateDict with {num_params} entries>"
        except Exception:
            return "<StateDict (not accessible)>"

    def get(self, key: str, default=None) -> Optional[torch.Tensor]:
        """
        Gets a tensor from the state dict.
        Returns `default` if the key is not found.
        Note: This method is for single key lookup and does not support patterns.
        """
        if key in self._get_all_keys():
            return self._load_tensors([key])[key]
        return default

    def __iter__(self) -> Iterable[str]:
        """Iterate over state dict keys."""
        return iter(self.keys())

    def __len__(self) -> int:
        """Get number of entries in the state dict."""
        return len(self.keys())


class StateSource(ABC, Mapping[str, torch.Tensor]):
    """
    Abstract base class for a source of model state.

    This class defines a standard interface for `StateDict` to access tensor
    data, abstracting away the details of how and where the data is stored.
    Subclasses can implement loading from different storage backends, such as
    in-memory dictionaries or files on disk. This allows `StateDict` to handle
    various checkpoint formats in a uniform way.
    """

    @abstractmethod
    def get_all_keys(self) -> List[str]:
        """Returns a list of all available tensor keys in the source."""
        pass

    @abstractmethod
    def load_tensors(self, keys: List[str]) -> Dict[str, torch.Tensor]:
        """Loads the specified tensors from the source."""
        pass

    def __getitem__(self, key: str) -> torch.Tensor:
        """Loads a single tensor by key."""
        tensors = self.load_tensors([key])
        if key not in tensors:
            raise KeyError(f"Key not found in source: {key}")
        return tensors[key]

    def __iter__(self) -> Iterable[str]:
        """Iterates over all tensor keys."""
        return iter(self.get_all_keys())

    def __len__(self) -> int:
        """Returns the total number of tensors in the source."""
        return len(self.get_all_keys())


class DictStateSource(StateSource):
    """
    A state source backed by an in-memory Python dictionary.

    This is the simplest `StateSource` implementation. It's used when the entire
    model state dict is already loaded into a dictionary in memory.

    Args:
        state_dict: A dictionary mapping tensor names (str) to `torch.Tensor` objects.
    """

    def __init__(self, state_dict: Dict[str, torch.Tensor]):
        self._dict = state_dict
        self._keys_cache: Optional[List[str]] = None

    def get_all_keys(self) -> List[str]:
        if self._keys_cache is None:
            self._keys_cache = sorted(list(self._dict.keys()))
        return self._keys_cache

    def load_tensors(self, keys: List[str]) -> Dict[str, torch.Tensor]:
        return {key: self._dict[key] for key in keys if key in self._dict}


class SafeTensorsStateSource(StateSource):
    """
    A state source backed by a directory of .safetensors files.

    """

    def __init__(self, path: Union[str, Path]):
        self.model_name_or_path = path
        self._resolved_path_cache: Optional[Path] = None
        self._keys_cache: Optional[List[str]] = None
        self._key_to_filename_map_cache: Optional[Dict[str, str]] = None

    @property
    def path(self) -> Path:
        """
        The local path to the checkpoint files.
        If the initial path is a Hugging Face Hub model ID, this property
        will handle downloading the necessary files and return the local
        cache path.
        """
        if self._resolved_path_cache is None:
            self._resolved_path_cache = self._resolve_path(self.model_name_or_path)
        return self._resolved_path_cache

    @staticmethod
    def _resolve_path(model_name_or_path: Union[str, Path]) -> Path:
        """
        Resolves a model name or path to a local directory.
        If the path is not a local directory, it is treated as a Hugging
        Face Hub model ID, and the corresponding files are downloaded.
        """
        local_path = Path(model_name_or_path)
        if local_path.is_dir():
            return local_path

        try:
            from huggingface_hub import snapshot_download
            from huggingface_hub.utils import HfHubHTTPError

            # Not a local directory, so we assume it's a model ID
            # on the Hugging Face Hub.
            return Path(
                snapshot_download(
                    repo_id=str(model_name_or_path),
                    allow_patterns=[
                        "*.safetensors",
                        "model.safetensors.index.json",
                    ],
                    # Ignore other large files.
                    ignore_patterns=["*.bin", "*.pt", "*.pth"],
                )
            )
        except (ImportError, HfHubHTTPError, ValueError):
            # If huggingface_hub is not installed, or if it's not a
            # valid model ID, we return the original path and let the
            # subsequent logic handle the file not found error.
            return local_path

    @property
    def key_to_filename_map(self) -> Dict[str, str]:
        """
        Provides a mapping from tensor keys to the safetensor filename they
        are stored in.

        This map is constructed either from `model.safetensors.index.json` if
        it exists, or by scanning all `.safetensors` files in the directory.
        The result is cached for efficiency.
        """
        if self._key_to_filename_map_cache is not None:
            return self._key_to_filename_map_cache

        # First, try to load from the index file.
        key_map = self._cached_get_key_to_filename_map(self.path)
        if key_map:
            self._key_to_filename_map_cache = key_map
            return key_map

        # If no index, scan the directory.
        import os
        from glob import glob as file_glob

        from safetensors import safe_open

        key_map = {}
        safetensor_files = file_glob(str(self.path / "*.safetensors"))
        for file_path in safetensor_files:
            filename = os.path.basename(file_path)
            try:
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key in key_map:
                            # This is an issue. Same key in multiple files, and no index.
                            # How to resolve ambiguity? Let's just warn and overwrite. Last one wins.
                            print(
                                f"Warning: duplicate key '{key}' found in '{filename}' and '{key_map[key]}'. Using '{filename}'."
                            )
                        key_map[key] = filename
            except Exception as e:
                # Can be not a safetensor file, etc.
                print(f"Warning: could not open {filename} as a safetensors file: {e}")

        self._key_to_filename_map_cache = key_map
        return key_map

    def get_all_keys(self) -> List[str]:
        if self._keys_cache is not None:
            return self._keys_cache

        from glob import glob as file_glob

        from safetensors import safe_open

        all_keys = set()
        key_to_filename_map = self.key_to_filename_map
        if key_to_filename_map:
            all_keys.update(key_to_filename_map.keys())

        if not all_keys:
            safetensor_files = file_glob(str(self.path / "*.safetensors"))
            if not safetensor_files and not key_to_filename_map:
                raise FileNotFoundError(f"No .safetensors files or index found in {self.model_name_or_path}")
            for safetensor_file in safetensor_files:
                with safe_open(safetensor_file, framework="pt", device="cpu") as f:
                    all_keys.update(f.keys())

        self._keys_cache = sorted(list(all_keys))
        return self._keys_cache

    def load_tensors(self, keys_to_load: List[str]) -> Dict[str, torch.Tensor]:
        if not keys_to_load:
            return {}

        from glob import glob as file_glob

        from safetensors import safe_open

        loaded_tensors = {}
        remaining_keys = set(keys_to_load)
        key_to_filename_map = self.key_to_filename_map

        if key_to_filename_map:
            file_to_keys_map = defaultdict(list)
            for key in list(remaining_keys):
                if key in key_to_filename_map:
                    filename = key_to_filename_map[key]
                    file_to_keys_map[filename].append(key)

            for filename, keys_in_file in file_to_keys_map.items():
                file_path = self.path / filename
                if file_path.exists():
                    with safe_open(file_path, framework="pt", device="cpu") as f:
                        for key in keys_in_file:
                            if key in f.keys():
                                loaded_tensors[key] = f.get_tensor(key)
                                remaining_keys.discard(key)

        if remaining_keys:
            safetensor_files = file_glob(str(self.path / "*.safetensors"))
            if not safetensor_files and not key_to_filename_map and not loaded_tensors:
                raise FileNotFoundError(
                    f"No .safetensors files found in {self.model_name_or_path} to load keys: {remaining_keys}"
                )
            for safetensor_file_path in safetensor_files:
                if not remaining_keys:
                    break
                with safe_open(safetensor_file_path, framework="pt", device="cpu") as f:
                    current_file_keys = f.keys()
                    for key in list(remaining_keys):
                        if key in current_file_keys:
                            loaded_tensors[key] = f.get_tensor(key)
                            remaining_keys.remove(key)

        if remaining_keys:
            raise KeyError(f"Keys not found in safetensors from {self.model_name_or_path}: {remaining_keys}")

        return loaded_tensors

    def _get_key_to_filename_map(self) -> Optional[Dict[str, str]]:
        return self._cached_get_key_to_filename_map(self.path)

    @staticmethod
    @lru_cache(maxsize=None)
    def _cached_get_key_to_filename_map(model_name_or_path: Union[str, Path]) -> Optional[Dict[str, str]]:
        """Static, cached method to get the key-to-filename map."""
        index_file = Path(model_name_or_path) / "model.safetensors.index.json"
        if index_file.exists():
            with open(index_file, "r") as f:
                try:
                    index_data = json.load(f)
                    if "weight_map" in index_data and isinstance(index_data["weight_map"], dict):
                        return index_data["weight_map"]
                except json.JSONDecodeError:
                    return None
        return None
