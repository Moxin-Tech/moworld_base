# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Thread-safe configuration loading utilities.

This module provides utilities for safely loading HuggingFace model configurations
in multi-threaded environments, preventing race conditions that can occur when
multiple threads try to download and cache the same model simultaneously.
"""

import hashlib
import os
import time
from pathlib import Path
from typing import Union

import filelock
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig


def safe_load_config_with_retry(
        path: Union[str, Path], trust_remote_code: bool = False, max_retries: int = 3, base_delay: float = 1.0, **kwargs
) -> PretrainedConfig:
    """
    Thread-safe and process-safe configuration loading with retry logic.

    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            # Use file locking for process-safe access
            # Create a lock file based on the path hash to avoid conflicts
            path_hash = hashlib.md5(str(path).encode()).hexdigest()

            # Allow override of lock directory via environment variable
            # This is useful for multi-node setups where a shared lock directory is needed
            lock_dir = os.getenv("MEGATRON_CONFIG_LOCK_DIR")
            if lock_dir:
                lock_file = Path(lock_dir) / f".megatron_config_lock_{path_hash}"
            else:
                lock_file = Path.home() / ".cache" / "huggingface" / f".megatron_config_lock_{path_hash}"

            lock_file.parent.mkdir(parents=True, exist_ok=True)

            with filelock.FileLock(str(lock_file) + ".lock", timeout=60):
                return AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code, **kwargs)

        except Exception as e:
            last_exception = e

            # Don't retry on certain types of errors
            error_msg = str(e).lower()
            non_retryable_phrases = [
                "does not appear to have a file named config.json",
                "repository not found",
                "entry not found",
                "401 client error",
                "403 client error",
            ]
            if any(phrase in error_msg for phrase in non_retryable_phrases):
                # Model doesn't exist or access denied, no point retrying
                raise ValueError(
                    f"Failed to load configuration from {path}. "
                    f"Ensure the path is valid and contains a config.json file. "
                    f"Error: {e}"
                ) from e

            if attempt < max_retries:
                # Exponential backoff with jitter
                delay = base_delay * (2 ** attempt) + (time.time() % 1) * 0.1
                time.sleep(delay)
            else:
                # Final attempt failed
                break

    # All retries exhausted
    raise ValueError(
        f"Failed to load configuration from {path} after {max_retries + 1} attempts. "
        f"This might be due to network issues or concurrent access conflicts. "
        f"Last error: {last_exception}"
    ) from last_exception
