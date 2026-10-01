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

import re
from typing import List, Optional

from bridge.models.conversion.param_mapping import MegatronParamMapping


class MegatronMappingRegistry:

    def _convert_pattern_to_regex(self, pattern: str) -> str:
        """Convert a pattern with wildcards to regex pattern.

        Args:
            pattern: Pattern string with * and ** wildcards

        Returns:
            Regex pattern string

        Note:
            ** must be processed before * to avoid conflicts.
            ** becomes (.*) - matches any characters including dots
            * becomes (\\d+) - matches digits only for layer indices
        """
        # Escape the pattern first
        regex_pattern = re.escape(pattern)

        # Process ** before * to avoid conflicts
        # Replace \*\* with (.*)
        regex_pattern = regex_pattern.replace(r"\*\*", r"(.*)")

        # Replace remaining \* with (\d+)
        regex_pattern = regex_pattern.replace(r"\*", r"(\d+)")

        return regex_pattern

    def __init__(self, *mappings: MegatronParamMapping):
        """
        Initialize MegatronMappingRegistry with weight mappings.

        Args:
            *mappings: MegatronParamMapping objects
        """
        self.mappings = list(mappings)

        # Pre-compile patterns for efficiency
        self._compiled_patterns = []
        self._reverse_patterns = []  # For hf_param -> megatron lookups

        for mapping in mappings:
            # Compile source patterns
            if "*" in mapping.megatron_param:
                # Convert glob pattern to regex with support for * and **
                pattern = self._convert_pattern_to_regex(mapping.megatron_param)
                self._compiled_patterns.append((re.compile(f"^{pattern}$"), mapping))
            else:
                self._compiled_patterns.append((None, mapping))

            # Compile destination patterns for reverse lookups
            if isinstance(mapping.hf_param, str):
                if "*" in mapping.hf_param:
                    pattern = self._convert_pattern_to_regex(mapping.hf_param)
                    self._reverse_patterns.append((re.compile(f"^{pattern}$"), mapping))
                else:
                    self._reverse_patterns.append((None, mapping))
            else:
                # For dict destinations, compile patterns for each value
                reverse_dict_patterns = {}
                for key, hf_pattern in mapping.hf_param.items():
                    if "*" in hf_pattern:
                        pattern = self._convert_pattern_to_regex(hf_pattern)
                        reverse_dict_patterns[key] = re.compile(f"^{pattern}$")
                    else:
                        reverse_dict_patterns[key] = None
                self._reverse_patterns.append((reverse_dict_patterns, mapping))

    def megatron_to_hf_lookup(self, megatron_param_name: str) -> Optional[MegatronParamMapping]:
        """
        Get mapping for a Megatron parameter name.

        This method performs efficient lookups by first checking for exact matches,
        then falling back to pattern matching using pre-compiled regex patterns.
        When a pattern match is found, wildcards are automatically resolved.

        Args:
            megatron_param_name: Megatron parameter name to look up
                Example: "decoder.layers.0.self_attention.linear_qkv.weight"

        Returns:
            MegatronParamMapping: Bridge instance with resolved wildcards, or None
                if no matching mapping is found. The returned bridge will have
                all wildcards replaced with actual values.
        """
        for pattern, mapping in self._compiled_patterns:
            if pattern is None:
                # Direct match
                if mapping.megatron_param == megatron_param_name:
                    return mapping
            else:
                # Pattern match
                match = pattern.match(megatron_param_name)
                if match:
                    # Return resolved mapping with wildcards replaced
                    return mapping.resolve(match.groups())
        return None

    def __len__(self) -> int:
        """Return number of mappings."""
        return len(self.mappings)

    def __iter__(self):
        """Iterate over mappings."""
        return iter(self.mappings)

    def __repr__(self) -> str:
        """String representation of MegatronMappingRegistry."""
        return f"MegatronMappingRegistry({len(self.mappings)} mappings)"
