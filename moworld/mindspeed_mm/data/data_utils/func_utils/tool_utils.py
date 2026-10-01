# Copyright 2025 the LlamaFactory team.
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

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, NamedTuple, Union

from typing_extensions import override


class FunctionCall(NamedTuple):
    name: str
    arguments: str


@dataclass
class ToolUtils(ABC):
    """Base class for tool utilities."""

    @staticmethod
    @abstractmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        r"""Generate the system message describing all the available tools."""
        ...

    @staticmethod
    @abstractmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        r"""Generate the assistant message including all the tool calls."""
        ...

    @staticmethod
    @abstractmethod
    def tool_extractor(content: str) -> Union[str, list["FunctionCall"]]:
        r"""Extract all the function calls from the assistant message.

        It should be an inverse function of `function_formatter`.
        """
        ...


class MistralToolUtils(ToolUtils):
    r"""Mistral v0.3 tool using template."""

    @override
    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        wrapped_tools = []
        for tool in tools:
            wrapped_tools.append(tool if tool.get("type") == "function" else {"type": "function", "function": tool})

        return "[AVAILABLE_TOOLS] " + json.dumps(wrapped_tools, ensure_ascii=False) + "[/AVAILABLE_TOOLS]"

    @override
    @staticmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        return json.dumps(
            [{"name": name, "arguments": json.loads(arguments)} for name, arguments in functions], ensure_ascii=False
        )

    @override
    @staticmethod
    def tool_extractor(content: str) -> Union[str, list["FunctionCall"]]:
        try:
            tools = json.loads(content.strip())
        except json.JSONDecodeError:
            return content

        tools = [tools] if not isinstance(tools, list) else tools
        try:
            return [FunctionCall(tool["name"], json.dumps(tool["arguments"], ensure_ascii=False)) for tool in tools]
        except KeyError:
            return content


TOOLS = {
    "mistral": MistralToolUtils(),
}


def get_tool_utils(name: str) -> "ToolUtils":
    tool_utils = TOOLS.get(name, None)
    if tool_utils is None:
        raise ValueError(f"Tool utils `{name}` not found.")

    return tool_utils
