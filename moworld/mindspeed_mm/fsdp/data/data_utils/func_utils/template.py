# Copyright 2024 the LlamaFactory team.
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

import re
from dataclasses import dataclass
from copy import deepcopy

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from typing_extensions import override

from transformers import PreTrainedTokenizer

from .convert import Role
from .formatters import EmptyFormatter, Formatter, StringFormatter
from .formatters import SLOTS
from .log import get_logger
from .mm_plugin import get_mm_plugin

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from .formatters import SLOTS
    from .mm_plugin import BasePlugin

logger = get_logger(__name__)


@dataclass
class Template:
    format_user: "Formatter"
    format_assistant: "Formatter"
    format_system: "Formatter"
    format_observation: "Formatter"
    format_prefix: "Formatter"
    format_tool: "Formatter"
    default_system: str
    stop_words: List[str]
    efficient_eos: bool
    replace_eos: bool
    enable_thinking: Optional[bool]
    thought_words: tuple[str, str]
    mm_plugin: "BasePlugin"

    def encode_oneturn(
            self,
            tokenizer: "PreTrainedTokenizer",
            messages: Sequence[Dict[str, str]],
            system: Optional[str] = None,
            tools: List[str] = None
    ) -> Tuple[List[int], List[int]]:
        r"""
        Returns a single pair of token ids representing prompt and response respectively.
        """
        encoded_messages = self._encode(tokenizer, messages, system, tools)
        prompt_ids = []
        for encoded_ids in encoded_messages[:-1]:
            prompt_ids += encoded_ids

        answer_ids = encoded_messages[-1]
        return prompt_ids, answer_ids

    def encode_multiturn(
            self,
            tokenizer: "PreTrainedTokenizer",
            messages: Sequence[Dict[str, str]],
            system: Optional[str] = None,
            tools: List[str] = None
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Returns multiple pairs of token ids representing prompts and responses respectively.
        """
        encoded_messages = self._encode(tokenizer, messages, system, tools)
        return [(encoded_messages[i], encoded_messages[i + 1]) for i in range(0, len(encoded_messages), 2)]

    def add_thought(self, content: str = "") -> str:
        r"""Add empty thought to assistant message."""
        return f"{self.thought_words[0]}\n\n{self.thought_words[1]}\n\n" + content

    def remove_thought(self, content: str) -> str:
        r"""Remove thought from assistant message."""
        pattern = re.compile(f"{re.escape(self.thought_words[0])}(.*?){re.escape(self.thought_words[1])}", re.DOTALL)
        return re.sub(pattern, "", content).lstrip("\n")

    def get_thought_word_ids(self, tokenizer: "PreTrainedTokenizer") -> list[int]:
        r"""Get the token ids of thought words."""
        return tokenizer.encode(self.add_thought(), add_special_tokens=False)

    def _encode(
            self,
            tokenizer: "PreTrainedTokenizer",
            messages: Sequence[Dict[str, str]],
            system: Optional[str],
            tools: List[str]
    ) -> List[List[int]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: prefix + system + query        resp
        Turn t: sep + query                    resp
        """
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []

            if i == 0:
                elements += self.format_prefix.apply()
                if system:
                    # add tool schema to the end of system prompt
                    if tools is not None and len(tools) > 0:
                        tool_schema = []
                        for t in tools:
                            tool_schema.append(t)
                        tool_schema = '\n'.join(tool_schema)
                        tools_prompt = self.format_tool.apply(content=tool_schema)
                        system += tools_prompt[0]
                    elements += self.format_system.apply(content=system)

            if message["role"] == Role.USER.value:
                elements += self.format_user.apply(
                    content=message["content"], idx=str(i // 2))
            elif message["role"] == Role.ASSISTANT.value:
                elements += self.format_assistant.apply(
                    content=message["content"])
            elif message["role"] == Role.OBSERVATION.value:
                elements += self.format_observation.apply(
                    content=message["content"])
            elif message["role"] == Role.FUNCTION.value:
                elements += self.format_function.apply(
                    content=message["content"])
            elif message["role"] == Role.TOOL_CALL:
                elements += self.format_assistant.apply(
                    content=message["content"])
            elif message["role"] == Role.TOOL_RESPONSE:
                elements += self.format_observation.apply(
                    content=message["content"])
            else:
                raise NotImplementedError(
                    "Unexpected role: {}".format(message["role"]))

            encoded_messages.append(
                self._convert_elements_to_ids(tokenizer, elements))

        return encoded_messages

    @staticmethod
    def _convert_elements_to_ids(tokenizer: "PreTrainedTokenizer", elements: "SLOTS") -> List[int]:
        r"""
        Converts elements to token ids.
        """
        token_ids = []
        for elem in elements:
            if isinstance(elem, str):
                if len(elem) != 0:
                    token_ids += tokenizer.encode(elem,
                                                  add_special_tokens=False)
            elif isinstance(elem, dict):
                token_ids += [
                    tokenizer.convert_tokens_to_ids(elem.get("token"))]
            elif isinstance(elem, set):
                if "bos_token" in elem and tokenizer.bos_token_id is not None:
                    token_ids += [tokenizer.bos_token_id]
                elif "eos_token" in elem and tokenizer.eos_token_id is not None:
                    token_ids += [tokenizer.eos_token_id]
            else:
                raise ValueError(
                    "Input must be string, set[str] or dict[str, str], got {}".format(type(elem)))

        return token_ids


@dataclass
class ReasoningTemplate(Template):
    r"""A template that add thought to assistant message."""

    @override
    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> tuple[list[int], list[int]]:
        messages = deepcopy(messages)
        for i in range(1, len(messages) - 2, 2):
            messages[i]["content"] = self.remove_thought(messages[i]["content"])

        if self.enable_thinking is False:  # remove all cot
            messages[-1]["content"] = self.remove_thought(messages[-1]["content"])

        prompt_ids, response_ids = super().encode_oneturn(tokenizer, messages, system, tools)
        if (
            self.thought_words[0] not in messages[-1]["content"]
            and self.thought_words[1] not in messages[-1]["content"]
        ):  # add empty cot
            if not self.enable_thinking:  # do not compute loss
                prompt_ids += self.get_thought_word_ids(tokenizer)
            else:  # do compute loss
                response_ids = self.get_thought_word_ids(tokenizer) + response_ids

        return prompt_ids, response_ids

    @override
    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> list[tuple[list[int], list[int]]]:
        messages = deepcopy(messages)
        if self.enable_thinking is False:  # remove all cot
            for i in range(1, len(messages), 2):
                messages[i]["content"] = self.remove_thought(messages[i]["content"])

        encoded_messages = self._encode(tokenizer, messages, system, tools)
        for i in range(0, len(messages), 2):
            if (
                self.thought_words[0] not in messages[i + 1]["content"]
                and self.thought_words[1] not in messages[i + 1]["content"]
            ):  # add empty cot
                if not self.enable_thinking:  # do not compute loss
                    encoded_messages[i] += self.get_thought_word_ids(tokenizer)
                else:  # do compute loss
                    encoded_messages[i + 1] = self.get_thought_word_ids(tokenizer) + encoded_messages[i + 1]

        return [(encoded_messages[i], encoded_messages[i + 1]) for i in range(0, len(encoded_messages), 2)]


TEMPLATES: Dict[str, "Template"] = {}


@dataclass
class RegisterParams:
    format_user: Optional["Formatter"] = None
    format_assistant: Optional["Formatter"] = None
    format_system: Optional["Formatter"] = None,
    format_observation: Optional["Formatter"] = None
    format_prefix: Optional["Formatter"] = None
    tool_prompt: Optional["Formatter"] = None
    default_system: str = ""
    stop_words: Optional[Sequence[str]] = None
    efficient_eos: bool = False
    replace_eos: bool = False
    enable_thinking: Optional[bool] = True
    thought_words: Optional[tuple[str, str]] = None


def _register_template(
        name: str,
        params: RegisterParams,
        mm_plugin: "BasePlugin" = get_mm_plugin(name="base"),
        template_class: type["Template"] = Template,
) -> None:
    r"""
    Registers a chat template.

    To add the following chat template:
    ```
    [HUMAN]:
    user prompt here
    [AI]:
    model response here

    [HUMAN]:
    user prompt here
    [AI]:
    model response here
    ```

    The corresponding code should be:
    ```
    _register_template(
        name="custom",
        RegisterParams(format_user=StringFormatter(slots=["[HUMAN]:\n{{content}}\n[AI]:\n"]),
        format_separator=EmptyFormatter(slots=["\n\n"]),
        efficient_eos=True),
    )
    ```
    """
    eos_slots = [] if params.efficient_eos else [{"eos_token"}]
    default_user_formatter = StringFormatter(slots=["{{content}}"])
    default_assistant_formatter = StringFormatter(
        slots=["{{content}}"] + eos_slots)

    default_separator_formatter = EmptyFormatter()
    default_prefix_formatter = EmptyFormatter()
    TEMPLATES[name] = template_class(
        format_user=params.format_user or default_user_formatter,
        format_assistant=params.format_assistant or default_assistant_formatter,
        format_system=params.format_system or default_user_formatter,
        format_observation=params.format_observation or params.format_user or default_user_formatter,
        format_prefix=params.format_prefix or default_prefix_formatter,
        format_tool=params.tool_prompt or None,
        default_system=params.default_system,
        stop_words=[] if params.stop_words is None else params.stop_words,
        efficient_eos=params.efficient_eos,
        replace_eos=params.replace_eos,
        enable_thinking=params.enable_thinking,
        thought_words=params.thought_words or ("<think>", "</think>"),
        mm_plugin=mm_plugin,
    )


_register_template(
    name="qwen2vl",
    params=RegisterParams(
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
        format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
        format_observation=StringFormatter(
            slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
        default_system="You are a helpful assistant.",
        stop_words=["<|im_end|>"],
        replace_eos=True),
    mm_plugin=get_mm_plugin(name="qwen2_vl", image_token="<|image_pad|>", video_token="<|video_pad|>")
)


# copied from qwen template
_register_template(
    name="qwen3_vl",
    params=RegisterParams(
        format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
        format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
        format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
        format_observation=StringFormatter(
            slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
        ),
        stop_words=["<|im_end|>"],
        replace_eos=True),
    mm_plugin=get_mm_plugin(name="qwen3_vl", image_token="<|image_pad|>", video_token="<|video_pad|>"),
    template_class=ReasoningTemplate,
)

tools_slot = '''

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{content}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>'''

# copied from qwen template
_register_template(
    name="qwen3_vl_nothink",
    params=RegisterParams(
        format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
        format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
        format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
        format_observation=StringFormatter(
            slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
        ),
        stop_words=["<|im_end|>"],
        replace_eos=True,
        tool_prompt=StringFormatter(slots=[tools_slot])
    ),
    mm_plugin=get_mm_plugin(name="qwen3_vl", image_token="<|image_pad|>", video_token="<|video_pad|>"),
)


# copied from qwen template
_register_template(
    name="qwen2_omni",
    params=RegisterParams(
        format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
        format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
        format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
        format_observation=StringFormatter(
            slots=[
                "<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
        ),

        default_system="You are a helpful assistant.",
        stop_words=["<|im_end|>"],
        replace_eos=True),
    mm_plugin=get_mm_plugin(
        name="qwen2_omni", audio_token="<|AUDIO|>", image_token="<|IMAGE|>", video_token="<|VIDEO|>")

)


_register_template(
    name="qwen3_omni",
    params=RegisterParams(
        format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
        format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
        format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
        format_observation=StringFormatter(
            slots=[
                "<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
        ),
        stop_words=["<|im_end|>"],
        replace_eos=True),
    mm_plugin=get_mm_plugin(
        name="qwen3_omni", audio_token="<|audio_pad|>", image_token="<|image_pad|>", video_token="<|video_pad|>"),
    template_class=ReasoningTemplate
)


_register_template(
    name="qwen3_omni_nothink",
    params=RegisterParams(
        format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
        format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
        format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
        format_observation=StringFormatter(
            slots=[
                "<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
        ),
        stop_words=["<|im_end|>"],
        replace_eos=True,
        tool_prompt=StringFormatter(slots=[tools_slot])),
    mm_plugin=get_mm_plugin(
        name="qwen3_omni", audio_token="<|audio_pad|>", image_token="<|image_pad|>", video_token="<|video_pad|>"),
)


_register_template(
    name="kimi_k25",
    params=RegisterParams(
        format_user=StringFormatter(
            slots=["<|im_user|>user<|im_middle|>{{content}}<|im_end|><|im_assistant|>assistant<|im_middle|>"]
        ),
        format_assistant=StringFormatter(slots=["{{content}}<|im_end|>"]),
        format_system=StringFormatter(slots=["<|im_system|>system<|im_middle|>{{content}}<|im_end|>"]),
        default_system="You are a helpful assistant",
        stop_words=["<|im_end|>"],
        thought_words=("◁think▷", "◁/think▷"),
        replace_eos=True,
    ),
    mm_plugin=get_mm_plugin("kimi_k25", image_token="<|media_pad|>"),
    template_class=ReasoningTemplate,
)


_register_template(
    name="glm4.1v",
    params=RegisterParams(
        format_user=StringFormatter(slots=["<|user|>\n{{content}}<|assistant|>"]),
        format_assistant=StringFormatter(slots=["\n{{content}}"]),
        format_system=StringFormatter(slots=["<|system|>\n{{content}}"]),
        format_observation=StringFormatter(slots=["<|observation|>\n{{content}}<|assistant|>"]),

        format_prefix=EmptyFormatter(slots=["[gMASK]<sop>"]),
        stop_words=["<|user|>", "<|observation|>", "</answer>"],
        efficient_eos=True
    ),
    mm_plugin=get_mm_plugin(name="glm4.1v", image_token="<|image|>", video_token="<|video|>"),
    template_class=ReasoningTemplate
)


_register_template(
    name="glm4v_moe",
    params=RegisterParams(
        format_user=StringFormatter(slots=["<|user|>\n{{content}}<|assistant|>"]),
        format_assistant=StringFormatter(slots=["\n{{content}}"]),
        format_system=StringFormatter(slots=["<|system|>\n{{content}}"]),
        format_observation=StringFormatter(slots=["<|observation|>\n{{content}}<|assistant|>"]),

        format_prefix=EmptyFormatter(slots=["[gMASK]<sop>"]),
        stop_words=["<|user|>", "<|observation|>", "</answer>"],
        efficient_eos=True
    ),
    mm_plugin=get_mm_plugin(name="glm4.1v", image_token="<|image|>", video_token="<|video|>"),
    template_class=ReasoningTemplate
)


_register_template(
    name="default",
    params=RegisterParams(
        format_user=StringFormatter(slots=["Human: {{content}}", {"eos_token"}, "\nAssistant:"]),
        format_assistant=StringFormatter(slots=["{{content}}", {"eos_token"}, "\n"]),
        format_system=StringFormatter(slots=["System: {{content}}", {"eos_token"}, "\n"]),
    ),
)


_register_template(
    name="empty",
    params=RegisterParams(
        format_assistant=StringFormatter(slots=["{{content}}"])
    ),
)


def parse_template(tokenizer: "PreTrainedTokenizer") -> "Template":
    r"""Extract a chat template from the tokenizer."""

    def find_diff(short_str: str, long_str: str) -> str:
        i, j = 0, 0
        diff = ""
        while i < len(short_str) and j < len(long_str):
            if short_str[i] == long_str[j]:
                i += 1
                j += 1
            else:
                diff += long_str[j]
                j += 1

        return diff

    prefix = tokenizer.decode(tokenizer.encode(""))

    # user role is mandatory for some templates
    messages = [{"role": "system", "content": "{{content}}"}, {"role": "user", "content": ""}]
    system_slot = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)[len(prefix):]
    messages = [{"role": "user", "content": ""}]
    user_slot_empty_content = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)[len(prefix):]
    system_slot = system_slot.split(user_slot_empty_content)[0]

    messages = [{"role": "system", "content": ""}, {"role": "user", "content": "{{content}}"}]
    user_slot_empty_system = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    user_slot_empty_system = user_slot_empty_system[len(prefix):]
    user_slot_empty_system = user_slot_empty_system.replace("<think>\n", "") # remove thought tags

    messages = [{"role": "user", "content": "{{content}}"}]
    user_slot = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    user_slot = user_slot[len(prefix):]
    user_slot = user_slot.replace("<think>\n", "") # remove thought tags

    messages = [{"role": "user", "content": "{{content}}"}, {"role": "assistant", "content": "{{content}}"}]
    assistant_slot = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    assistant_slot = assistant_slot[len(prefix) + len(user_slot):]
    template_class = ReasoningTemplate if "<think>" in assistant_slot else Template
    assistant_slot = assistant_slot.replace("<think>", "").replace("</think>", "").lstrip("\n")  # remove thought tags

    if len(user_slot) > len(user_slot_empty_system):
        default_system = find_diff(user_slot_empty_system, user_slot)
        sole_system = system_slot.replace("{{content}}", default_system, 1)
        user_slot = user_slot[len(sole_system):]
    else:  # if defaut_system is empty, user_slot_empty_system will be longer than user_slot
        default_system = ""

    return template_class(
        format_user=StringFormatter(slots=[user_slot]),
        format_assistant=StringFormatter(slots=[assistant_slot]),
        format_system=StringFormatter(slots=[system_slot]),
        format_observation=StringFormatter(slots=[user_slot]),
        format_tool=StringFormatter(slots=[tools_slot]),
        format_prefix=EmptyFormatter(slots=[prefix]) if prefix else EmptyFormatter(),
        default_system=default_system,
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        efficient_eos=False,
        replace_eos=False,
        enable_thinking=True,
        mm_plugin=get_mm_plugin(name="base"),
    )


def get_template_and_fix_tokenizer(tokenizer: "PreTrainedTokenizer", template: str) -> "Template":
    r"""
    Gets chat template and fixes the tokenizer.
    """

    if template is None:
        if isinstance(tokenizer.chat_template, str):
            logger.warning("`template` was not specified, try parsing the chat template from the tokenizer.")
            template = parse_template(tokenizer)
        else:
            logger.warning("`template` was not specified, use `empty` template.")
            template = TEMPLATES["empty"]  # placeholder
    else:
        template = TEMPLATES.get(template, None)
        if template is None:
            raise ValueError(
                "Template {} does not exist.".format(template))

    stop_words = template.stop_words
    if template.replace_eos:
        if not stop_words:
            raise ValueError(
                "Stop words are required to replace the EOS token.")

        _add_or_replace_eos_token(tokenizer, eos_token=stop_words[0])
        stop_words = stop_words[1:]

    if tokenizer.eos_token_id is None:
        _add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Add pad token: {}".format(tokenizer.pad_token))

    if stop_words:
        num_added_tokens = tokenizer.add_special_tokens(
            dict(additional_special_tokens=stop_words), replace_additional_special_tokens=False
        )
        logger.info("Add {} to stop words.".format(",".join(stop_words)))
        if num_added_tokens > 0:
            logger.warning(
                "New tokens have been added, make sure `resize_vocab` is True.")

    return template


def _add_or_replace_eos_token(tokenizer: "PreTrainedTokenizer", eos_token: str) -> None:
    is_added = tokenizer.eos_token_id is None
    num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

    if is_added:
        logger.info("Add eos token: {}".format(tokenizer.eos_token))
    else:
        logger.info("Replace eos token: {}".format(tokenizer.eos_token))

    if num_added_tokens > 0:
        logger.warning(
            "New tokens have been added, make sure `resize_vocab` is True.")
