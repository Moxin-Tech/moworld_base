import dataclasses
from enum import auto, Enum
from typing import Dict, List, Tuple, Union


class SeparatorStyle(Enum):
    """Different separator style."""
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    DeepSeek = auto()
    DeepSeekV2 = auto()
    LuminaMGPT2 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = "{system_message}"
    # The system message
    system_message: str = ""
    # The names of two roles
    roles: Tuple[str] = ("USER", "ASSISTANT")
    # All messages. Each item is (role, message).
    messages: Tuple = ()
    # The number of few shot examples
    offset: int = 0
    # The separator style and configurations
    sep_style: SeparatorStyle = SeparatorStyle.TWO
    sep: str = "\n"
    sep2: str = None
    version: str = "Unknown"
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        system_prompt = self.system_template.format(system_message=self.system_message)
        if self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if isinstance(message, tuple):
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.MPT:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    if isinstance(message, tuple):
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.DeepSeek:
            seps = [self.sep, self.sep2]
            if system_prompt == "" or system_prompt is None:
                ret = ""
            else:
                ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.DeepSeekV2:
            seps = [self.sep, self.sep2]
            if system_prompt == "" or system_prompt is None:
                ret = ""
            else:
                ret = system_prompt + seps[0]
            for _, (role, message) in enumerate(self.messages):
                if message:
                    if role == "User":
                        # <｜sft▁begin｜>User Input<｜sft▁end｜>\nResponse<｜end▁of▁sentence｜>
                        ret += "<｜sft▁begin｜>\n" + message + self.sep
                    else:
                        ret += message + self.sep2
                else:
                    ret = ret
        elif self.sep_style == SeparatorStyle.LuminaMGPT2:
            ret = ""
            pieces = []
            for role, message in self.messages:
                if message:
                    turn = message + self.sep
                    ret += turn
                    if role == self.roles[1]:
                        pieces.append({"data": turn, "predict": True})
                    else:
                        pieces.append({"data": turn, "predict": False})
            return {
                "conv": ret,  # text involving the complete conversation
                "pieces": pieces,  # list to help correctly mark the labels
            }
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def reset_message(self):
        """Reset a new message."""
        self.messages = []

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override and template.name in conv_templates:
        raise AssertionError(f"{template.name} has been registered.")

    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    try:
        return conv_templates[name].copy()
    except KeyError as e:
        raise KeyError(f"Conversation template '{name}' not found.") from e


register_conv_template(
    Conversation(
        name="internlm2-chat",
        system_template="<|im_start|>system\n{system_message}",
        # note: The new system prompt was not used here to avoid changes in benchmark performance.
        system_message="你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。",
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>",
        stop_token_ids=[
            2,
            92543,
            92542
        ]
    )
)

register_conv_template(
    Conversation(
        name='internvl2_5',
        system_template='<|im_start|>system\n{system_message}',
        system_message='你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。',
        roles=('<|im_start|>user\n', '<|im_start|>assistant\n'),
        sep_style=SeparatorStyle.MPT,
        sep='<|im_end|>\n',
    )
)

register_conv_template(
    Conversation(
        name='internvit_qwen3',
        system_template='<|im_start|>system\n{system_message}',
        system_message='',
        roles=('<|im_start|>user\n', '<|im_start|>assistant\n'),
        sep_style=SeparatorStyle.MPT,
        sep='<|im_end|>\n',
    )
)

register_conv_template(
    Conversation(
        name="deepseek",
        system_template="{system_message}",
        # system_message："You are a helpful assistant. Please answer truthfully and write out your "
        # "thinking step by step to be sure you get the right answer.",
        system_message="",
        roles=("<|User|>", "<|Assistant|>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.DeepSeek,
        sep="\n\n",
        sep2="<｜end▁of▁sentence｜>",
        stop_token_ids=[100001],
        stop_str=["User:", "<｜end▁of▁sentence｜>"]
    )
)

register_conv_template(
    Conversation(
        name="deepseekv2",
        system_template="{system_message}",
        system_message="",
        roles=("|<User>|", "|<Assistant>|"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.DeepSeekV2,
        sep="\n<｜sft▁end｜>",
        sep2="<｜end▁of▁sentence｜>",
        stop_token_ids=[100001],
        stop_str=["User:", "<｜end▁of▁sentence｜>"]
    )
)

register_conv_template(
    Conversation(
        name="lumina-mgpt2",
        roles=("Human", "Assistant"),
        messages=(),
        sep_style=SeparatorStyle.LuminaMGPT2,
        sep="<reserved08706>"
    )
)
