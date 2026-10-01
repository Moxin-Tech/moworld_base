from copy import deepcopy

import transformers

from mindspeed_mm.models.text_encoder.hunyuan_mllm_tokenizer import HunyuanMllmTokenizer


class Hunyuan15MllmTokenizer(HunyuanMllmTokenizer):
    def __init__(
            self,
            **config,
    ):
        super().__init__(**config)

    @staticmethod
    def apply_text_to_template(text, template, prevent_empty_text=True):
        if isinstance(template, str):
            return template.format(text)
        elif isinstance(template, list):
            template_copy = deepcopy(template)
            for item in template_copy:
                if isinstance(item, dict) and "content" in item:
                    item["content"] = item["content"].format(text if text else (" " if prevent_empty_text else ""))
            return template_copy
        else:
            raise TypeError(f"Unsupported template type: {type(template)}")

    def __call__(
            self,
            text,
            data_type: str = "video",
            max_length: int = 300,
            padding: str = "max_length",
            truncation: bool = True,
            return_attention_mask: bool = True,
            return_tensors: str = "pt",
            **kwargs
    ):
        # depending on the data type, different templates are used below.
        tokenize_input_type = "str"
        if data_type not in ["video", "images"]:
            raise AssertionError(f"Unsupported data type: {data_type}")
        prompt_template = self.template_info["template"]
        crop_start = self.template_info.get("crop_start", -1)

        # change the text according to the template
        if isinstance(text, (list, tuple)):
            text = [
                self.apply_text_to_template(one_text, prompt_template)
                for one_text in text
            ]
            if isinstance(text[0], list):
                tokenize_input_type = "list"
        elif isinstance(text, str):
            text = self.apply_text_to_template(text, prompt_template)
            if isinstance(text, list):
                tokenize_input_type = "list"
        else:
            raise TypeError(f"Unsupported text type: {type(text)}")

        # tokenize with the proper max_length using the found crop_start
        args = dict(
            truncation=truncation,
            max_length=max_length + (crop_start if crop_start > 0 else 0),
            padding=padding,
            return_tensors=return_tensors,
        )

        if tokenize_input_type == "str":
            tokenized_output = self.tokenizer(
                text,
                return_length=False,
                return_overflowing_tokens=False,
                return_attention_mask=return_attention_mask,
                **args,
            )
        elif tokenize_input_type == "list":
            tokenized_output = self.tokenizer.apply_chat_template(
                text,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                **args,
            )
        else:
            raise ValueError(f"Unsupported tokenize_input_type: {tokenize_input_type}")

        return tokenized_output
