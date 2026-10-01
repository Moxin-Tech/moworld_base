import json

import transformers


class HunyuanMllmTokenizer:
    def __init__(
        self,
        **config,
    ):
        template_file_path = config.pop("template_file_path")
        template_id = config.pop("template_id", "hyv-llm-encode-video")
        with open(template_file_path, "r") as f:
            templates = json.load(f)
        self.template_info = templates[template_id]
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(**config)


    @staticmethod
    def apply_template(text, template):
        if isinstance(text, str):
            return [template.format(text)]
        elif isinstance(text, list) or isinstance(text, tuple):
            return [template.format(t) for t in text]
        else:
            raise NotImplementedError(f"Not Support text type: {type(text)}")


    def __call__(
        self,
        prompt,
        padding: str = "max_length",
        max_length: int = 256,
        truncation: bool = True,
        return_attention_mask: bool = True,
        add_special_tokens: bool = True,
        return_tensors: str = "pt",
        **kwargs,
    ):
        prompt = HunyuanMllmTokenizer.apply_template(prompt, self.template_info["template"])
        text_inputs = self.tokenizer(
            prompt,
            padding=padding,
            max_length=max_length + self.template_info["crop_start"],
            truncation=truncation,
            return_attention_mask=return_attention_mask,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors,
            **kwargs,
        )
        return text_inputs

    def __getattr__(self, name):
        if name in dir(self):
            return super().__getattr__(name)
        else:
            return getattr(self.tokenizer, name)
