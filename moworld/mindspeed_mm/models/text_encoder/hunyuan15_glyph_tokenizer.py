import json

import torch
import transformers

from mindspeed_mm.models.text_encoder.hunyuan15_byt5.format_prompt import MultilingualPromptFormat


class Hunyuan15GlyphTokenizer:
    def __init__(
            self,
            **config,
    ):
        pretrained_model_name_or_path = config.get("pretrained_model_name_or_path", None)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path,
                                                                    cache_dir=None)
        self.color_ann_path = config.get("color_ann_path", "")
        self.font_ann_path = config.get("font_ann_path", "")
        self.byt5_max_length = config.get("byt5_max_length", 256)
        self.add_special_token(
            tokenizer=self.tokenizer,
            text_encoder=None,
            add_color=True,
            add_font=True,
            color_ann_path=self.color_ann_path,
            font_ann_path=self.font_ann_path,
            multilingual=True,
        )
        self.prompt_format = MultilingualPromptFormat(
            font_path=self.font_ann_path,
            color_path=self.color_ann_path
        )

    @staticmethod
    def add_special_token(
            tokenizer,
            text_encoder,
            add_color,
            add_font,
            color_ann_path,
            font_ann_path,
            multilingual=False,
            token_len=1510,
            token_add=True,
    ):
        """
        Add special tokens for color and font to tokenizer and text encoder.

        Args:
            token_add: (bool) Whether to add special token.
            token_len:max_token_length.
            text_encoder: Hunyuan_video second text encoder.
            tokenizer: Huggingface tokenizer.
            add_color (bool): Whether to add color tokens.
            add_font (bool): Whether to add font tokens.
            color_ann_path (str): Path to color annotation JSON.
            font_ann_path (str): Path to font annotation JSON.
            multilingual (bool): Whether to use multilingual font tokens.
        """
        with open(font_ann_path, 'r') as f:
            idx_font_dict = json.load(f)
        with open(color_ann_path, 'r') as f:
            idx_color_dict = json.load(f)

        if multilingual:
            font_token = [f'<{font_code[:2]}-font-{idx_font_dict[font_code]}>' for font_code in idx_font_dict]
        else:
            font_token = [f'<font-{i}>' for i in range(len(idx_font_dict))]

        color_token = [f'<color-{i}>' for i in range(len(idx_color_dict))]
        additional_special_tokens = []
        if add_color:
            additional_special_tokens += color_token
        if add_font:
            additional_special_tokens += font_token

        if token_add:
            tokenizer.add_tokens(additional_special_tokens, special_tokens=True)
        if text_encoder is not None:
            text_encoder.resize_token_embeddings(len(tokenizer) if tokenizer is not None else token_len,
                                                 mean_resizing=False)

    def __getattr__(self, name):
        if name in dir(self):
            return super().__getattr__(name)
        else:
            return getattr(self.tokenizer, name)

    def _extract_glyph_texts(self, prompt):
        """
        Extract glyph texts from prompt using regex pattern.

        Args:
            prompt: Input prompt string containing quoted text.

        Returns:
            List[str]: List of extracted glyph texts (deduplicated if multiple).
        """
        # extract en text
        en_results = []
        start = 0
        while True:
            open_idx = prompt.find('"', start)
            if open_idx == -1:
                break
            close_idx = prompt.find('"', open_idx + 1)
            if close_idx == -1:
                break
            en_results.append(prompt[open_idx + 1: close_idx])
            start = close_idx + 1

        # extract zh text
        zh_results = []
        start = 0
        while True:
            open_idx = prompt.find('“', start)
            if open_idx == -1:
                break
            close_idx = prompt.find('”', open_idx + 1)
            if close_idx == -1:
                break
            zh_results.append(prompt[open_idx + 1: close_idx])
            start = close_idx + 1

        # combine text res and keep sequence
        seen = set()
        final = []
        for t in en_results + zh_results:
            if t not in seen:
                seen.add(t)
                final.append(t)
        return final

    @staticmethod
    def get_byt5_text_tokens(byt5_tokenizer, byt5_max_length, text_prompt):
        """
        Tokenize text prompt for byT5 model.

        Args:
            byt5_tokenizer: The byT5 tokenizer.
            byt5_max_length: Maximum sequence length for tokenization.
            text_prompt: Text prompt string to tokenize.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - input_ids: Tokenized input IDs.
                - attention_mask: Attention mask tensor.
        """
        byt5_text_inputs = byt5_tokenizer(
            text_prompt,
            padding="max_length",
            max_length=byt5_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        return byt5_text_inputs

    def __call__(
            self,
            prompt,
            **kwargs,
    ):
        glyph_texts = self._extract_glyph_texts(prompt)

        if len(glyph_texts) > 0:
            text_styles = [{'color': None, 'font-family': None} for _ in range(len(glyph_texts))]
            formatted_text = self.prompt_format.format_prompt(glyph_texts, text_styles)

            byt5_text_inputs = self.get_byt5_text_tokens(
                self.tokenizer, self.byt5_max_length, formatted_text
            )
            return byt5_text_inputs

        # res includes input_ids and attention mask
        return {
            "input_ids": torch.zeros((1, 256)),
            "attention_mask": torch.zeros((1, self.byt5_max_length), dtype=torch.int64)
        }
