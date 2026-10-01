# Copyright 2025 StepFun Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# ==============================================================================
from typing import List

import sentencepiece
import torch


TOKEN_FLAG_MAPPINGS = {
    "_bot_id": "<|BOT|>",  # Begin of Turn
    "_eot_id": "<|EOT|>",  # End of Turn
    "_call_start_id": "<|CALL_START|>",  # Call Start
    "_call_end_id": "<|CALL_END|>",  # Call End
    "_think_start_id": "<|THINK_START|>",  # Think Start
    "_think_end_id": "<|THINK_END|>",  # Think End
    "_mask_start_id": "<|MASK_1e69f|>",  # Mask start
    "_mask_end_id": "<|UNMASK_1e69f|>",  # Mask end
}


class StepChatTokenizer:
    """Step Chat Tokenizer"""
    def __init__(
        self, pretrained_model_name_or_path,
        model_max_length=320,
        **kwargs
    ):
        self._tokenizer = sentencepiece.SentencePieceProcessor(model_file=pretrained_model_name_or_path)
        self.model_max_length = model_max_length

        self._vocab = {}
        self._inv_vocab = {}

        self._special_tokens = {}
        self._inv_special_tokens = {}

        self._t5_tokens = []

        for idx in range(self._tokenizer.get_piece_size()):
            text = self._tokenizer.id_to_piece(idx)
            self._inv_vocab[idx] = text
            self._vocab[text] = idx

            if self._tokenizer.is_control(idx) or self._tokenizer.is_unknown(idx):
                self._special_tokens[text] = idx
                self._inv_special_tokens[idx] = text

        self._unk_id = self._tokenizer.unk_id()
        self._bos_id = self._tokenizer.bos_id()
        self._eos_id = self._tokenizer.eos_id()

        self.check_tokens()

        for key, value in TOKEN_FLAG_MAPPINGS.items():
            setattr(self, key, self._tokenizer.piece_to_id(value))

        self._underline_id = self._tokenizer.piece_to_id("\u2581")
        with open("./token.txt", 'w') as f:
            f.write(str(self.__dict__))

    @property
    def get_tokenizer(self):
        return self._tokenizer

    @property
    def vocab(self):
        return self._vocab

    @property
    def inv_vocab(self):
        return self._inv_vocab

    @property
    def vocab_size(self):
        return self._tokenizer.vocab_size()

    def tokenize(self, text: str) -> List[int]:
        return self._tokenizer.encode_as_ids(text)

    def detokenize(self, token_ids: List[int]) -> str:
        return self._tokenizer.decode_ids(token_ids)

    def check_tokens(self):
        for key in ["_bot_id", "_eot_id", "_call_start_id", "_call_end_id", "_think_start_id", "_think_end_id",
                    "_mask_start_id", "_mask_end_id"]:
            token = TOKEN_FLAG_MAPPINGS[key]
            if token not in self._vocab:
                raise Exception(f"Token '{token}' not found in tokenizer")

        for key in ["_bot_id", "_eot_id", "_call_start_id", "_call_end_id", "_think_start_id", "_think_end_id"]:
            token = TOKEN_FLAG_MAPPINGS[key]
            if token not in self._special_tokens:
                raise Exception(f"Token '{token}' is not a special token")


class Tokens:
    def __init__(self, input_ids, attention_mask) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        return self

    def __getitem__(self, item):
        if hasattr(self, item):
            return getattr(self, item)
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")


class WrappedStepChatTokenizer(StepChatTokenizer):
    def __call__(self, text, max_length=320, **kwargs):
        self.BOS = 1
        self.EOS = 2
        self.PAD = 2
        out_tokens = []
        attn_mask = []
        if not isinstance(text, list):
            text = [text]
        if len(text) == 0:
            part_tokens = [self.BOS] + [self.EOS]
            valid_size = len(part_tokens)
            if len(part_tokens) < max_length:
                part_tokens += [self.PAD] * (max_length - valid_size)
            out_tokens.append(part_tokens)
            attn_mask.append([1] * valid_size + [0] * (max_length - valid_size))
        else:
            for part in text:
                part_tokens = self.tokenize(part)
                part_tokens = part_tokens[:(max_length - 2)]  # leave 2 space for bos and eos
                part_tokens = [self.BOS] + part_tokens + [self.EOS]
                valid_size = len(part_tokens)
                if len(part_tokens) < max_length:
                    part_tokens += [self.PAD] * (max_length - valid_size)
                out_tokens.append(part_tokens)
                attn_mask.append([1] * valid_size + [0] * (max_length - valid_size))

        out_tokens = torch.tensor(out_tokens, dtype=torch.long)
        attn_mask = torch.tensor(attn_mask, dtype=torch.long)

        return Tokens(out_tokens, attn_mask)
