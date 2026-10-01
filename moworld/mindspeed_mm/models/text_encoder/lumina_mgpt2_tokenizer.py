from typing import List, Optional

from transformers import AutoTokenizer


class LuminaMGPT2Tokenizer:
    def __init__(self, **config):
        self.tokenizer = AutoTokenizer.from_pretrained(**config)
         # BOS / EOS token IDs
        self.bos_id: int = self.tokenizer.bos_token_id
        if self.bos_id is None:
            self.bos_id = self.tokenizer.eos_token_id
        self.eos_id: int = self.tokenizer.eos_token_id
        self._probe_tokenizer_style()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        t = self.tokenizer.encode(s, truncation=False, add_special_tokens=False)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def encode_wo_prefix_space(self, s: str):
        if self.need_space_before_segment:
            return self.encode(s, bos=False, eos=False)
        else:
            # prefix chars that, when preceding other strings without separator in between,
            # are relatively more likely to be tokenized independently rather than getting
            # merged into the following strings.
            l_prefix = ["@", "\n", "\\", "=", ">", "`"]
            for prefix in l_prefix:
                prefix_tokens = self.encode(prefix, bos=False, eos=False)
                cat_tokens = self.encode(prefix + s, bos=False, eos=False)
                if cat_tokens[: len(prefix_tokens)] == prefix_tokens:
                    return cat_tokens[len(prefix_tokens):]

            raise NotImplementedError(
                f"All prefixes are merged into {s} during tokenization,"
                f"This is weird behavior, please open an issue to report this problem",
            )

    def _probe_tokenizer_style(self):
        """
        Given a sentence, e.g. "Hi my darling", some tokenizers (e.g. LLaMA's) will pose the following behavior:
        >>> # leading characters will be treated as if there were an " " in the beginning
        >>> tokenizer.encode("Hi my darling") == tokenizer.encode("Hi") + tokenizer.encode("my darling")
        >>> # leading space " " is redundant and should not be added
        >>> tokenizer.encode("Hi my darling") != tokenizer.encode("Hi") + tokenizer.encode(" my darling")
        However, some others (e.g. InternLM's) will behave differently:
        >>> # leading space " " has to be explicitly added
        >>> tokenizer.encode("Hi my darling") == tokenizer.encode("Hi") + tokenizer.encode(" my darling")
        Knowing which style the tokenizer takes is necessary when tokenzing a segment cut from the complete
        text, so that the result is the same as the corresponding part in the tokenized original text.
        """
        sentence1 = self.encode("Hi my darling", bos=False, eos=False)
        sentence2 = self.encode("my darling", bos=False, eos=False)
        if sentence1[-len(sentence2):] == sentence2:
            self.need_space_before_segment = False
        else:
            sentence3 = self.encode(" my darling", bos=False, eos=False)
            if sentence1[-len(sentence3):] != sentence3:
                raise AssertionError
            self.need_space_before_segment = True
