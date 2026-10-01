import os
import copy
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import PIL
from PIL import Image

import torch

from megatron.training import get_args, print_rank_0

from mindspeed_mm.models.text_encoder import Tokenizer
from mindspeed_mm.models.ae import AEModel
from mindspeed_mm.utils.utils import Registry
from mindspeed_mm.data.data_utils.data_transform import center_crop_arr
from mindspeed_mm.data.data_utils.conversation import get_conv_template


def var_center_crop(pil_image, crop_size_list, random_top_k=1):
    w, h = pil_image.size

    if (w, h) in crop_size_list:
        return pil_image

    rem_percent = [min(cw / w, ch / h) / max(cw / w, ch / h) for cw, ch in crop_size_list]
    crop_size = random.choice(sorted(((x, y) for x, y in zip(rem_percent, crop_size_list)), reverse=True)[:random_top_k])[1]
    return center_crop_arr(pil_image, crop_size)


def generate_crop_size_list(num_patches, patch_size, max_ratio=4.0):
    crop_size_list = []
    wp, hp = num_patches, 1
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            crop_size_list.append((wp * patch_size, hp * patch_size))
        if (hp + 1) * wp <= num_patches:
            hp += 1
        else:
            wp -= 1
    return crop_size_list


class ItemProcessor:
    """
    Factory class for creating video processor instances
    """
    @staticmethod
    def create(item_processor_type=None, **kwargs) -> "ItemProcessorBase":
        """
        Initialize with specified item processor type

        Args:
            item_processor_type: Registered item processor type (e.g., 'LoadFeatureItemProcessor', 'TokenizerItemProcessor')
        """
        processor_cls = Registry.get_class(item_processor_type)
        return processor_cls(**kwargs)


class ItemProcessorBase(ABC):
    @abstractmethod
    def process_item(self, data_item: dict, training_mode=False) -> Tuple[List, List]:
        raise NotImplementedError

    def predict_item_token_length(self, data_item: dict) -> int:
        """
        estimate the token length of the data item for gathering items of similar lengths into a batch
        """
        return 1


@Registry.register
class LoadFeatureItemProcessor(ItemProcessorBase):
    image_end_token_id = 151666  # image_end_token <eoss> in vocabulary

    def __init__(self, **kwargs):
        pass

    def process_item(self, data_item: dict, training_mode=False) -> Tuple[List, List]:
        if "token" in data_item and "label" in data_item:
            data_item = data_item
        else:
            path = data_item["file"]
            data_item = torch.load(path)
        tokens = data_item["token"]
        labels = data_item["label"]
        if len(tokens) != len(labels):
            raise AssertionError(f"The length of tokens({len(tokens)}) should be equal to the length of labels({len(labels)}).")
        if tokens[-2] == labels[-2] == self.image_end_token_id and tokens.count(self.image_end_token_id) == 1:
            if random.random() < 0.1:
                tokens = labels = [_ for _ in labels[:-1] if _ != -100]
        return tokens, labels

    def predict_item_token_length(self, data_item: dict) -> int:
        if "token" in data_item:
            return len(data_item["token"])
        elif "len" in data_item:
            return data_item["len"]
        else:
            raise ValueError()


@Registry.register
class TokenizerItemProcessor(ItemProcessorBase):
    image_start_token = "<racm3:break>"  # fixed tokens for start and end, so can hardcode
    image_end_token = "<eoss>"
    full_sub_sep_token = "<reserved08796>"
    sub_sub_sep_token = "<reserved08797>"
    sub_skip_token = "<reserved08798>"
    new_line_token = "<reserved08799>"

    def __init__(
        self,
        patch_size=32,
        target_size=512,
        tokenizer_config: Optional[Union[dict, List[dict]]] = None,
        movqgan_config: Optional[Union[dict, List[dict]]] = None,
        device="npu",
        **kwargs
    ):
        args = get_args()
        self.patch_size = patch_size
        self.crop_size_list = generate_crop_size_list((target_size // self.patch_size) ** 2, self.patch_size)
        print_rank_0("List of crop sizes:")
        for i in range(0, len(self.crop_size_list), 6):
            print_rank_0(" " + "".join([f"{f'{w} x {h}':14s}" for w, h in self.crop_size_list[i: i + 6]]))

        # prepare model
        self.tokenizer = Tokenizer(tokenizer_config).get_tokenizer()
        self.vqgan = AEModel(args.mm.model.ae).get_model().to(device).eval()

        # media
        self.media_symbols = ["<|image|>"]
        self.tokenizer.tokenizer.add_tokens(self.media_symbols)
        self.d_media_symbol2token = {}
        self.d_media_token2symbol = {}
        for media_symbol in self.media_symbols:
            tokenized_symbol = self.tokenizer.encode(media_symbol, bos=False, eos=False)
            self.d_media_symbol2token[media_symbol] = tokenized_symbol[0]
            self.d_media_token2symbol[tokenized_symbol[0]] = media_symbol

    @staticmethod
    def get_n_grids_token(n_grids):
        return f"<reserved{8800 + n_grids:05d}>"

    @staticmethod
    def get_image_token(img_token):
        return img_token + 155000

    def token2id(self, token: str) -> int:
        return self.tokenizer.tokenizer.vocab[token]

    def id2token(self, id_) -> str:
        voc = self.tokenizer.tokenizer.vocab
        return list(voc.keys())[list(voc.values()).index(id_.item())]

    def _whiten_transparency(self, img: PIL.Image) -> PIL.Image:
        # Check if it's already in RGB format.
        if img.mode == "RGB":
            return img

        vals_rgba = np.array(img.convert("RGBA"))
        # If there is no transparency layer, simple convert and return.
        if not (vals_rgba[:, :, 3] < 255).any():
            return img.convert("RGB")
        # There is a transparency layer, blend it with a white background.
        # Calculate the alpha proportion for blending.
        alpha = vals_rgba[:, :, 3] / 255.0
        # Blend with white background.
        vals_rgb = (1 - alpha[:, :, np.newaxis]) * 255 + alpha[:, :, np.newaxis] * vals_rgba[:, :, :3]
        return PIL.Image.fromarray(vals_rgb.astype("uint8"), "RGB")

    def img_tokens_from_pil(self, img: PIL.Image) -> list[int]:
        img = self._whiten_transparency(img)
        # Convert to tensor.
        np_img = np.array(img) / 255.0  # Normalize to [0, 1]
        np_img = np_img * 2 - 1  # Scale to [-1, 1]
        img = torch.from_numpy(np_img).permute(2, 0, 1).to(self.vqgan.encoder.conv_in.weight)

        img = img.unsqueeze(0)
        info = self.vqgan.encode(img)
        return info

    def process_image(self, image) -> Dict:
        if isinstance(image, Image.Image):
            pass
        else:
            with Image.open(image) as img:
                image = img.copy()

        image = var_center_crop(image, crop_size_list=self.crop_size_list)
        w_grids, h_grids = image.size[0] // self.patch_size, image.size[1] // self.patch_size

        image_toks = self.img_tokens_from_pil(image)
        image_toks = image_toks.view(-1)
        full_image_toks = self.get_image_token(image_toks.reshape(image.size[1] // 8, image.size[0] // 8))
        new_line_id = self.token2id(self.new_line_token)

        full_image_toks = torch.cat(
            (
                full_image_toks,
                torch.ones(image.size[1] // 8, 1, device=full_image_toks.device, dtype=full_image_toks.dtype)
                * new_line_id,
            ),
            dim=1,
        ).flatten()

        result_toks = [
            self.token2id(self.image_start_token),
            self.token2id(self.get_n_grids_token(h_grids)),
            self.token2id(self.get_n_grids_token(w_grids)),
            *full_image_toks.tolist(),
            self.token2id(self.image_end_token),
        ]
        return {"input_ids": result_toks, "labels": result_toks}

    def process_item(self, item, training_mode=False):
        if training_mode:
            file_name = item.get('file', "NA")
            item = self.preprocess_item(item)
            tokens, labels = self.get_txt_and_img_tokens(item)
            if all([_ <= 0 for _ in labels]): # nothing to predict
                raise Exception("all label values are zero, nothing to predict")
            input_tokens_item = []
            modified_labels_item = []
            for _, (token_or_media, ori_label) in enumerate(zip(tokens, labels)):
                if isinstance(token_or_media, int):
                    token = token_or_media
                    input_tokens_item.append(token)
                    modified_labels_item.append(ori_label)
                else:
                    input_tokens_item += token_or_media["input_ids"]
                    if ori_label <= 0:  # in the prompt part
                        modified_labels_item += [-100] * len(token_or_media["input_ids"])
                    else:
                        modified_labels_item += token_or_media["labels"]

            return file_name, input_tokens_item, modified_labels_item
        else:
            tokens, _ = self.get_txt_and_img_tokens(item)
            input_tokens_item = []
            for _, token_or_media in enumerate(tokens):
                if isinstance(token_or_media, int):
                    input_tokens_item.append(token_or_media)
                else:
                    input_tokens_item += token_or_media["input_ids"]

            return input_tokens_item

    def get_txt_and_img_tokens(self, data_item):
        d_media = self.collect_and_process_media(data_item)
        source = self.insert_implicit_media_symbol_in_q1(data_item["conversations"], d_media)
        conversation, pieces = self.add_speaker_and_signal(source)

        # dialog does not need eos
        tokens = self.tokenizer.encode(conversation, bos=True, eos=False)
        labels = [-100 for _ in tokens]

        # check special token num as expected
        for media_symbol, l_media in d_media.items():
            media_token = self.d_media_symbol2token[media_symbol]
            media_token_count = tokens.count(media_token)
            if media_token_count != len(l_media):
                raise Exception(
                    f"{media_token_count} {media_token} (for {media_symbol}) exists in tokenized conversation, "
                    f"but {len(l_media)} actual media are given"
                )

        check_pos = 0
        for i, p in enumerate(pieces):
            if i == 0:
                tokenized_value = self.tokenizer.encode(p["data"], bos=True, eos=False)
            else:
                tokenized_value = self.tokenizer.encode_wo_prefix_space(p["data"])

            if tokens[check_pos: check_pos + len(tokenized_value)] != tokenized_value:
                raise Exception("inconsistent complete conversation and corresponding piece after tokenization")

            if p["predict"]:
                labels[check_pos: check_pos + len(tokenized_value)] = tokenized_value
            check_pos = check_pos + len(tokenized_value)

        # labels will be processed later by the model
        tokens, labels = self.replace_media_token_with_media(tokens, labels, d_media)
        return tokens, labels

    def collect_and_process_media(self, data_item):
        """
        this function receives a raw piece of data (e.g. read from `.json` data file),
        and returns d_media, containing the prepared media readily usable by model
        YOU MAY OVERRIDE THIS FUNCTION TO SUPPORT COMPLEX LOADING OF VARIOUS FORMS OF DATA
        """
        d_media = {}
        for media_symbol in self.media_symbols:
            if media_symbol in data_item:
                l_media = data_item[media_symbol]  # a list of media paths
            elif media_symbol.lstrip("<|").rstrip("|>") in data_item:
                l_media = data_item[media_symbol.lstrip("<|").rstrip("|>")]
            else:
                l_media = []
            if not isinstance(l_media, list):  # data with only one media, in format {"image": image_name, ...}
                l_media = [l_media]

            d_media[media_symbol] = []
            for media in l_media:
                media = self.process_image(media)
                media["type"] = media_symbol
                d_media[media_symbol].append(media)

        return d_media

    def replace_media_token_with_media(
        self, tokens: List[int], labels: Union[List[int], None], d_media: Dict[str, List]
    ):
        d_media_counter = {key: 0 for key in d_media}
        for i, t in enumerate(tokens):
            if t in self.d_media_token2symbol:
                media_symbol = self.d_media_token2symbol[t]
                media = d_media[media_symbol][d_media_counter[media_symbol]]
                d_media_counter[media_symbol] += 1
                tokens[i] = media
                media["to_predict"] = labels[i] > 0

        for key in d_media:
            if d_media_counter[key] != len(d_media[key]):
                raise AssertionError(f"{d_media_counter[key]} {key} exists in text, but {len(d_media[key])} actual media are given")
        return tokens, labels

    @staticmethod
    def insert_implicit_media_symbol_in_q1(conv_list: List[Dict], d_media: Dict):
        """
        Add the media tokens to the beginning of the first instruction from
        human. This logic may be more reasonable. However, it is incompatible
        with old-version Accessory models, which are trained with image tokens
        inserted directly behind the first token (<bos>).
        :param conv_list: [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}, ...]
        :param d_media: a dict of media for all media types
        """
        conv_list = copy.deepcopy(conv_list)

        for media_symbol, l_media in d_media.items():
            media_symbol_count = "".join([_["value"] for _ in conv_list if _["value"] is not None]).count(media_symbol)
            if media_symbol_count > 0:
                if media_symbol_count != len(l_media):
                    raise AssertionError(f"{media_symbol_count} {media_symbol} exists in text, but {len(l_media)} actual media are given")
            else:
                conv_list[0]["value"] = (media_symbol + " ") * len(l_media) + conv_list[0]["value"]

        return conv_list

    def add_speaker_and_signal(self, source: List):
        """
        Given source instruction and response pieces, return the text containing the complete conversation,
        and the list of values that the model should learn to predict during training
        :param source: [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}, ...]
        :return: `conversation`: string containing the complete conversation;
                 `to_predict_list`: the list of values that the model should learn to predict during training
        """
        conv = get_conv_template("lumina-mgpt2")

        for i, sentence in enumerate(source):
            from_str = sentence["from"].lower()
            if i % 2 == 0 and from_str in ["human"]:
                role = conv.roles[0]
            elif i % 2 == 1 and from_str in ["gpt", "assistant"]:
                role = conv.roles[1]
            else:
                raise ValueError(f"unknown dialog role: {from_str.lower()}")

            value = sentence["value"]
            conv.append_message(role, value)

        processed = conv.get_prompt()
        conversation, pieces = processed["conv"], processed["pieces"]
        return conversation, pieces

    def preprocess_item(self, raw_item):
        # Add custom codes here to convert raw_item to the standard format
        # The standard format contains the "conversations" and "image" keys
        # The data format contains the "file" and "prompt" keys
        # ********* <start>  Add your custom codes here *******
        if "file" in raw_item and os.path.isfile(raw_item["file"]):
            img_path = raw_item["file"]
            with Image.open(img_path) as img:
                image = img.copy()
        else:
            raise ValueError(f"No 'file' key found in {raw_item} or the file does not exist.")

        if "prompt" in raw_item:
            caption = raw_item["prompt"]
        else:
            raise ValueError(f"No 'prompt' key found in {raw_item}.")

        image = var_center_crop(image, crop_size_list=self.crop_size_list)

        if random.random() < 0.9:
            prompt = f"Generate an image of {image.size[0]}x{image.size[1]} according to the following prompt:\n{caption}"  # noqa
        else:
            prompt = f"Generate an image according to the following prompt:\n{caption}"

        raw_item["conversations"] = [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": "<|image|>"},
        ]
        # *********  <end>   Add your custom codes here *******

        item = {
            "conversations": raw_item["conversations"],
            "image": img_path,
        }
        return item

    def predict_item_token_length(self, data_item: dict) -> int:
        """
        estimate the length of each item
        """

        if "conversations" in data_item:
            return sum([len(_["value"]) for _ in data_item["conversations"]])
        else:
            return 1
