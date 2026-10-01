# Copyright 2024 OpenAccess AI Collective and the LlamaFactory team.
#
# This code is inspired by the OpenAccess AI Collective's axolotl library.
# https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/monkeypatch/utils.py
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

import inspect
from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from peft import PeftModel
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import DataCollatorForSeq2Seq, ProcessorMixin
from .mm_plugin import IGNORE_INDEX, IMAGE_PLACEHOLDER, AUDIO_PLACEHOLDER
from .template import Template


def postprocess_position_ids(new_postion_ids, packed_postion_ids):
    result = []
    for row_id, flat_tensor in enumerate(packed_postion_ids.unbind(0)):
        row_position_ids = new_postion_ids[0, row_id, ...]
        shift_mask = torch.zeros_like(row_position_ids)

        zero_indices = torch.where(flat_tensor == 0)[0]
        if len(zero_indices) == 0:
            raise ValueError("There should be at least one example in each packed data.")
        end_idx = zero_indices[-1]
        for i in range(len(zero_indices) - 1):
            start_idx, end_idx = zero_indices[i], zero_indices[i + 1]
            shift_mask[start_idx:end_idx] = row_position_ids[start_idx]
        shift_mask[end_idx:] = row_position_ids[end_idx]
        result.append(new_postion_ids[:, row_id, :] - shift_mask)
    return torch.stack(result).transpose(0, 1)


@dataclass
class MultiModalDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    r"""Data collator that supports VLMs.

    Features should contain input_ids, attention_mask, labels, and optionally contain images, videos and audios.
    """

    template: Optional["Template"] = None
    processor: Optional["ProcessorMixin"] = None

    def __post_init__(self):
        if self.template is None:
            raise ValueError("Template is required for MultiModalDataCollator.")

        # Background: In single-NPU LoRA training, model is wrapped as DDP(PeftModel(OriginalModel)).
        # Original code only checked isinstance(model, PeftModel) which returns False when wrapped in DDP,
        # causing AttributeError when accessing model.config. Solution: unwrap DDP first, then PeftModel.
        if isinstance(self.model, DDP):
            self.model = self.model.module

        if isinstance(self.model, PeftModel):
            self.model = self.model.base_model.model

        if self.model is not None and hasattr(self.model, "get_rope_index"):  # for qwen2vl mrope
            self.get_rope_func = self.model.get_rope_index  # transformers < 4.52.0 or qwen2.5 omni
        elif self.model is not None and hasattr(self.model, "model") and hasattr(self.model.model, "get_rope_index"):
            self.get_rope_func = self.model.model.get_rope_index  # transformers >= 4.52.0
        else:
            self.get_rope_func = None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        batch_images, batch_videos, batch_audios = [], [], []
        batch_imglens, batch_vidlens, batch_audlens, batch_input_ids = [], [], [], []
        for feature in features:
            images = feature.pop("images", None) or []
            videos = feature.pop("videos", None) or []
            audios = feature.pop("audios", None) or []
            batch_images.extend(images)
            batch_videos.extend(videos)
            batch_audios.extend(audios)
            batch_imglens.append(len(images))
            batch_vidlens.append(len(videos))
            batch_audlens.append(len(audios))
            batch_input_ids.append(feature["input_ids"])

        fake_input_ids = []
        if (
            self.template.mm_plugin.image_token is not None and sum(batch_imglens) == 0 and sum(batch_vidlens) == 0
        ):  # avoid process hanging in zero3/fsdp case
            fake_messages = [{"role": "user", "content": IMAGE_PLACEHOLDER}]
            fake_images = [Image.new("RGB", (64, 64), (255, 255, 255))]
            fake_messages = self.template.mm_plugin.process_messages(
                fake_messages, fake_images, [], [], self.processor
            )
            _fake_input_ids = self.tokenizer.encode(fake_messages[0]["content"], add_special_tokens=False)
            _fake_input_ids, _ = self.template.mm_plugin.process_token_ids(
                _fake_input_ids, None, fake_images, [], [], self.tokenizer, self.processor
            )
            fake_input_ids.extend(_fake_input_ids)
            batch_images = fake_images
            batch_imglens[0] = 1

        if (
            self.template.mm_plugin.audio_token is not None and sum(batch_audlens) == 0
        ):  # avoid process hanging in zero3/fsdp case
            fake_messages = [{"role": "user", "content": AUDIO_PLACEHOLDER}]
            fake_audios = [np.zeros(1600)]
            fake_messages = self.template.mm_plugin.process_messages(
                fake_messages, [], [], fake_audios, self.processor
            )
            _fake_input_ids = self.tokenizer.encode(fake_messages[0]["content"], add_special_tokens=False)
            _fake_input_ids, _ = self.template.mm_plugin.process_token_ids(
                _fake_input_ids, None, [], [], fake_audios, self.tokenizer, self.processor
            )
            fake_input_ids.extend(_fake_input_ids)
            batch_audios = fake_audios
            batch_audlens[0] = 1

        if len(fake_input_ids) != 0:
            if self.tokenizer.padding_side == "right":
                features[0]["input_ids"] = features[0]["input_ids"] + fake_input_ids
                features[0]["attention_mask"] = features[0]["attention_mask"] + [0] * len(fake_input_ids)
                features[0]["labels"] = features[0]["labels"] + [IGNORE_INDEX] * len(fake_input_ids)
                if 'cu_seqlens' in features[0]:
                    features[0]["cu_seqlens"][-1] = features[0]["cu_seqlens"][-1] + len(fake_input_ids)
            else:
                features[0]["input_ids"] = fake_input_ids + features[0]["input_ids"]
                features[0]["attention_mask"] = [0] * len(fake_input_ids) + features[0]["attention_mask"]
                features[0]["labels"] = [IGNORE_INDEX] * len(fake_input_ids) + features[0]["labels"]
                if 'cu_seqlens' in features[0]:
                    features[0]["cu_seqlens"][1] = features[0]["cu_seqlens"][1] + len(fake_input_ids)

            batch_input_ids[0] = features[0]["input_ids"]

        mm_inputs = self.template.mm_plugin.get_mm_inputs(
            batch_images,
            batch_videos,
            batch_audios,
            batch_imglens,
            batch_vidlens,
            batch_audlens,
            batch_input_ids,
            self.processor,
        )
        if "token_type_ids" in mm_inputs:
            token_type_ids = mm_inputs.pop("token_type_ids")
            for i, feature in enumerate(features):
                feature["token_type_ids"] = token_type_ids[i]

        features: dict[str, torch.Tensor] = super().__call__(features)

        packed_postion_ids = None
        if "position_ids" in features:
            pad_size = features["input_ids"].shape[1] - features["position_ids"].shape[1]
            if pad_size == 0:
                packed_postion_ids = features["position_ids"]
            elif pad_size < 0:
                raise ValueError("Position ids length should not be greater than input ids length.")
            elif self.tokenizer.padding_side == "right":
                packed_postion_ids = F.pad(
                    features["position_ids"], (0, pad_size), mode="constant", value=0)
            else:
                packed_postion_ids = F.pad(
                    features["position_ids"], (pad_size, 0), mode="constant", value=0)

        if self.get_rope_func is not None:
            rope_index_kwargs = {
                "input_ids": features["input_ids"],
                "image_grid_thw": mm_inputs.get("image_grid_thw"),
                "video_grid_thw": mm_inputs.get("video_grid_thw"),
                "attention_mask": (features["attention_mask"] >= 1).float(),
            }
            if "mm_token_type_ids" in inspect.signature(self.get_rope_func).parameters:
                image_token_id = getattr(self.model.config, "image_token_id", None)
                video_token_id = getattr(self.model.config, "video_token_id", None)
                if image_token_id is not None or video_token_id is not None:
                    mm_token_type_ids = torch.zeros_like(features["input_ids"])
                    if image_token_id is not None:
                        mm_token_type_ids[features["input_ids"] == image_token_id] = 1
                    if video_token_id is not None:
                        mm_token_type_ids[features["input_ids"] == video_token_id] = 2
                    rope_index_kwargs["mm_token_type_ids"] = mm_token_type_ids
            if "second_per_grid_ts" in mm_inputs:  # for qwen2vl
                rope_index_kwargs["second_per_grid_ts"] = mm_inputs.get("second_per_grid_ts")
            elif "video_second_per_grid" in mm_inputs:  # for qwen2.5 omni
                rope_index_kwargs["second_per_grids"] = mm_inputs.get("video_second_per_grid")

            if getattr(self.model.config, "model_type", None) in ["qwen2_5_omni_thinker", "qwen3_omni_moe_thinker"]:
                rope_index_kwargs["use_audio_in_video"] = getattr(self.processor, "use_audio_in_video", False)
                feature_attention_mask = mm_inputs.get("feature_attention_mask", None)
                if feature_attention_mask is not None:  # refer llamafactory: need to get video image lengths
                    audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
                    rope_index_kwargs["audio_seqlens"] = audio_feature_lengths  # prepare for input

                features["position_ids"], rope_deltas = self.get_rope_func(**rope_index_kwargs)
                features["rope_deltas"] = rope_deltas - (1 - rope_index_kwargs["attention_mask"]).sum(
                    dim=-1
                ).unsqueeze(-1)
            else:  # for qwen vl
                features["position_ids"], features["rope_deltas"] = self.get_rope_func(**rope_index_kwargs)

            # remove shift offset for postion ids
            if packed_postion_ids is not None:
                features["position_ids"] = postprocess_position_ids(features["position_ids"], packed_postion_ids)

        if (
            self.model is not None
            and getattr(self.model.config, "model_type", None)
            in [
                "glm4v",
                "glm_ocr",
                "Keye",
                "qwen2_vl",
                "qwen2_5_vl",
                "qwen2_5_omni_thinker",
                "qwen3_omni_moe_thinker",
                "qwen3_5",
                "qwen3_vl",
                "qwen3_vl_moe",
            ]
            and ("position_ids" not in features or features["position_ids"].dim() != 3)
        ):
            raise ValueError(f"{self.model.config.model_type} requires 3D position ids for mrope.")

        if "cross_attention_mask" in mm_inputs:  # for mllama inputs when pad_to_multiple_of is enabled
            cross_attention_mask = mm_inputs.pop("cross_attention_mask")
            seq_len = features["input_ids"].size(1)
            orig_len = cross_attention_mask.size(1)
            mm_inputs["cross_attention_mask"] = F.pad(cross_attention_mask, (0, 0, 0, 0, 0, seq_len - orig_len))

        features.update(mm_inputs)

        if "image_bound" in features:  # for minicpmv inputs
            bsz, seq_length = features["input_ids"].shape
            features["position_ids"] = torch.arange(seq_length).long().repeat(bsz, 1)
            return {"data": features, "input_ids": features["input_ids"], "labels": features["labels"]}

        return features


@dataclass
class PairwiseDataCollatorWithPadding(MultiModalDataCollatorForSeq2Seq):
    r"""
    Data collator for pairwise data.
    """

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        r"""
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        concatenated_features = []
        for key in ("chosen", "rejected"):
            for feature in features:
                target_feature = {
                    "input_ids": feature[f"{key}_input_ids"],
                    "attention_mask": feature[f"{key}_attention_mask"],
                    "labels": feature[f"{key}_labels"],
                    "images": feature["images"],
                    "videos": feature["videos"],
                }
                concatenated_features.append(target_feature)

        return super().__call__(concatenated_features)
