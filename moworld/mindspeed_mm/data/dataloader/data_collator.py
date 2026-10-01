from dataclasses import dataclass
from typing import Dict, Sequence, List, Union, Tuple, Any
import math
from collections import Counter
import random
import warnings

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import WhisperProcessor, DataCollatorForLanguageModeling

from megatron.training import get_args
from mindspeed_mm.data.data_utils.constants import MODEL_CONSTANTS
from mindspeed_mm.data.data_utils.func_utils.collator import MultiModalDataCollatorForSeq2Seq, \
    PairwiseDataCollatorWithPadding
from mindspeed_mm.data.data_utils.func_utils.convert import load_tokenizer, IGNORE_INDEX
from mindspeed_mm.data.data_utils.func_utils.model_args import ProcessorArguments
from mindspeed_mm.data.data_utils.func_utils.template import get_template_and_fix_tokenizer
from mindspeed_mm.data.data_utils.utils import get_value_from_args
from mindspeed_mm.data.data_utils.constants import (
    PROMPT_IDS,
    PROMPT_MASK,
    VIDEO,
    VIDEO_MASK,
    MASKED_VIDEO,
    INPUT_MASK,
    FILE_INFO
)


class DataCollatorForInternvl(object):
    def __init__(self, pad_id, **kwargs):
        self.pad_id = pad_id
        self.ignore_index = MODEL_CONSTANTS['internvl']['IGNORE_INDEX']

    def __call__(self, features):
        first = features[0]
        batch = {}

        batch_lens = [feat["input_ids"].shape for feat in features]
        max_item_length = max(batch_lens)[0]
        for feat in features:
            temp_input_ids = torch.LongTensor([self.pad_id] * max_item_length)
            temp_input_ids[:feat["input_ids"].shape[0]] = feat["input_ids"]
            feat["input_ids"] = temp_input_ids
            temp_labels = torch.LongTensor([self.ignore_index] * max_item_length)
            temp_labels[:feat["labels"].shape[0]] = feat["labels"]
            feat["labels"] = temp_labels
            feat["attention_mask"] = feat["input_ids"].ne(self.pad_id)

        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let"s make sure of it.)
        if "label" in first and first["label"] is not None:
            label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
        elif "label_ids" in first and first["label_ids"] is not None:
            if isinstance(first["label_ids"], torch.Tensor):
                batch["labels"] = torch.stack([f["label_ids"] for f in features])
            else:
                dtype = torch.long if isinstance(first["label_ids"][0], int) else torch.float
                batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

        # Handling of all other possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in first.items():
            if k not in ("label", "label_ids", "pixel_values", "image_flags") and \
                    v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
            if k in ("pixel_values", "image_flags"):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.concat([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.concat(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.concat([f[k] for f in features])
        return batch


class DataCollatorForDeepSeekVL(object):
    def __init__(self, pad_id, **kwargs):
        self.pad_id = pad_id
        self.ignore_id = MODEL_CONSTANTS["deepseekvl2"]["IGNORE_INDEX"]

    def __call__(self, sample_list):
        batched_input_ids = [sample["input_ids"] for sample in sample_list]
        batched_labels = [sample["labels"] for sample in sample_list]
        batched_images_seq_mask = [sample["images_seq_mask"] for sample in sample_list]
        seq_lens = [len(sample) for sample in sample_list]

        """padding input_ids and images_seq_mask"""
        batched_input_ids = pad_sequence(batched_input_ids, batch_first=True, padding_value=self.pad_id)
        batched_labels = pad_sequence(batched_labels, batch_first=True, padding_value=self.ignore_id)
        batched_images_seq_mask = pad_sequence(batched_images_seq_mask, batch_first=True, padding_value=0)
        batched_attention_mask = batched_input_ids != self.pad_id

        """padding images to max_patch_num"""
        max_n_patches = max(sample["images"].shape[0] for sample in sample_list)
        batched_images = []
        for sample in sample_list:
            images = sample["images"]
            n_pads = max_n_patches - images.shape[0]
            if n_pads > 0:
                pad_images = torch.zeros((n_pads, *images.shape[1:]), dtype=images.dtype)
                images = torch.cat([images, pad_images], dim=0)
            batched_images.append(images)
        batched_images = torch.stack(batched_images, dim=0)

        """padding images_spatial_crop to max_n_images"""
        max_n_images = max(sample["images_spatial_crop"].shape[0] for sample in sample_list)
        batched_images_spatial_crop = []
        for sample in sample_list:
            images_spatial_crop = sample["images_spatial_crop"]
            n_pads = max_n_images - sample["images_spatial_crop"].shape[0]
            if n_pads > 0:
                pad_images_spatial_crop = torch.full((n_pads, 2), 0, dtype=images_spatial_crop.dtype)
                images_spatial_crop = torch.cat([images_spatial_crop, pad_images_spatial_crop], dim=0)
            batched_images_spatial_crop.append(images_spatial_crop)
        batched_images_spatial_crop = torch.stack(batched_images_spatial_crop, dim=0)

        return {
            "input_ids": batched_input_ids,
            "labels": batched_labels,
            "attention_mask": batched_attention_mask,
            "images": batched_images,
            "images_seq_mask": batched_images_seq_mask,
            "images_spatial_crop": batched_images_spatial_crop
        }


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor_name_or_path, language, task, **kwargs):
        self.processor = WhisperProcessor.from_pretrained(
            processor_name_or_path,
            language=language,
            task=task,
            local_files_only=True,
        )

    def __call__(
            self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"]}
            for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


class DataCollatorForQwen2vl:
    def __init__(self, ignore_pad_token_for_loss: bool, dataset_param=None, **kwargs):
        process_args = ProcessorArguments(**dataset_param.preprocess_parameters.to_dict())
        tokenizer_module = load_tokenizer(process_args)
        tokenizer = tokenizer_module.get('tokenizer')
        template = get_template_and_fix_tokenizer(tokenizer, dataset_param.basic_parameters.template)
        self.data_collator = MultiModalDataCollatorForSeq2Seq(
            template=template,
            pad_to_multiple_of=8,  # for shift short attention
            label_pad_token_id=IGNORE_INDEX if ignore_pad_token_for_loss else tokenizer.pad_token_id,
            **tokenizer_module,
        )

    def __call__(self, *args, **kwargs):
        return self.data_collator(*args, **kwargs)


@dataclass
class OmniModalDataCollatorForSeq2Seq(MultiModalDataCollatorForSeq2Seq):
    r"""Omni data collator that adds audio feature before the __call__ method returns
    """

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        features = super().__call__(features)
        features["use_audio_in_video"] = getattr(self.processor, "use_audio_in_video", False)
        return features


class DataCollatorForQwen3Omni:
    def __init__(self, ignore_pad_token_for_loss: bool, dataset_param=None, **kwargs):
        process_args = ProcessorArguments(**dataset_param.preprocess_parameters.to_dict())
        tokenizer_module = load_tokenizer(process_args)
        tokenizer = tokenizer_module.get('tokenizer')
        template = get_template_and_fix_tokenizer(tokenizer, dataset_param.basic_parameters.template)
        self.data_collator = OmniModalDataCollatorForSeq2Seq(
            template=template,
            pad_to_multiple_of=8,  # for shift short attention
            label_pad_token_id=IGNORE_INDEX if ignore_pad_token_for_loss else tokenizer.pad_token_id,
            **tokenizer_module,
        )

    def __call__(self, *args, **kwargs):
        return self.data_collator(*args, **kwargs)


class DataCollatorForLLMPretrain:
    def __init__(self, dataset_param=None, **kwargs):
        if dataset_param is None:
            raise ValueError("dataset_param is required for DataCollatorForLLM Pretrain")
        process_args = ProcessorArguments(**dataset_param.preprocess_parameters.to_dict())
        tokenizer_module = load_tokenizer(process_args)
        tokenizer = tokenizer_module.get('tokenizer')
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def __call__(self, *args, **kwargs):
        return self.data_collator(*args, **kwargs)


class DataCollatorForOpenSoraPlan:
    def __init__(
            self,
            batch_size: int = 1,
            num_frames: int = 13,
            group_frame: bool = False,
            group_resolution: bool = False,
            group_data: bool = False,
            max_height: int = 480,
            max_width: int = 640,
            vae_scale_factor: Tuple[int] = (4, 8, 8),
            use_video_feature: bool = False,
            use_text_feature: bool = False,
            **kwargs
    ):
        self.batch_size = batch_size
        self.group_frame = group_frame
        self.group_resolution = group_resolution
        self.group_data = group_data

        self.max_height = max_height
        self.max_width = max_width
        predictor_model_config = get_value_from_args("mm.model.predictor")
        patch_size_thw = predictor_model_config.patch_size_thw
        self.patch_size = patch_size_thw[1]
        self.patch_size_t = patch_size_thw[0]
        self.ae_stride = vae_scale_factor[1]
        self.ae_stride_t = vae_scale_factor[0]
        self.ae_stride_thw = vae_scale_factor

        self.num_frames = num_frames
        self.max_thw = (self.num_frames, self.max_height, self.max_width)

        self.use_video_feature = use_video_feature
        self.use_text_feature = use_text_feature

    @staticmethod
    def package(batch):
        batch_tubes = [i.get(VIDEO, None) for i in batch]  # b [c t h w]
        file = [i.get(FILE_INFO, None) for i in batch]
        if not isinstance(batch[0].get(PROMPT_IDS, None), list):
            input_ids = [i.get(PROMPT_IDS, None) for i in batch]  # b [1 l]
            cond_mask = [i.get(PROMPT_MASK, None) for i in batch]  # b [1 l]
        else:
            input_ids = list(map(list, zip(*[i[PROMPT_IDS] for i in batch])))
            cond_mask = list(map(list, zip(*[i[PROMPT_MASK] for i in batch])))
        return batch_tubes, input_ids, cond_mask, file

    @staticmethod
    def check_prompt_ids_shape(prompt_ids, is_list):
        if not is_list:
            if prompt_ids.dim() != 2 and prompt_ids.dim() != 3:
                raise ValueError(
                    f"prompt shape must have dim 2 for non featured data or 3 for featured data, but got {prompt_ids.dim()}")
        else:
            if prompt_ids[0].dim() != 2 and prompt_ids[0].dim() != 3:
                raise ValueError(
                    f"prompt shape must have dim 2 for non featured data or 3 for featured data, but got {prompt_ids.dim()}")

    def package_feature(self, batch):
        is_list = isinstance(batch[0].get(PROMPT_IDS, None), list)
        for item in batch:
            if item.get(VIDEO).dim() != 4:
                raise ValueError(f"video shape must have dim 4, but got {item.get(VIDEO).dim()}")

            if item.get(PROMPT_MASK, None) and item.get(PROMPT_MASK).dim() != 2:
                raise ValueError(
                    f"prompt mask must be None or have dim 2 for non featured and featured data, but got {item.get(PROMPT_MASK).dim()}")

            if item.get(VIDEO_MASK, None) and item.get(VIDEO_MASK).dim() != 3:
                raise ValueError(f"video_mask shape must be None or have dim 3, but got {item.get(VIDEO_MASK).dim()}")

            prompt_ids = item.get(PROMPT_IDS)
            self.check_prompt_ids_shape(prompt_ids, is_list)

        batch_tubes = [item.get(VIDEO, None) for item in batch]
        video_mask = [item.get(VIDEO_MASK, None) for item in batch]
        if all([i is None or not any(i) for i in video_mask]):
            video_mask = None

        if not is_list:
            input_ids = [item.get(PROMPT_IDS, None) for item in batch]  # b [1 l]
            cond_mask = [item.get(PROMPT_MASK, None) for item in batch]  # b [1 l]
        elif self.use_text_feature:
            input_ids = [item.get(PROMPT_IDS, None)[0] for item in batch]  # b [1 l]
            cond_mask = [item.get(PROMPT_MASK, None)[0] for item in batch]  # b [1
            warnings.warn("input_ids_2 and cond_mask_2 features are not supported yet and will be None for now",
                          FutureWarning)
        else:
            input_ids = list(map(list, zip(*[item[PROMPT_IDS] for item in batch])))
            cond_mask = list(map(list, zip(*[item[PROMPT_MASK] for item in batch])))

        return batch_tubes, video_mask, input_ids, cond_mask

    def __call__(self, batch):
        if not self.use_video_feature:
            batch_tubes, input_ids, cond_mask, file = self.package(batch)

            ds_stride = self.ae_stride * self.patch_size
            t_ds_stride = self.ae_stride_t * self.patch_size_t

            processed_res = self.process(
                batch_tubes,
                input_ids,
                cond_mask,
                t_ds_stride,
                ds_stride,
                self.max_thw,
                self.ae_stride_thw,
            )
            if torch.any(torch.isnan(processed_res.pad_batch_tubes)):
                raise AssertionError("after pad_batch_tubes.")
            return {
                VIDEO: processed_res.pad_batch_tubes,
                PROMPT_IDS: processed_res.input_ids,
                VIDEO_MASK: processed_res.attention_mask,
                PROMPT_MASK: processed_res.cond_mask,
                MASKED_VIDEO: processed_res.masked_video,
                INPUT_MASK: processed_res.input_mask,
                FILE_INFO: file
            }
        else:
            batch_tubes, video_mask, input_ids, cond_mask = self.package_feature(batch)
            if not isinstance(input_ids[0], list):
                input_ids = torch.stack(input_ids)  # b 1 l
                cond_mask = torch.stack(cond_mask)  # b 1 l
            else:
                input_ids = [
                    torch.stack(_input_ids)  # b 1 l
                    for _input_ids in input_ids
                ]
                cond_mask = [
                    torch.stack(_cond_mask)  # b 1 l
                    for _cond_mask in cond_mask
                ]
            return {
                VIDEO: torch.stack(batch_tubes),
                PROMPT_IDS: input_ids,
                VIDEO_MASK: torch.stack(video_mask) if video_mask else None,
                PROMPT_MASK: cond_mask,
                MASKED_VIDEO: None,
                INPUT_MASK: None,
            }

    def process(
        self,
        batch_tubes,
        input_ids,
        cond_mask,
        t_ds_stride,
        ds_stride,
        max_thw,
        ae_stride_thw,
    ):
        # pad to max multiple of ds_stride
        batch_input_size = [i.shape for i in batch_tubes]  # [(c t h w), (c t h w)]
        if len(batch_input_size) != self.batch_size:
            raise AssertionError("batch_input_size and batch_size are not equal.")

        is_grouped = self.group_frame or self.group_resolution or self.group_data or self.batch_size == 1
        if is_grouped:  #
            len_each_batch = batch_input_size
            idx_length_dict = dict([*zip(list(range(self.batch_size)), len_each_batch)])
            count_dict = Counter(len_each_batch)
            if len(count_dict) != 1:
                sorted_by_value = sorted(count_dict.items(), key=lambda item: item[1])
                pick_length = sorted_by_value[-1][0]  # the highest frequency
                candidate_batch = [
                    idx
                    for idx, length in idx_length_dict.items()
                    if length == pick_length
                ]
                random_select_batch = [
                    random.choice(candidate_batch)
                    for _ in range(len(len_each_batch) - len(candidate_batch))
                ]
                print(
                    batch_input_size,
                    idx_length_dict,
                    count_dict,
                    sorted_by_value,
                    pick_length,
                    candidate_batch,
                    random_select_batch,
                )
                pick_idx = candidate_batch + random_select_batch

                batch_tubes = [batch_tubes[i] for i in pick_idx]
                batch_input_size = [
                    i.shape
                    for i in batch_tubes
                ]  # [(c t h w), (c t h w)]
                if not isinstance(input_ids[0], list):
                    input_ids = [input_ids[i] for i in pick_idx]  # b [1, l]
                    cond_mask = [cond_mask[i] for i in pick_idx]  # b [1, l]
                else:
                    input_ids = [
                        [_input_ids[i] for i in pick_idx]  # b [1, l]
                        for _input_ids in input_ids
                    ]
                    cond_mask = [
                        [_cond_mask[i] for i in pick_idx]  # b [1, l]
                        for _cond_mask in cond_mask
                    ]

            for i in range(1, self.batch_size):
                if batch_input_size[0] != batch_input_size[i]:
                    raise AssertionError(
                        f"batch_input_size{0} and batch_input_size{i} are not equal."
                    )
            max_t = max([i[1] for i in batch_input_size])
            max_h = max([i[2] for i in batch_input_size])
            max_w = max([i[3] for i in batch_input_size])
        else:
            max_t, max_h, max_w = max_thw
        pad_max_t, pad_max_h, pad_max_w = (
            self.pad_to_multiple(max_t - 1 + self.ae_stride_t, t_ds_stride),
            self.pad_to_multiple(max_h, ds_stride),
            self.pad_to_multiple(max_w, ds_stride),
        )
        pad_max_t = pad_max_t + 1 - self.ae_stride_t
        each_pad_t_h_w = [
            [pad_max_t - i.shape[1], pad_max_h - i.shape[2], pad_max_w - i.shape[3]]
            for i in batch_tubes
        ]
        pad_batch_tubes = [
            F.pad(im, (0, pad_w, 0, pad_h, 0, pad_t), value=0)
            for (pad_t, pad_h, pad_w), im in zip(each_pad_t_h_w, batch_tubes)
        ]
        pad_batch_tubes = torch.stack(pad_batch_tubes, dim=0)

        max_tube_size = [pad_max_t, pad_max_h, pad_max_w]
        max_latent_size = [
            ((max_tube_size[0] - 1) // ae_stride_thw[0] + 1),
            max_tube_size[1] // ae_stride_thw[1],
            max_tube_size[2] // ae_stride_thw[2],
        ]
        valid_latent_size = [
            [
                int(math.ceil((i[1] - 1) / ae_stride_thw[0])) + 1,
                int(math.ceil(i[2] / ae_stride_thw[1])),
                int(math.ceil(i[3] / ae_stride_thw[2])),
            ]
            for i in batch_input_size
        ]
        attention_mask = [
            F.pad(
                torch.ones(i, dtype=pad_batch_tubes.dtype),
                (
                    0,
                    max_latent_size[2] - i[2],
                    0,
                    max_latent_size[1] - i[1],
                    0,
                    max_latent_size[0] - i[0],
                ),
                value=0,
            )
            for i in valid_latent_size
        ]
        attention_mask = torch.stack(attention_mask)  # b t h w
        if self.batch_size == 1 or self.group_frame or self.group_resolution:
            if not torch.all(attention_mask.bool()):
                raise AssertionError("All elements of attention_mask are zero")

        if not isinstance(input_ids[0], list):
            input_ids = torch.stack(input_ids)  # b 1 l
            cond_mask = torch.stack(cond_mask)  # b 1 l
        else:
            input_ids = [
                torch.stack(_input_ids)  # b 1 l
                for _input_ids in input_ids
            ]
            cond_mask = [
                torch.stack(_cond_mask)  # b 1 l
                for _cond_mask in cond_mask
            ]

        # if opensoraplan i2v dataset, batch_tube has masked_video and mask
        if pad_batch_tubes.shape[1] == 7:
            pad_batch_tubes, masked_video, input_mask = pad_batch_tubes[:, :3], pad_batch_tubes[:,
                                                                                3:6], pad_batch_tubes[:, 6:7]
        else:
            masked_video = None
            input_mask = None

        processed_res = ProcessedData(pad_batch_tubes, attention_mask, input_ids, cond_mask,
                                      masked_video, input_mask)
        return processed_res

    @staticmethod
    def pad_to_multiple(number, ds_stride):
        remainder = number % ds_stride
        if remainder == 0:
            return number
        else:
            padding = ds_stride - remainder
            return number + padding


class ProcessedData:
    def __init__(self, pad_batch_tubes, attention_mask, input_ids, cond_mask, masked_video,
                 input_mask):
        self.pad_batch_tubes = pad_batch_tubes
        self.attention_mask = attention_mask
        self.input_ids = input_ids
        self.cond_mask = cond_mask
        self.masked_video = masked_video
        self.input_mask = input_mask


class DataCollatorForQwen2vlDPO:
    def __init__(self, ignore_pad_token_for_loss: bool, dataset_param=None, **kwargs):
        process_args = ProcessorArguments(**dataset_param.preprocess_parameters.to_dict())
        tokenizer_module = load_tokenizer(process_args)
        tokenizer = tokenizer_module.get('tokenizer')
        template = get_template_and_fix_tokenizer(tokenizer, dataset_param.basic_parameters.template)
        self.data_collator = PairwiseDataCollatorWithPadding(
            template=template,
            pad_to_multiple_of=8,  # for shift short attention
            label_pad_token_id=IGNORE_INDEX if ignore_pad_token_for_loss else tokenizer.pad_token_id,
            **tokenizer_module,
        )

    def __call__(self, *args, **kwargs):
        return self.data_collator(*args, **kwargs)


class DataCollatorForVideoAlign:
    def __init__(self, dataset_param, **kwargs):
        from mindspeed_mm.data.data_utils.func_utils.convert import reward_setting_processor, VideoRewardProcessor
        if not isinstance(dataset_param, dict):
            dataset_param = dataset_param.to_dict()
        preprocess_args_dict = dataset_param["preprocess_parameters"]

        video_reader, video_processor, tokenizer, processor, model_args = reward_setting_processor(preprocess_args_dict)

        self.reward_processor = VideoRewardProcessor(None, tokenizer, processor, dataset_param, video_reader,
                                                     video_processor)

    def __call__(self, examples):
        process_data = self.reward_processor.preprocess_dataset(examples)

        model_inputs = {
            'input_ids_A': process_data['A'][0]['input_ids'],
            'attention_mask_A': process_data['A'][0]['attention_mask'],
            'pixel_values_A': process_data['A'][0]['pixel_values_videos'],
            'image_grid_thw_A': process_data['A'][0]['video_grid_thw'],
            'input_ids_B': process_data['B'][0]['input_ids'],
            'attention_mask_B': process_data['B'][0]['attention_mask'],
            'pixel_values_B': process_data['B'][0]['pixel_values_videos'],
            'image_grid_thw_B': process_data['B'][0]['video_grid_thw'],
            'A_scores': process_data['A_scores'],
            'B_scores': process_data['B_scores'],
            'chosen_label': process_data['chosen_label']
        }
        return model_inputs


class DataCollatorForBagel:
    def __init__(self, batch=None):
        if batch is not None:
            data = batch[0]
            self.batch_data_indexes = data['batch_data_indexes']
            self.sequence_length = data["sequence_length"]
            self.sample_lens = data["sample_lens"]
            self.packed_text_ids = data["packed_text_ids"]
            self.packed_text_indexes = data["packed_text_indexes"]
            self.packed_position_ids = data["packed_position_ids"]

            self.use_flex = "nested_attention_masks" not in data.keys()

            if self.use_flex:
                self.split_lens = data["split_lens"]
                self.attn_modes = data["attn_modes"]
            else:
                self.nested_attention_masks = data["nested_attention_masks"]

            if "padded_images" in data.keys():
                self.padded_images = data["padded_images"]
                self.patchified_vae_latent_shapes = data["patchified_vae_latent_shapes"]
                self.packed_latent_position_ids = data["packed_latent_position_ids"]
                self.packed_vae_token_indexes = data["packed_vae_token_indexes"]

            if "packed_vit_tokens" in data.keys():
                self.packed_vit_tokens = data["packed_vit_tokens"]
                self.packed_vit_position_ids = data["packed_vit_position_ids"]
                self.packed_vit_token_indexes = data["packed_vit_token_indexes"]
                self.vit_token_seqlens = data["vit_token_seqlens"]

            if "packed_timesteps" in data.keys():
                self.packed_timesteps = data["packed_timesteps"]
                self.mse_loss_indexes = data["mse_loss_indexes"]

            if "packed_label_ids" in data.keys():
                self.packed_label_ids = data["packed_label_ids"]
                self.ce_loss_indexes = data["ce_loss_indexes"]
                self.ce_loss_weights = data["ce_loss_weights"]

    def __call__(self, batch):
        return DataCollatorForBagel(batch)

    def to_dict(self):
        data = dict(
            sequence_length=self.sequence_length,
            sample_lens=self.sample_lens,
            packed_text_ids=self.packed_text_ids,
            packed_text_indexes=self.packed_text_indexes,
            packed_position_ids=self.packed_position_ids,
            batch_data_indexes=self.batch_data_indexes,
        )

        if not self.use_flex:
            data['nested_attention_masks'] = self.nested_attention_masks
        else:
            data['split_lens'] = self.split_lens
            data['attn_modes'] = self.attn_modes

        if hasattr(self, 'padded_images'):
            data['padded_images'] = self.padded_images
            data['patchified_vae_latent_shapes'] = self.patchified_vae_latent_shapes
            data['packed_latent_position_ids'] = self.packed_latent_position_ids
            data['packed_vae_token_indexes'] = self.packed_vae_token_indexes

        if hasattr(self, 'packed_vit_tokens'):
            data['packed_vit_tokens'] = self.packed_vit_tokens
            data['packed_vit_position_ids'] = self.packed_vit_position_ids
            data['packed_vit_token_indexes'] = self.packed_vit_token_indexes
            data['vit_token_seqlens'] = self.vit_token_seqlens

        if hasattr(self, 'packed_timesteps'):
            data['packed_timesteps'] = self.packed_timesteps
            data['mse_loss_indexes'] = self.mse_loss_indexes

        if hasattr(self, 'packed_label_ids'):
            data['packed_label_ids'] = self.packed_label_ids
            data['ce_loss_indexes'] = self.ce_loss_indexes
            data['ce_loss_weights'] = self.ce_loss_weights

        return data

DATA_COLLATOR = {
    "internvl": DataCollatorForInternvl,
    "whisper": DataCollatorSpeechSeq2SeqWithPadding,
    "qwen2vl": DataCollatorForQwen2vl,
    "qwen2vl_dpo": DataCollatorForQwen2vlDPO,
    "qwen3vl": DataCollatorForQwen2vl,
    "qwen3omni": DataCollatorForQwen3Omni,
    "llm_pretrain": DataCollatorForLLMPretrain,
    "open_sora_plan": DataCollatorForOpenSoraPlan,
    "deepseekvl2": DataCollatorForDeepSeekVL,
    "videoalign": DataCollatorForVideoAlign,
    "glm4v_moe": DataCollatorForQwen2vl,
    "bagel": DataCollatorForBagel
}
