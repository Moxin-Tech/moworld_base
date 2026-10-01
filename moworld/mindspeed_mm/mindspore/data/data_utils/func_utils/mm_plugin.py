# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

import os
from copy import deepcopy
from typing import Optional, List, Dict
import torch

VIDEO_PLACEHOLDER = os.getenv("VIDEO_PLACEHOLDER", "<video>")
IMAGE_PLACEHOLDER = os.getenv("IMAGE_PLACEHOLDER", "<image>")
AUDIO_PLACEHOLDER = os.getenv("AUDIO_PLACEHOLDER", "<audio>")


def process_messages(
            self,
            messages: List[Dict[str, str]],
            images: List["ImageInput"],
            videos: List["VideoInput"],
            audios: List["AudioInput"],
            processor: Optional["MMProcessor"],
    ) -> List[Dict[str, str]]:
    """
    Change ".sum(-1).numpy()" to ".numpy().sum(-1)" to avoid the dataset processing hang issue with Qwen2.5-Omni.
    """
    self._validate_input(processor, images, videos, audios)
    self._validate_messages(messages, images, videos, audios)
    num_image_tokens, num_video_tokens, num_audio_tokens = 0, 0, 0
    messages = deepcopy(messages)
    image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)

    merge_length = processor.image_processor.merge_size ** 2
    use_audio_in_video = getattr(processor, "use_audio_in_video", False)
    if self.expand_mm_tokens:
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
        image_grid_thw = mm_inputs.get("image_grid_thw", [])
        video_grid_thw = mm_inputs.get("video_grid_thw", [])
        if "feature_attention_mask" in mm_inputs:
            # Change ".sum(-1).numpy()" to ".numpy().sum(-1)" to avoid the dataset processing hang issue.
            input_lengths = (mm_inputs["feature_attention_mask"].numpy().sum(-1) - 1) // 2 + 1
            audio_lengths = (input_lengths - 2) // 2 + 1
    else:
        mm_inputs = {}
        image_grid_thw = [None] * len(images)
        video_grid_thw = [None] * len(videos)
        audio_lengths = [None] * len(audios)
    for message in messages:
        content = message["content"]
        while IMAGE_PLACEHOLDER in content:
            image_seqlen = image_grid_thw[num_image_tokens].prod() // merge_length if self.expand_mm_tokens else 1
            content = content.replace(
                IMAGE_PLACEHOLDER, f"<|vision_bos|>{self.image_token * image_seqlen}<|vision_eos|>", 1
            )
            num_image_tokens += 1
        if (
                use_audio_in_video and len(audios) and len(videos)
        ):  # if use the audio of video # deal video token and audio token togather
            if len(videos) != len(audios):
                raise ValueError(
                    f"Number of videos ({len(videos)}) must match number of audios ({len(audios)}) when using audio in video."
                )

            while VIDEO_PLACEHOLDER in content:
                video_pos = content.find(VIDEO_PLACEHOLDER)
                audio_pos = content.find(AUDIO_PLACEHOLDER, video_pos)
                if audio_pos == -1 or audio_pos < video_pos:
                    raise ValueError(
                        f"Each {VIDEO_PLACEHOLDER} must be followed by an {AUDIO_PLACEHOLDER} when using audio in video."
                    )

                audio_t_index = torch.arange(audio_lengths[num_audio_tokens])
                video_t_index = (
                        torch.arange(video_grid_thw[num_video_tokens][0])
                        .view(-1, 1, 1)
                        .expand(
                            -1,
                            video_grid_thw[num_video_tokens][1] // image_processor.merge_size,
                            video_grid_thw[num_video_tokens][2] // image_processor.merge_size,
                        )
                        .flatten()
                        * mm_inputs.get("video_second_per_grid").get(num_video_tokens)
                        * 25
                ).long()
                t_ntoken_per_chunk = 50
                video_chunk_indices = processor.get_chunked_index(video_t_index, t_ntoken_per_chunk)
                audio_chunk_indices = processor.get_chunked_index(audio_t_index, t_ntoken_per_chunk)
                placeholder_string = ""
                placeholder_string += "<|vision_bos|>" + "<|audio_bos|>"
                for j in range(max(len(video_chunk_indices), len(audio_chunk_indices))):
                    video_chunk_index = video_chunk_indices[j] if j < len(video_chunk_indices) else None
                    audio_chunk_index = audio_chunk_indices[j] if j < len(audio_chunk_indices) else None
                    if video_chunk_index is not None:
                        placeholder_string += self.video_token * (video_chunk_index[1] - video_chunk_index[0])

                    if audio_chunk_index is not None:
                        placeholder_string += self.audio_token * (audio_chunk_index[1] - audio_chunk_index[0])

                placeholder_string += "<|audio_eos|>" + "<|vision_eos|>"
                content = content.replace(VIDEO_PLACEHOLDER, placeholder_string, 1)
                content = content.replace(AUDIO_PLACEHOLDER, "", 1)
                num_audio_tokens += 1
                num_video_tokens += 1
        else:
            while AUDIO_PLACEHOLDER in content:
                audio_seqlen = audio_lengths[num_audio_tokens] if self.expand_mm_tokens else 1
                content = content.replace(
                    AUDIO_PLACEHOLDER, f"<|audio_bos|>{self.audio_token * audio_seqlen}<|audio_eos|>", 1
                )
                num_audio_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                video_seqlen = (
                    video_grid_thw[num_video_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                )
                content = content.replace(
                    VIDEO_PLACEHOLDER, f"<|vision_bos|>{self.video_token * video_seqlen}<|vision_eos|>", 1
                )
                num_video_tokens += 1

        message["content"] = content
    return messages
