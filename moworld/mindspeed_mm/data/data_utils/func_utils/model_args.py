# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
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

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProcessorArguments:
    r"""
    Arguments pertaining to the image processor.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether or not to use one of the fast tokenizer (backed by the tokenizers library)."},
    )
    split_special_tokens: bool = field(
        default=False,
        metadata={"help": "Whether or not the special tokens should be split during the tokenization process."},
    )
    image_resolution: int = field(
        default=512,
        metadata={"help": "Keeps the height or width of image below this resolution."},
    )
    video_resolution: int = field(
        default=128,
        metadata={"help": "Keeps the height or width of video below this resolution."},
    )
    image_max_pixels: int = field(
        default=768 * 768,
        metadata={"help": "The maximum number of pixels of image inputs."},
    )
    image_min_pixels: int = field(
        default=32 * 32,
        metadata={"help": "The minimum number of pixels of image inputs."},
    )
    video_max_pixels: int = field(
        default=256 * 256,
        metadata={"help": "The maximum number of pixels of video inputs."},
    )
    video_min_pixels: int = field(
        default=16 * 16,
        metadata={"help": "The minimum number of pixels of video inputs."},
    )
    video_fps: float = field(
        default=2.0,
        metadata={"help": "The frames to sample per second for video inputs."},
    )
    video_maxlen: int = field(
        default=128,
        metadata={"help": "The maximum number of sampled frames for video inputs."},
    )
    image_do_pan_and_scan: bool = field(
        default=False,
        metadata={"help": "Use pan and scan to process image for gemma3."},
    )
    crop_to_patches: bool = field(
        default=False,
        metadata={"help": "Whether to crop the image to patches for internvl."},
    )
    use_audio_in_video: bool = field(
        default=False,
        metadata={"help": "Whether or not to use audio in video inputs."},
    )
    audio_sampling_rate: int = field(
        default=16000,
        metadata={"help": "The sampling rate of audio inputs."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether or not to allow for custom models defined on the Hub in their own modeling files."},
    )
    fix_mistral_regex: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to fix the Mistral tokenizer's regex issue (specifically for Mistral/Mixtral models with incorrect spaces)."},
    )
