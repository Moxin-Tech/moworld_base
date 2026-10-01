# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
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
import json
import os
from pathlib import Path
import shutil

import torch
from safetensors.torch import save_file

from checkpoint.common.converter import DcpConverter
from checkpoint.common.permissions import set_directory_permissions
from checkpoint.common.merge_dcp_to_hf import load_dcp_state_dict, merge_dcp_to_hf


class Qwen3TTSConverter(DcpConverter):
    """
    A utility class to convert model checkpoints of Qwen3TTS between different formats,
    specifically between Hugging Face (HF) and torch-dcp (DCP) formats.

    Supports:
    - HF → DCP conversion
    - DCP → HF merging
    - Placeholder methods for megatron format and resharding operations.
    """

    dcp_prefix = "model."
    hf_prefix = ""

    def dcp_to_hf(
        self,
        load_dir: str = "mm_save_dir/release",     # Input: Directory containing DCP shards
        save_dir: str = "Qwen3-TTS-12Hz-1.7B-hf",         # Output: Directory to save merged HF model
        model_assets_dir: str = "Qwen3-TTS-12Hz-1.7B-Base",     # Reference: Original HF model dir (for config/tokenizer)
        speaker_name: str = None,
        speaker_audio_path: str = None
    ):
        """
        Merges torch-dcp shards and converts them back into standard Hugging Face format.

        This is typically used after training or inference in torch-dcp format to export
        a model that can be easily loaded with Hugging Face Transformers.
        """
        state_dict = load_dcp_state_dict(load_dir)
        shutil.copytree(model_assets_dir, save_dir, dirs_exist_ok=True)


        input_config_file = os.path.join(model_assets_dir, "config.json")
        output_config_file = os.path.join(save_dir, "config.json")
        with open(input_config_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        talker_config = config_dict.get("talker_config", {})


        if speaker_name is not None and speaker_audio_path is not None:
            import librosa
            import numpy as np
            from mindspeed_mm.fsdp.models.qwen3tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig
            from mindspeed_mm.fsdp.models.qwen3tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder
            from mindspeed_mm.fsdp.models.qwen3tts.core.models.modeling_qwen3_tts import mel_spectrogram

            weight = state_dict["talker.model.codec_embedding.weight"]
            _speaker_encoder_config = Qwen3TTSSpeakerEncoderConfig(**config_dict.get("speaker_encoder_config"))
            _speaker_encoder = Qwen3TTSSpeakerEncoder(_speaker_encoder_config)
            _speaker_encoder_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("speaker_encoder"):
                    _speaker_encoder_state_dict[key.removeprefix("speaker_encoder.")] = value

            _speaker_encoder.load_state_dict(_speaker_encoder_state_dict)

            audio, sr = librosa.load(speaker_audio_path, sr=None, mono=True)
            audio, sr = audio.astype(np.float32), int(sr)
            mels = mel_spectrogram(
                torch.from_numpy(audio).unsqueeze(0),
                n_fft=1024,
                num_mels=128,
                sampling_rate=24000,
                hop_size=256,
                win_size=1024,
                fmin=0,
                fmax=12000
            ).transpose(1, 2)
            state_dict["talker.model.codec_embedding.weight"][3000] = _speaker_encoder(mels).to(weight.dtype)

            config_dict["tts_model_type"] = "custom_voice"
            talker_config["spk_id"] = {
                speaker_name: 3000
            }
            talker_config["spk_is_dialect"] = {
                speaker_name: False
            }
            drop_prefix = "speaker_encoder"
            keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
            for k in keys_to_drop:
                del state_dict[k]

        config_dict["talker_config"] = talker_config
        with open(output_config_file, 'w', encoding='utf-8') as f:
            # 落盘json文件使用2空格进行缩进，并保持非ASCII原始字符不转义
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        save_path = os.path.join(save_dir, "model.safetensors")
        save_file(state_dict, save_path)
        set_directory_permissions(Path(save_path))


    @staticmethod
    def hf_to_dcp():
        pass
