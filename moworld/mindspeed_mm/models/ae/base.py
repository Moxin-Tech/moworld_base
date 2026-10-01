# Copyright (c) 2024 Huawei Technologies Co., Ltd.
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
from collections import defaultdict

import torch.nn as nn

from mindspeed_mm.utils.extra_processor.i2v_processors import I2VProcessor
from mindspeed_mm.models.ae.diffusers_ae_model import DiffusersAEModel
from mindspeed_mm.models.ae.casualvae import CausalVAE
from mindspeed_mm.models.ae.wfvae import WFVAE
from mindspeed_mm.models.ae.contextparallel_causalvae import ContextParallelCasualVAE
from mindspeed_mm.models.ae.autoencoder_kl_hunyuanvideo import AutoencoderKLHunyuanVideo
from mindspeed_mm.models.ae.wan_video_vae import WanVideoVAE
from mindspeed_mm.models.ae.stepvideo_vae import StepVideoVae
from mindspeed_mm.models.ae.movqvae import MOVQ
from mindspeed_mm.models.ae.flux_vae import FluxVae
from mindspeed_mm.models.ae.hunyuanvideo_15_vae import AutoencoderKLConv3D


AE_MODEL_MAPPINGS = {
    "casualvae": CausalVAE,
    "wfvae": WFVAE,
    "contextparallelcasualvae": ContextParallelCasualVAE,
    "autoencoder_kl_hunyuanvideo": AutoencoderKLHunyuanVideo,
    "wan_video_vae": WanVideoVAE,
    "stepvideovae": StepVideoVae,
    "movqvae": MOVQ,
    "flux_vae": FluxVae,
    "hunyuanvideo_15_vae": AutoencoderKLConv3D
}


class AEModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config.to_dict()

        self.i2v_processor_config = self.config.pop("i2v_processor", None)
        if self.i2v_processor_config is not None:
            self.i2v_processor = I2VProcessor(self.i2v_processor_config).get_processor()
        else:
            self.i2v_processor = None

        if config.model_id in AE_MODEL_MAPPINGS:
            self.model = AE_MODEL_MAPPINGS[config.model_id](**self.config)
        else:
            # import auto encoder from diffusers
            self.model = DiffusersAEModel(
                model_name=config.model_id, config=self.config
            )

    def get_model(self):
        return self.model

    def encode(self, x, **kwargs):
        if not isinstance(x, (list, tuple)):
            return self._single_encode(x, **kwargs)

        video_latents = []
        i2v_results = defaultdict(list)

        for i, _ in enumerate(x):
            kwargs_i = {
                key: value[i] if isinstance(value, (list, tuple)) and len(value) == len(x) else value
                for key, value in kwargs.items()
            }
            _video_latents, _i2v_results = self._single_encode(x[i], **kwargs_i)
            video_latents.append(_video_latents)
            if _i2v_results is None:
                continue
            for key, value in _i2v_results.items():
                i2v_results[key].append(value)

        return video_latents, i2v_results

    def _single_encode(self, x, **kwargs):
        # For I2V/control-camera tasks, the input video is concatenated as
        # [video(3ch), masked_video(3ch), mask(1ch)] = 7 channels.
        # Split before VAE encoding since VAE expects 3 channels.
        if x.shape[1] > 3 and self.i2v_processor is not None:
            video = x[:, :3]
        else:
            video = x

        _video_latents = self.model.encode(video)
        _i2v_results = None

        if self.i2v_processor is not None:
            _i2v_results = self.i2v_processor(
                vae_model=self.model,
                videos=video,
                video_latents=_video_latents,
                **kwargs
            )

        return _video_latents, _i2v_results

    def decode(self, x):
        return self.model.decode(x)

    def forward(self, x):
        raise NotImplementedError("forward function is not implemented")
