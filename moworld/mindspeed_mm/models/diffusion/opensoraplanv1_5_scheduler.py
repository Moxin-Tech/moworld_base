# Copyright 2024 Stability AI, Katherine Crowson and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Callable
from tqdm import tqdm

from megatron.core import mpu

import numpy as np
import torch
import torch.distributed as dist


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`.
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)

    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def opensora_linear_quadratic_schedule(num_inference_steps, approximate_steps=1000):
    if approximate_steps % 2 != 0:
        raise ValueError(f"approximate_steps must be even")
    if num_inference_steps % 2 != 0:
        raise ValueError(f"num_inference_steps must be even")
    if num_inference_steps > approximate_steps:
        raise ValueError(f"num_inference_steps must be less than or equal to approximate_steps")

    _num_inference_steps = num_inference_steps // 2
    _approximate_steps = approximate_steps // 2

    linear_sigmas = [i / (2 * _approximate_steps) for i in range(_num_inference_steps)]
    # NOTE we define a quadratic schedule that is f(x) = ax^2 + bx + c
    quadratic_a = (_approximate_steps - _num_inference_steps) / (_approximate_steps * _num_inference_steps ** 2)
    quadratic_b = (5 * _num_inference_steps - 4 * _approximate_steps) / (2 * _approximate_steps * _num_inference_steps)
    quadratic_c = (_approximate_steps - _num_inference_steps) / _approximate_steps
    quadratic_sigmas = [
        quadratic_a * i ** 2 + quadratic_b * i + quadratic_c
        for i in range(_num_inference_steps, 2 * _num_inference_steps)
    ]
    sigmas = linear_sigmas + quadratic_sigmas + [1.0]
    sigmas = [1.0 - x for x in sigmas]
    return sigmas


class OpenSoraPlanScheduler:
    """
        In OpenSoraPlan v1.5, we use FlowMatching to train the model.
    """

    order = 1

    def __init__(
        self,
        num_inference_steps: bool = None,
        guidance_scale: float = 4.5,
        guidance_rescale: float = 0.7,
        use_linear_quadratic_schedule: bool = False,
        use_dynamic_shifting: bool = False,
        weighting_scheme: str = "logit_normal",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        mode_scale: float = 1.29,
        base_image_seq: int = 256,
        max_image_seq: int = 4096,
        shift: float = 1.0,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        sigma_eps: float = None,
        device: str = "npu",
        **kwargs,
    ):
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.guidance_rescale = guidance_rescale
        self.use_linear_quadratic_schedule = use_linear_quadratic_schedule
        self.device = device

        self.shift = shift
        self.use_dynamic_shifting = use_dynamic_shifting
        self.weighting_scheme = weighting_scheme

        # we use sd3 config
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.mode_scale = mode_scale

        if self.use_dynamic_shifting:
            self.base_image_seq = base_image_seq
            self.max_image_seq = max_image_seq
            self.base_shift = base_shift
            self.max_shift = max_shift
            self.shift_k = (self.max_shift - self.base_shift) / (self.max_image_seq - self.base_image_seq)
            self.shift_b = self.base_shift - self.shift_k * self.base_image_seq

        sigma_eps = sigma_eps

        if sigma_eps is not None:
            if not (sigma_eps >= 0 and sigma_eps <= 1e-2):
                raise ValueError("sigma_eps should be in the range of [0, 1e-2]")
        else:
            sigma_eps = 0.0

        self._sigma_eps = sigma_eps
        self._sigma_min = 0.0
        self._sigma_max = 1.0

        self.sigmas = None

    @property
    def sigma_eps(self):
        return self._sigma_eps

    @property
    def sigma_min(self):
        return self._sigma_min

    @property
    def sigma_max(self):
        return self._sigma_max

    @staticmethod
    def add_noise(
        sample: torch.FloatTensor,
        sigmas: torch.FloatTensor,
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Forward process in flow-matching

        Args:
            sample (`torch.FloatTensor`):
                The input sample.1
            sigma (`float` or `torch.FloatTensor`):
                sigma value in flow matching.

        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """
        sample_dtype = sample.dtype
        sigmas = sigmas.float()
        noise = noise.float()
        sample = sample.float()

        noised_sample = sigmas * noise + (1.0 - sigmas) * sample

        noised_sample = noised_sample.to(sample_dtype)

        return noised_sample

    def compute_density_for_sigma_sampling(
        self,
        batch_size: int,
    ):
        """Compute the density for sampling the sigmas when doing SD3 training.
        """
        if self.weighting_scheme == "logit_normal":
            # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
            sigmas = torch.normal(mean=self.logit_mean, std=self.logit_std, size=(batch_size,), device="cpu")
            sigmas = torch.nn.functional.sigmoid(sigmas)
        elif self.weighting_scheme == "mode":
            sigmas = torch.rand(size=(batch_size,), device="cpu")
            sigmas = 1 - sigmas - self.mode_scale * (torch.cos(math.pi * sigmas / 2) ** 2 - 1 + sigmas)
        else:
            sigmas = torch.rand(size=(batch_size,), device="cpu")

        return sigmas

    def compute_loss_weighting_for_sd3(self, sigmas=None):
        """Computes loss weighting scheme for SD3 training.
        """
        if self.weighting_scheme == "sigma_sqrt":
            weighting = (sigmas**-2.0).float()
        elif self.weighting_scheme == "cosmap":
            bot = 1 - 2 * sigmas + 2 * sigmas**2
            weighting = 2 / (math.pi * bot)
        else:
            weighting = torch.ones_like(sigmas)
        return weighting

    def sigma_shift_opensoraplan(
        self,
        sigmas: Union[float, torch.Tensor],
        image_seq_len: Optional[int] = None,
        gamma: Optional[float] = 1.0,
    ):
        if not self.use_dynamic_shifting:
            sigmas_ = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        else:
            if image_seq_len is None:
                raise ValueError("you have to pass `image_seq_len` when `use_dynamic_shifting` is set to be `True`")
            shift = image_seq_len * self.shift_k + self.shift_b
            shift = math.exp(shift)
            if math.isclose(gamma, 1.0):
                sigmas_ = shift * sigmas / (1 + (shift - 1) * sigmas)
            else:
                sigmas_ = shift / (shift + (1 / sigmas - 1) ** gamma)

        if isinstance(sigmas_, torch.Tensor):
            sigmas_ = torch.where(sigmas_ > self.sigma_eps, sigmas_, torch.ones_like(sigmas_) * self.sigma_eps)
        elif isinstance(sigmas_, np.ndarray):
            sigmas_ = np.where(sigmas_ > self.sigma_eps, sigmas_, np.ones_like(sigmas_) * self.sigma_eps)
        else:
            sigmas_ = max(sigmas_, self.sigma_eps)
        return sigmas_

    def set_sigmas(
        self,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        image_seq_len: Optional[int] = None,
        inversion: Optional[bool] = False,
        **kwargs,
    ):

        if self.use_linear_quadratic_schedule:
            print("use OpenSoraPlanScheduler and linear quadratic schedule")
            approximate_steps = min(max(self.num_inference_steps * 10, 250), 1000)
            sigmas = opensora_linear_quadratic_schedule(self.num_inference_steps, approximate_steps=approximate_steps)
            sigmas = np.array(sigmas)
        else:
            if sigmas is None:
                sigmas = np.linspace(self._sigma_max, self._sigma_min, self.num_inference_steps + 1)
            if self.shift > 1.0 or self.use_dynamic_shifting:
                print("use OpenSoraPlanScheduler and shifting schedule")
                sigmas = self.sigma_shift_opensoraplan(sigmas, image_seq_len=image_seq_len)

        if inversion:
            sigmas = np.copy(np.flip(sigmas))

        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)

        self.sigmas = sigmas

        return sigmas

    def step(
        self,
        model_output: torch.FloatTensor,
        step_index: int,
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ):

        if not (
            isinstance(step_index, int)
            or isinstance(step_index, torch.IntTensor)
            or isinstance(step_index, torch.LongTensor)
        ):
            raise ValueError("step_index should be an integer or a tensor of integer")

        if not (step_index >= 0 and step_index < len(self.sigmas)):
            raise ValueError("step_index should be in the range of [0, len(sigmas)]")

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        sigma = self.sigmas[step_index]
        sigma_next = self.sigmas[step_index + 1]

        prev_sample = sample + (sigma_next - sigma) * model_output

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        return prev_sample

    def training_losses(
        self,
        model_output: torch.Tensor,
        x_start: torch.Tensor,
        noise: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs
    ):
        if torch.all(mask.bool()):
            mask = None

        b, c, _, _, _ = model_output.shape
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, c, 1, 1, 1).float()  # b t h w -> b c t h w
            mask = mask.reshape(b, -1)

        # these weighting schemes use a uniform timestep sampling and instead post-weight the loss
        weighting = self.compute_loss_weighting_for_sd3(sigmas=self.sigmas)

        # flow matching loss
        target = noise - x_start

        # Compute regular loss.
        loss_mse = (weighting.float() * (model_output.float() - target.float()) ** 2).reshape(target.shape[0], -1)
        if mask is not None:
            loss = (loss_mse * mask).sum() / mask.sum()
        else:
            loss = loss_mse.mean()

        return loss

    def q_sample(
        self,
        x_start: torch.Tensor,
        sigmas: torch.Tensor = None,
        noise: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param sigmas: interpolation factor in flow matching.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        b, c, _, _, _ = x_start.shape
        if noise is None:
            noise = torch.randn_like(x_start)
            self.broadcast_tensor(noise)
        if noise.shape != x_start.shape:
            raise ValueError("The shape of noise and x_start must be equal.")
        if sigmas is None:
            sigmas = self.compute_density_for_sigma_sampling(b).to(x_start.device)
            image_seq_len = (noise.shape[-1] * noise.shape[-2]) // 4 if self.use_dynamic_shifting else None  # divided by patch embedding size
            sigmas = self.sigma_shift_opensoraplan(sigmas, image_seq_len=image_seq_len)
            timesteps = sigmas * 1000
            while sigmas.ndim < x_start.ndim:
                sigmas = sigmas.unsqueeze(-1)
            self.broadcast_tensor(sigmas)
            self.broadcast_tensor(timesteps)

        self.sigmas = sigmas
        x_t = self.add_noise(x_start, sigmas, noise)
        return x_t, noise, timesteps

    def sample(
        self,
        model: Callable,
        shape: Union[List, Tuple],
        latents: torch.Tensor,
        model_kwargs: dict = None,
        added_cond_kwargs: dict = None,
        extra_step_kwargs: dict = None,
        **kwargs
    ):

        if not isinstance(shape, (tuple, list)):
            raise AssertionError("param shape is incorrect")
        if latents is None:
            latents = torch.randn(*shape, device=self.device)
        if added_cond_kwargs:
            model_kwargs.update(added_cond_kwargs)

        image_seq_len = (shape[-1] * shape[-2]) // 4 if self.use_dynamic_shifting else None  # patch embedding size
        sigmas = self.set_sigmas(device=self.device, sigmas=None, image_seq_len=image_seq_len)
        timesteps = sigmas.clone() * 1000
        timesteps = timesteps[:-1]

        do_classifier_free_guidance = self.guidance_scale > 1.0

        encoder_hidden_states = model_kwargs.pop("prompt")
        encoder_attention_mask = model_kwargs.pop("prompt_mask")

        with tqdm(total=self.num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                timestep = t.expand(latent_model_input.shape[0])
                attention_mask = torch.ones_like(latent_model_input)[:, 0].to(device=self.device)

                noise_pred = model(
                    latent_model_input,
                    timestep=timestep,
                    prompt=encoder_hidden_states,
                    video_mask=attention_mask,
                    prompt_mask=encoder_attention_mask
                )
                if torch.any(torch.isnan(noise_pred)):
                    raise ValueError("noise_pred contains nan values")

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                latents = self.step(noise_pred, i, latents, **extra_step_kwargs)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (i + 1) % self.order == 0:
                    progress_bar.update()

        return latents

    @staticmethod
    def broadcast_tensor(input_: torch.Tensor):
        cp_src_rank = list(mpu.get_context_parallel_global_ranks())[0]
        if mpu.get_context_parallel_world_size() > 1:
            dist.broadcast(input_, cp_src_rank, group=mpu.get_context_parallel_group())

        tp_src_rank = mpu.get_tensor_model_parallel_src_rank()
        if mpu.get_tensor_model_parallel_world_size() > 1:
            dist.broadcast(input_, tp_src_rank, group=mpu.get_tensor_model_parallel_group())
