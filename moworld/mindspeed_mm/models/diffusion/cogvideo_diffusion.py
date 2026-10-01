import math
from abc import abstractmethod
from inspect import isfunction
from functools import partial
from typing import Union, Tuple, List, Callable

import torch
import torch.nn as nn
import torch.distributed
from torch import Tensor
from tqdm.auto import tqdm
import numpy as np

from megatron.core import mpu

from diffusers.schedulers import CogVideoXDPMScheduler


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


class DiscreteSampling:
    def __init__(self, discretization_config, num_idx, do_append_zero=False, flip=True, uniform_sampling=False,
                 group_num=None):
        self.num_idx = num_idx
        self.sigmas = ZeroSNRDDPMDiscretization(**discretization_config)(num_idx, do_append_zero=do_append_zero, flip=flip)
        world_size = mpu.get_data_parallel_world_size()
        self.uniform_sampling = uniform_sampling
        if group_num:
            self.group_num = group_num
        else:
            if self.uniform_sampling:
                i = 1
                while True:
                    if world_size % i != 0 or num_idx % (world_size // i) != 0:
                        i += 1
                    else:
                        self.group_num = world_size // i
                        break

        if self.uniform_sampling:
            if self.group_num <= 0:
                raise ValueError("group_num should not be less than or equal to 0")

            if world_size % self.group_num != 0:
                raise ValueError("The remainder of world_size to group_num should be equal to 0")

            self.group_width = world_size // self.group_num  # the number of rank in one group
            self.sigma_interval = self.num_idx // self.group_num

    def idx_to_sigma(self, idx):
        return self.sigmas[idx]

    def __call__(self, n_samples, rand=None, return_idx=False):
        if self.uniform_sampling:
            rank = mpu.get_data_parallel_rank()
            group_index = rank // self.group_width
            idx = default(
                rand,
                torch.randint(
                    group_index * self.sigma_interval, (group_index + 1) * self.sigma_interval, (n_samples,)
                ),
            )
        else:
            idx = default(
                rand,
                torch.randint(0, self.num_idx, (n_samples,)),
            )

        if return_idx:
            return self.idx_to_sigma(idx), idx
        else:
            return self.idx_to_sigma(idx)



def generate_roughly_equally_spaced_steps(num_substeps: int, max_step: int) -> np.ndarray:
    return np.linspace(max_step - 1, 0, num_substeps, endpoint=False).astype(int)[::-1]


def make_beta_schedule(
    schedule,
    n_timestep,
    linear_start=1e-4,
    linear_end=2e-2,
):
    if schedule == "linear":
        betas = torch.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64) ** 2
    else:
        raise NotImplementedError("Only support linear schedule")
    return betas.numpy()


class Discretization:
    def __call__(self, n, do_append_zero=True, device="cpu", flip=False, return_idx=False):
        if return_idx:
            sigmas, idx = self.get_sigmas(n, device=device, return_idx=return_idx)
        else:
            sigmas = self.get_sigmas(n, device=device, return_idx=return_idx)
        sigmas = append_zero(sigmas) if do_append_zero else sigmas
        if return_idx:
            return sigmas if not flip else torch.flip(sigmas, (0,)), idx
        else:
            return sigmas if not flip else torch.flip(sigmas, (0,))

    @abstractmethod
    def get_sigmas(self, n, device):
        pass


class ZeroSNRDDPMDiscretization(Discretization):
    def __init__(
        self,
        linear_start=0.00085,
        linear_end=0.0120,
        num_timesteps=1000,
        shift_scale=1.0,  # noise schedule t_n -> t_m: logSNR(t_m) = logSNR(t_n) - log(shift_scale)
        keep_start=False,
        post_shift=False,
    ):
        super().__init__()
        if keep_start and not post_shift:
            linear_start = linear_start / (shift_scale + (1 - shift_scale) * linear_start)
        self.num_timesteps = num_timesteps
        betas = make_beta_schedule("linear", num_timesteps, linear_start=linear_start, linear_end=linear_end)
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.to_torch = partial(torch.tensor, dtype=torch.float32)

        # SNR shift
        if not post_shift:
            self.alphas_cumprod = self.alphas_cumprod / (shift_scale + (1 - shift_scale) * self.alphas_cumprod)

        self.post_shift = post_shift
        self.shift_scale = shift_scale

    def get_sigmas(self, n, device="cpu", return_idx=False):
        if n < self.num_timesteps:
            timesteps = generate_roughly_equally_spaced_steps(n, self.num_timesteps)
            alphas_cumprod = self.alphas_cumprod[timesteps]
        elif n == self.num_timesteps:
            alphas_cumprod = self.alphas_cumprod
        else:
            raise ValueError

        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        alphas_cumprod = to_torch(alphas_cumprod)
        alphas_cumprod_sqrt = alphas_cumprod.sqrt()
        alphas_cumprod_sqrt_0 = alphas_cumprod_sqrt[0].clone()
        alphas_cumprod_sqrt_T = alphas_cumprod_sqrt[-1].clone()

        alphas_cumprod_sqrt -= alphas_cumprod_sqrt_T
        alphas_cumprod_sqrt *= alphas_cumprod_sqrt_0 / (alphas_cumprod_sqrt_0 - alphas_cumprod_sqrt_T)

        if self.post_shift:
            alphas_cumprod_sqrt = (
                alphas_cumprod_sqrt**2 / (self.shift_scale + (1 - self.shift_scale) * alphas_cumprod_sqrt**2)
            ) ** 0.5

        if return_idx:
            return torch.flip(alphas_cumprod_sqrt, (0,)), timesteps
        else:
            return torch.flip(alphas_cumprod_sqrt, (0,))  # sqrt(alpha_t): 0 -> 0.99


class EpsWeighting:
    @staticmethod
    def __call__(sigma):
        return sigma ** -2.0


class VideoScaling:  # similar to VScaling
    @staticmethod
    def __call__(
            alphas_cumprod_sqrt: torch.Tensor, **additional_model_inputs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c_skip = alphas_cumprod_sqrt
        c_out = -((1 - alphas_cumprod_sqrt ** 2) ** 0.5)
        c_in = torch.ones_like(alphas_cumprod_sqrt, device=alphas_cumprod_sqrt.device)
        c_noise = additional_model_inputs["idx"].clone()
        return (c_skip, c_out, c_in, c_noise)


class DiscreteDenoiser(nn.Module):
    def __init__(
            self,
            num_idx,
            discretization_config,
            do_append_zero=False,
            quantize_c_noise=True,
            flip=True,
    ):
        super().__init__()
        self.weighting = EpsWeighting()
        self.scaling = VideoScaling()
        sigmas = ZeroSNRDDPMDiscretization(**discretization_config)(num_idx, do_append_zero=do_append_zero, flip=flip)
        self.sigmas = sigmas
        self.quantize_c_noise = quantize_c_noise

    def forward(
            self,
            input_x: torch.Tensor,
            sigma: torch.Tensor,
            **additional_model_inputs,
    ) -> torch.Tensor:
        sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, input_x.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma, **additional_model_inputs)
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))

        return c_in, c_noise, c_out, c_skip

    def sigma_to_idx(self, sigma):
        dists = sigma - self.sigmas.to(sigma.device)[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)

    def idx_to_sigma(self, idx):
        return self.sigmas.to(idx.device)[idx]

    def possibly_quantize_sigma(self, sigma):
        return self.idx_to_sigma(self.sigma_to_idx(sigma))

    def possibly_quantize_c_noise(self, c_noise):
        if self.quantize_c_noise:
            return self.sigma_to_idx(c_noise)
        else:
            return c_noise

    def w(self, sigma):
        return self.weighting(sigma)


class CogVideoDiffusion(nn.Module):
    def __init__(self,
                 sigma_sampler_config,
                 denoiser_config,
                 scheduler_config=None,
                 block_scale=None,
                 block_size=None,
                 min_snr_value=None,
                 fixed_frames=0,
                 loss_type="l2",
                 offset_noise_level=0.0,
                 batch2model_keys=None,
                 **kwargs
                 ):
        super().__init__()
        self.fixed_frames = fixed_frames
        self.block_scale = block_scale
        self.block_size = block_size
        self.min_snr_value = min_snr_value

        if loss_type not in ["l2", "l1"]:
            raise ValueError("Only support l2 or l1 type")

        self.sigma_sampler = DiscreteSampling(**sigma_sampler_config)

        self.loss_type = loss_type
        self.offset_noise_level = offset_noise_level

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)
        self.alphas_cumprod_sqrt = None

        self.denoiser = DiscreteDenoiser(**denoiser_config)
        self.c_in, self.c_noise, self.c_out, self.c_skip = None, None, None, None
        self.x_start = None
        self.latents = None

        self.device = kwargs.pop("device", "npu")
        self.num_inference_steps = kwargs.pop("num_inference_steps", 5)
        scheduler_config = {} if scheduler_config is None else scheduler_config
        self.diffusion = CogVideoXDPMScheduler(**scheduler_config)
        self.diffusion.set_timesteps(self.num_inference_steps)
        self.timesteps = self.diffusion.timesteps
        self.init_noise_sigma = self.diffusion.init_noise_sigma
        self.step = self.diffusion.step
        self.num_warmup_steps = max(
            len(self.timesteps) - self.num_inference_steps * self.diffusion.order, 0
        )

    def q_sample(self, latents, **kwargs):
        noise = torch.randn_like(latents)

        additional_model_inputs = dict()
        alphas_cumprod_sqrt, idx = self.sigma_sampler(latents.shape[0], return_idx=True)
        self.alphas_cumprod_sqrt = alphas_cumprod_sqrt.to(latents.device)
        idx = idx.to(latents.device)

        # broadcast noise here

        additional_model_inputs["idx"] = idx

        if self.offset_noise_level > 0.0:
            noise = (
                    noise + append_dims(torch.randn(latents.shape[0]).to(latents.device),
                                        latents.ndim) * self.offset_noise_level
            )

        noised_input = latents.float() * append_dims(self.alphas_cumprod_sqrt, latents.ndim) + noise * append_dims(
            (1 - self.alphas_cumprod_sqrt ** 2) ** 0.5, latents.ndim
        )

        self.c_in, self.c_noise, self.c_out, self.c_skip = self.denoiser(latents,
                                                                         self.alphas_cumprod_sqrt,
                                                                         **additional_model_inputs)
        self.latents = latents
        self.x_start = noised_input

        kwargs["model_kwargs"]["c_out"] = self.c_out
        kwargs["model_kwargs"]["noised_start"] = self.x_start * self.c_skip
        kwargs["model_kwargs"]["alphas_cumprod"] = self.alphas_cumprod_sqrt

        return noised_input * self.c_in, noise, idx

    def training_losses(self, model_output, x_start, **kwargs):
        model_output = model_output * kwargs['c_out'] + kwargs["noised_start"]

        w = append_dims(1 / (1 - kwargs['alphas_cumprod'] ** 2), x_start.ndim)  # v-pred

        if self.min_snr_value is not None:
            w = min(w, self.min_snr_value)

        return self.get_loss(model_output, x_start, w)

    def get_loss(self, model_output, target, w):
        model_output = model_output.transpose(1, 2)
        target = target.transpose(1, 2)
        if self.loss_type == "l2":
            return torch.mean((w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1)
        elif self.loss_type == "l1":
            return torch.mean((w * (model_output - target).abs()).reshape(target.shape[0], -1), 1)
        else:
            raise NotImplementedError


    def sample(
        self,
        model: Callable,
        latents: Tensor,
        model_kwargs: dict = None,
        extra_step_kwargs: dict = None,
        **kwargs
    ) -> Tensor:
        """
        Generate samples from the model.
        :param model: the noise predict model.
        :param shape: the shape of the samples, (N, C, H, W).
        :param latents: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
            {
                "attention_mask": attention_mask,
                "encoder_hidden_states": prompt_embeds
                "encoder_attention_mask": prompt_attention_mask
            }
        :return: a non-differentiable batch of samples.
        Returns clean latents.
        """
        self.diffusion.set_timesteps(self.num_inference_steps, device=self.device)
        self.timesteps = self.diffusion.timesteps

        guidance_scale = self.guidance_scale

        # for loop denoising to get latents
        with tqdm(total=self.num_inference_steps) as progress_bar:
            old_pred_original_sample = None
            for i, t in enumerate(self.timesteps):
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.diffusion.scale_model_input(latent_model_input, t)
                current_timestep = t.expand(latent_model_input.shape[0])
                model_kwargs["latents"] = latent_model_input.permute(0, 2, 1, 3, 4)

                with torch.no_grad():
                    noise_pred = model(timestep=current_timestep, **model_kwargs)

                if isinstance(noise_pred, tuple) or isinstance(noise_pred, list):
                    noise_pred = noise_pred[0]

                # perform guidance
                noise_pred = noise_pred.permute(0, 2, 1, 3, 4).float()
                self.guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(
                            math.pi * ((self.num_inference_steps - t.item()) / self.num_inference_steps) ** 5.0)) / 2
                )

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents, old_pred_original_sample = self.diffusion.step(
                    noise_pred,
                    old_pred_original_sample,
                    t,
                    self.timesteps[i - 1] if i > 0 else None,
                    latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )
                latents = latents.to(latent_model_input.dtype)

                # call the callback, if provided
                if i == len(self.timesteps) - 1 or (
                        (i + 1) > self.num_warmup_steps and (i + 1) % self.diffusion.order == 0):
                    progress_bar.update()
        return latents
