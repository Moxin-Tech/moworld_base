# Copyright 2024 Stability AI, Katherine Crowson and The HuggingFace Team. All rights reserved.
import math
from enum import Enum
from typing import Optional, Union, Callable

import torch
from diffusers.training_utils import compute_density_for_timestep_sampling
from torch import nn
from tqdm.auto import tqdm

from mindspeed_mm.models.predictor.dits.hunyuanvideo15.utils import get_parallel_state, sync_tensor_for_sp, \
    initialize_parallel_state


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed].
    """
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True
    )
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # Mix the rescaled noise prediction with the original noise configuration
    # to avoid overly uniform or "monotonous" images.
    noise_cfg = (
            guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    )
    return noise_cfg


class LinearInterpolationSchedule:
    """Simple linear interpolation schedule for flow matching"""

    def __init__(self, T: int = 1000):
        self.T = T

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Linear interpolation: x_t = (1 - t/T) * x0 + (t/T) * x1
        Args:
            x0: starting point (clean latents)
            x1: ending point (noise)
            t: timesteps
        """
        t_normalized = t / self.T
        t_normalized = t_normalized.view(-1, *([1] * (x0.ndim - 1)))
        return (1 - t_normalized) * x0 + t_normalized * x1


class SNRType(str, Enum):
    UNIFORM = "uniform"
    LOGNORM = "lognorm"
    MIX = "mix"
    MODE = "mode"


class TimestepSampler:
    TRAIN_EPS = 1e-5
    SAMPLE_EPS = 1e-3

    def __init__(
            self,
            T: int = 1000,
            device: torch.device = None,
            snr_type: SNRType = SNRType.LOGNORM,
    ):
        self.T = T  # 1000
        self.device = device
        self.snr_type = SNRType(snr_type) if isinstance(snr_type, str) else snr_type  # lognorm

    def _check_interval(self, eval_state: bool = False):
        # For ICPlan-like path with velocity model, use [eps, 1-eps]
        eps = self.SAMPLE_EPS if eval_state else self.TRAIN_EPS
        t0 = eps
        t1 = 1.0 - eps
        return t0, t1

    def sample(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        if device is None:
            device = self.device if self.device is not None else torch.device("npu")

        t0, t1 = self._check_interval(eval_state=False)

        if self.snr_type == SNRType.UNIFORM:
            # Uniform sampling: t = rand() * (t1 - t0) + t0
            t = torch.rand((batch_size,), device=device) * (t1 - t0) + t0

        elif self.snr_type == SNRType.LOGNORM:
            # Log-normal sampling: t = 1 / (1 + exp(-u)) * (t1 - t0) + t0
            u = torch.normal(mean=0.0, std=1.0, size=(batch_size,), device=device)
            t = 1.0 / (1.0 + torch.exp(-u)) * (t1 - t0) + t0

        elif self.snr_type == SNRType.MIX:
            # Mix sampling: 30% lognorm + 70% clipped uniform
            u = torch.normal(mean=0.0, std=1.0, size=(batch_size,), device=device)
            t_lognorm = 1.0 / (1.0 + torch.exp(-u)) * (t1 - t0) + t0

            # Clipped uniform: delta = 0.0 (0.0~0.01 clip)
            delta = 0.0
            t0_clip = t0 + delta
            t1_clip = t1 - delta
            t_clip_uniform = torch.rand((batch_size,), device=device) * (t1_clip - t0_clip) + t0_clip

            # Mix with 30% lognorm, 70% uniform
            mask = (torch.rand((batch_size,), device=device) > 0.3).float()
            t = mask * t_lognorm + (1 - mask) * t_clip_uniform

        elif self.snr_type == SNRType.MODE:
            # Mode sampling: t = 1 - u - mode_scale * (cos(pi * u / 2)^2 - 1 + u)
            mode_scale = 1.29
            u = torch.rand(size=(batch_size,), device=device)
            t = 1.0 - u - mode_scale * (torch.cos(math.pi * u / 2.0) ** 2 - 1.0 + u)
            # Scale to [t0, t1] range
            t = t * (t1 - t0) + t0
        else:
            raise ValueError(f"Unknown SNR type: {self.snr_type}")

        # Scale to [0, T] range
        timesteps = t * self.T
        return timesteps


def timestep_transform(timesteps: torch.Tensor, T: int, shift: float = 1.0) -> torch.Tensor:
    """Transform timesteps with shift"""
    EPSILON = 1e-15
    if abs(shift - 1.0) < EPSILON:
        return timesteps
    timesteps_normalized = timesteps / T
    timesteps_transformed = shift * timesteps_normalized / (1 + (shift - 1) * timesteps_normalized)
    return timesteps_transformed * T


class Hunyuan_15_FlowMatchDiscreteScheduler:
    """
    Euler scheduler.
    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        timestep_spacing (`str`, defaults to `"linspace"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed].
        shift (`float`, defaults to 1.0):
            The shift value for the timestep schedule.
        reverse (`bool`, defaults to `True`):
            Whether to reverse the timestep schedule.
    """

    _compatibles = []
    order = 1

    def __init__(
            self,
            num_train_timesteps: int = 1000,
            num_inference_timesteps: Optional[int] = None,
            shift: float = 1.0,
            reverse: bool = True,
            solver: str = "euler",
            sample_method: str = "logit_normal",
            logit_mean: float = 0.0,
            logit_std: float = 1.0,
            precondition_outputs: bool = False,
            n_tokens: Optional[int] = None,
            snr: str = "lognorm",
            train_timestep_shift: float = 3.0,
            **kwargs
    ):

        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_inference_timesteps
        self.shift = shift
        self.n_tokens = n_tokens
        self.reverse = reverse
        self.solver = solver
        self.sample_method = sample_method
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.precondition_outputs = precondition_outputs

        sigmas = torch.linspace(1, 0, num_train_timesteps + 1)

        if not reverse:
            sigmas = sigmas.flip(0)

        self.sigmas = sigmas
        # the value fed to model
        self.timesteps = (sigmas[:-1] * num_train_timesteps).to(dtype=torch.float32)

        self._step_index = None
        self._begin_index = None

        self.supported_solver = ["euler"]
        if solver not in self.supported_solver:
            raise ValueError(
                f"Solver {solver} not supported. Supported solvers: {self.supported_solver}"
            )

        self.device = torch.device("npu")
        self.noise_schedule = LinearInterpolationSchedule(T=num_train_timesteps)
        self.timestep_sampler = TimestepSampler(
            T=num_train_timesteps,
            device=None,
            snr_type=snr,
        )
        self.train_timestep_shift = train_timestep_shift

        initialize_parallel_state(sp=kwargs['sp_size'], dp_replicate=kwargs['dp_replicate'])
        self.parallel_state = get_parallel_state()
        self.sp_enabled = self.parallel_state.sp_enabled
        self.sp_group = self.parallel_state.sp_group if self.sp_enabled else None

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def _sigma_to_t(self, sigma):
        return sigma * self.num_train_timesteps

    def set_timesteps(
            self,
            num_inference_steps: int,
            device: Union[str, torch.device] = None,
            n_tokens: int = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            n_tokens (`int`, *optional*):
                Number of tokens in the input sequence.
        """
        self.num_inference_timesteps = num_inference_steps

        sigmas = torch.linspace(1, 0, num_inference_steps + 1)
        sigmas = self.sd3_time_shift(sigmas)

        if not self.reverse:
            sigmas = 1 - sigmas

        self.sigmas = sigmas
        self.timesteps = (sigmas[:-1] * self.num_train_timesteps).to(
            dtype=torch.float32, device=device
        )

        # Reset step index
        self._step_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    @staticmethod
    def scale_model_input(
            sample: torch.Tensor, timestep: Optional[int] = None
    ) -> torch.Tensor:
        return sample

    def sd3_time_shift(self, t: torch.Tensor):
        return (self.shift * t) / (1 + (self.shift - 1) * t)

    def step(
            self,
            model_output: torch.FloatTensor,
            timestep: Union[float, torch.FloatTensor],
            sample: torch.FloatTensor,
            **kwargs
    ) -> torch.Tensor:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            n_tokens (`int`, *optional*):
                Number of tokens in the input sequence.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or
                tuple.

        Returns:
            sample_tensor
        """

        if (
                isinstance(timestep, int)
                or isinstance(timestep, torch.IntTensor)
                or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Ensure that you pass"
                    " one of the values from `scheduler.timesteps` as the timestep argument."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        dt = self.sigmas[self.step_index + 1] - self.sigmas[self.step_index]

        if self.solver == "euler":
            prev_sample = sample + model_output.to(torch.float32) * dt
        else:
            raise ValueError(
                f"Solver {self.solver} not supported. Supported solvers: {self.supported_solver}"
            )

        # upon completion increase step index by one
        self._step_index += 1

        return prev_sample

    def get_sigmas(
            self,
            timesteps: torch.Tensor,
            n_dim: int = 4,
            dtype: torch.dtype = torch.float32
    ):
        sigmas = self.sigmas.to(device=timesteps.device, dtype=dtype)
        schedule_timesteps = self.timesteps.to(timesteps.device)

        # self.begin_index is None when scheduler is used for training, or pipeline does not implement set_begin_index
        if self.begin_index is None:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        elif self.step_index is not None:
            # add_noise is called after first denoising step (for inpainting)
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            # add_noise is called before first denoising step to create initial latent
            step_indices = [self.begin_index] * timesteps.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)

        return sigma

    def q_sample(
            self,
            x_start: Optional[torch.Tensor],
            t: Optional[torch.Tensor] = None,
            noise: Optional[torch.Tensor] = None,
            **kwargs
    ):
        latents = kwargs.get("latents", x_start)
        if self.sp_enabled:
            latents = sync_tensor_for_sp(latents, self.sp_group)
        b, _, _, _, _ = latents.shape
        if noise is None:
            noise = torch.randn_like(latents)

        if noise.shape != latents.shape:
            raise ValueError("The shape of noise and x_start must be equal.")

        timesteps = self.timestep_sampler.sample(b, device=self.device)
        timesteps = timestep_transform(timesteps, self.num_train_timesteps, self.train_timestep_shift)

        x_t = self.noise_schedule.forward(latents, noise, timesteps)

        return x_t, noise, timesteps

    def sample(
            self,
            model: Callable,
            latents: torch.Tensor,
            img_latents: Optional[torch.Tensor] = None,
            device: torch.device = "npu",
            do_classifier_free_guidance: bool = False,
            guidance_scale: float = 1.0,
            guidance_rescale: float = 0.0,
            embedded_guidance_scale: Optional[float] = None,
            model_kwargs: dict = None,
            extra_step_kwargs: dict = None,
            i2v_mode: bool = False,
            i2v_condition_type: str = "token_replace",
            **kwargs
    ) -> torch.Tensor:
        extra_step_kwargs = {} if extra_step_kwargs is None else extra_step_kwargs
        dtype = latents.dtype
        # denoising loop
        num_inference_steps = self.num_train_timesteps if self.num_inference_timesteps is None else self.num_inference_timesteps
        self.set_timesteps(self.num_inference_timesteps, device=device)

        # for loop denoising to get latents
        with tqdm(total=num_inference_steps) as propress_bar:
            for t in self.timesteps:
                if i2v_mode and i2v_condition_type == "token_replace":
                    latents = torch.concat([img_latents, latents[:, :, 1:, :, :]], dim=2)

                latent_model_input = (
                    torch.cat([latents] * 2)
                    if do_classifier_free_guidance
                    else latents
                ).to(dtype)
                latent_model_input = self.scale_model_input(latent_model_input, t)

                t_expand = t.repeat(latent_model_input.shape[0])

                if embedded_guidance_scale is not None:
                    guidance_expand = torch.tensor(
                        [embedded_guidance_scale] * latent_model_input.shape[0],
                        dtype=torch.float32,
                        device=device,
                    ).to(dtype) * 1000.0
                    model_kwargs.update({"guidance": guidance_expand})

                with torch.no_grad():
                    noise_pred = model(latent_model_input, t_expand, **model_kwargs)

                if isinstance(noise_pred, tuple) or isinstance(noise_pred, list):
                    noise_pred = noise_pred[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                    )

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=guidance_rescale,
                    )

                # compute the previous noisy sample x_t -> x_t-1
                if i2v_mode and i2v_condition_type == "token_replace":
                    latents = self.step(
                        noise_pred[:, :, 1:, :, :],
                        t,
                        latents[:, :, 1:, :, :],
                        **extra_step_kwargs
                    )
                    latents = torch.concat([img_latents, latents], dim=2)
                else:
                    latents = self.step(
                        noise_pred,
                        t,
                        latents,
                        **extra_step_kwargs
                    )
                propress_bar.update()

        return latents

    def training_losses(
            self,
            model_output: torch.Tensor,
            x_start: Optional[torch.Tensor] = None,
            x_t: Optional[torch.Tensor] = None,
            noise: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
            t: Optional[torch.Tensor] = None,
            **kwargs
    ):
        if self.precondition_outputs:
            sigmas = self.get_sigmas(t, n_dim=len(model_output.shape), dtype=x_start.dtype)
            model_output = model_output * (-sigmas) + x_t
            target = x_start
        else:
            target = noise - x_start

        if kwargs.get("target", None) is not None:
            target = kwargs.get("target", None).to(dtype=model_output.dtype)

        if self.sp_enabled:
            target = sync_tensor_for_sp(target, self.sp_group)

        loss = nn.functional.mse_loss(model_output, target.to(dtype=model_output.dtype))
        return loss
