from typing import Optional

import torch
import torch.distributed as dist
from megatron.core import mpu
from tqdm.auto import tqdm



class WanFlowMatchScheduler():

    def __init__(
        self,
        num_inference_timesteps=None,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003 / 1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
        guidance_scale=0.0,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        **kwargs
        ):
        self.num_inference_timesteps = num_inference_timesteps
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.guidance_scale = guidance_scale
        self.do_classifier_free_guidance = guidance_scale > 1.0 if isinstance(guidance_scale, (int, float)) else True
        self.num_inference_steps = num_inference_timesteps
        if self.num_inference_timesteps is not None:
            self._set_timesteps(self.num_inference_timesteps, training=False)
        else:
            self._set_timesteps(self.num_train_timesteps, training=True)
        self.num_warmup_steps = max(len(self.timesteps) - (self.num_inference_steps or len(self.timesteps)), 0)

        if not (0 <= min_timestep_boundary < max_timestep_boundary <= 1):
            raise ValueError("min_timestep_boundary and max_timestep_boundary must satisfy 0 <= min < max <= 1")

        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

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
        reduction = kwargs.get("reduction", "mean")
        target = noise - x_start
        loss = torch.nn.functional.mse_loss(model_output.float(), target.float(), reduction=reduction)
        loss *= self._training_weight(t)
        return loss

    def sample(
        self,
        model,
        latents,
        model_kwargs
    ):
        num_inference_steps = self.num_inference_timesteps or self.num_train_timesteps
        num_warmup_steps = len(self.timesteps) - num_inference_steps

        # for loop denoising to get clean latents
        with tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(self.timesteps):
                latent_model_input = latents
                timestep = t.expand(latent_model_input.shape[0]).to(latents.device)

                noise_pred = model(
                    latent_model_input,
                    timestep,
                    model_kwargs.get('prompt_embeds'),
                    **model_kwargs
                )

                if self.do_classifier_free_guidance:
                    noise_uncond = model(
                        latent_model_input,
                        timestep,
                        model_kwargs.get('negative_prompt_embeds'),
                        **model_kwargs
                    )
                    noise_pred = noise_uncond + self.guidance_scale * (noise_pred - noise_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self._step(noise_pred, t, latents)

                if i == len(self.timesteps) - 1 or ((i + 1) > num_warmup_steps):
                    progress_bar.update()

        return latents

    def q_sample(self, latents, noise=None, t=None, **kwargs):
        if noise is None:
            noise = torch.randn_like(latents)

        # use first frame latents in noise
        model_kwargs = kwargs.get("model_kwargs", {})
        first_frame_latents = model_kwargs.get("first_frame_latents", None)
        if first_frame_latents is not None:
            noise[:, :, 0: 1] = first_frame_latents

        if t is None:
            max_timestep_boundary = int(self.max_timestep_boundary * self.num_train_timesteps)
            min_timestep_boundary = int(self.min_timestep_boundary * self.num_train_timesteps)
            timestep_idx = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
            timestep = self.timesteps[timestep_idx].to(latents.device)
        else:
            timestep = t
            timestep_idx = (t.to("cpu") == self.timesteps).nonzero()[0]

        sigma = self.sigmas[timestep_idx].to(latents.device)
        noised_latents = (1 - sigma) * latents + sigma * noise
        return noised_latents, noise, timestep

    def _set_timesteps(self, num_steps=100, training=False):
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min)
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            y = torch.exp(-2 * ((self.timesteps - num_steps / 2) / num_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing


    def step(self, model_output, timestep, sample, return_dict=True, **kwargs):
        prev_sample = self._step(model_output, timestep, sample)
        if not return_dict:
            return (prev_sample,)
        return {"prev_sample": prev_sample}


    def _step(self, model_output, timestep, sample):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_idx = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_idx]
        if timestep_idx + 1 >= len(self.timesteps):
            sigma_next = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_next = self.sigmas[timestep_idx + 1]
        prev_sample = sample + model_output * (sigma_next - sigma)
        return prev_sample

    def _training_weight(self, timestep):
        if len(timestep) > 1:
            timestep = timestep[0]
        timestep_idx = torch.argmin((self.timesteps - timestep.to(self.timesteps.device)).abs())
        weights = self.linear_timesteps_weights[timestep_idx]
        return weights


