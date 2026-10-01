from typing import Optional
import enum
import math
import numpy as np
import torch


class SNRType(enum.Enum):
    UNIFORM = enum.auto()
    LOGNORM = enum.auto()


def expand_t_like_x(t, x):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * len(x[0].size())
    t = t.view(t.size(0), *dims)
    return t


class HunyuanVideoI2VDiffusion:

    def __init__(
            self,
            num_train_timesteps=1000,
            shift=1.0,
            video_shift=None,
            reverse=False,
            reverse_time_schedule=False,
            train_eps=None,
            sample_eps=None,
            snr_type="lognorm",
            **kwargs
    ):
        self.shift = shift  # flow matching shift factor, =sqrt(m/n)
        if video_shift is None:
            video_shift = shift  # if video shift is not given, set it to be the same as flow shift
        self.video_shift = video_shift
        self.reverse = reverse
        self.reverse_time_schedule = reverse_time_schedule
        self.training_timesteps = num_train_timesteps
        self.train_eps = 0
        self.sample_eps = 0

        self.snr_type = snr_type

    @staticmethod
    def check_interval(
            train_eps,
            sample_eps,
            *,
            diffusion_form="SBDM",
            sde=False,
            reverse=False,
            eval_mode=False,
            last_step_size=0.0,
    ):
        t0 = 0
        t1 = 1
        eps = train_eps if not eval_mode else sample_eps
        t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1

    def sample(self, x1, n_tokens=None):
        """Sampling x0 & t based on shape of x1 (if needed)
        Args:
          x1 - data point; [batch, *dim]
        """
        if isinstance(x1, (list, tuple)):
            x0 = [torch.randn_like(img_start) for img_start in x1]
        else:
            x0 = torch.randn_like(x1)
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)

        if self.snr_type == "uniform":
            t = torch.rand((len(x1),)) * (t1 - t0) + t0
        elif self.snr_type == "lognorm":
            u = torch.normal(mean=0.0, std=1.0, size=(len(x1),))
            t = 1 / (1 + torch.exp(-u)) * (t1 - t0) + t0
        else:
            raise ValueError(f"Unknown snr type: {self.snr_type}")

        if not math.isclose(self.shift, 1.0):
            if self.reverse:
                t = (self.shift * t) / (1 + (self.shift - 1) * t)
            else:
                t = t / (self.shift - (self.shift - 1) * t)

        t = t.to(x1[0])
        return t, x0, x1

    def plan(self, t, x0, x1):
        xt = self.compute_xt(t, x0, x1)
        ut = self.compute_ut(t, x0, x1, xt)
        return t, xt, ut

    def compute_alpha_t(self, t):
        """Compute the data coefficient along the path"""
        if self.reverse:
            return 1 - t, -1
        else:
            return t, 1

    def compute_sigma_t(self, t):
        """Compute the noise coefficient along the path"""
        if self.reverse:
            return t, 1
        else:
            return 1 - t, -1

    def compute_mu_t(self, t, x0, x1):
        """Compute the mean of time-dependent density p_t"""
        t = expand_t_like_x(t, x1)
        alpha_t, _ = self.compute_alpha_t(t)
        sigma_t, _ = self.compute_sigma_t(t)
        if isinstance(x1, (list, tuple)):
            return [alpha_t[i] * x1[i] + sigma_t[i] * x0[i] for i in range(len(x1))]
        else:
            return alpha_t * x1 + sigma_t * x0

    def compute_xt(self, t, x0, x1):
        """Sample xt from time-dependent density p_t; rng is required"""
        xt = self.compute_mu_t(t, x0, x1)
        return xt

    def compute_ut(self, t, x0, x1, xt):
        """Compute the vector field corresponding to p_t"""
        t = expand_t_like_x(t, x1)
        _, d_alpha_t = self.compute_alpha_t(t)
        _, d_sigma_t = self.compute_sigma_t(t)
        if isinstance(x1, (list, tuple)):
            return [d_alpha_t * x1[i] + d_sigma_t * x0[i] for i in range(len(x1))]
        else:
            return d_alpha_t * x1 + d_sigma_t * x0

    def get_model_t(self, t):
        if self.reverse_time_schedule:
            return (1 - t) * self.training_timesteps
        else:
            return t * self.training_timesteps

    def q_sample(
            self,
            x1,
            timestep=None,
            **kwargs
    ):
        cond_latents = kwargs["model_kwargs"].get("cond_latents", None)
        self.shift = self.video_shift
        t, x0, x1 = self.sample(x1)
        if timestep is not None:
            t = torch.ones_like(t) * timestep
        t, xt, ut = self.plan(t, x0, x1)
        input_t = self.get_model_t(t)

        xt = torch.concat([cond_latents, xt[:, :, 1:, :, :]], dim=2)
        return xt, ut, input_t

    @staticmethod
    def training_losses(
            model_output: torch.Tensor,
            x_start: Optional[torch.Tensor] = None,
            x_t: Optional[torch.Tensor] = None,
            noise: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
            t: Optional[torch.Tensor] = None,
            **kwargs
    ):
        model_output = model_output[:, :, 1:, :, :]
        noise = noise[:, :, 1:, :, :]

        loss = torch.mean(((model_output - noise) ** 2), dim=list(range(1, len(model_output.size()))))

        return loss
