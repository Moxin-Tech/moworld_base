import math
from typing import Optional
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat


def get_oscillation_gs(guidance_scale: float, i: int, force_num=10):
    """
    get oscillation guidance for cfg.

    Args:
        guidance_scale: original guidance value
        i: denoising step
        force_num: before which don't apply oscillation
    """
    if i < force_num or (i >= force_num and i % 2 == 0):
        gs = guidance_scale
    else:
        gs = 1.0
    return gs


def time_shift(alpha: float, t: Tensor):
    return alpha * t / (1 + (alpha - 1) * t)


def get_res_lin_function(
    x1: float = 256, y1: float = 1, x2: float = 4096, y2: float = 3
) -> callable:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_timesteps(
    num_steps: int,
    image_seq_len: int,
    num_frames: int,
    base_shift: float = 1,
    max_shift: float = 3,
):
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    # estimate mu based on linear estimation between two points
    # spatial scale
    shift_alpha = get_res_lin_function(y1=base_shift, y2=max_shift)(
        image_seq_len
    )
    # temporal scale
    shift_alpha *= math.sqrt(num_frames)
    # calculate shifted timesteps
    timesteps = time_shift(shift_alpha, timesteps)
    return timesteps.tolist()


def pack(x: Tensor, patch_size: int = 2):
    return rearrange(
        x, "b c t (h ph) (w pw) -> b (t h w) (c ph pw)", ph=patch_size, pw=patch_size
    )


class Opensora2FlowMatchScheduler:
    def __init__(
        self,
        num_inference_steps: int = None,
        num_train_steps=1000,
        sigma_min=1e-5,
        text_osci=True,
        image_osci=True,
        guidance_img=3.0,
        guidance=7.5,
        scale_temporal_osci=False,
        **kwargs,
    ):
        self.num_sampling_steps = num_inference_steps
        self.num_timesteps = num_train_steps
        self.sigma_min = sigma_min
        self.text_osci = text_osci
        self.image_osci = image_osci
        self.guidance_img = guidance_img
        self.guidance = guidance
        self.scale_temporal_osci = scale_temporal_osci

    def q_sample(
        self,
        x_start: Optional[torch.Tensor],
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        **kwargs
    ):
        dtype = x_start.dtype
        device = x_start.device
        # == prepare timestep ==
        if t is None:
            shift_alpha = get_res_lin_function()((x_start.shape[-1] * x_start.shape[-2]) // 4)
            # add temporal influence
            shift_alpha *= math.sqrt(x_start.shape[-3])  # for image, T=1 so no effect
            t = torch.sigmoid(torch.randn((x_start.shape[0]), device=device))
            t = time_shift(shift_alpha, t).to(dtype)

        x_start = pack(x_start, patch_size=kwargs.get("patch_size", 2))

        # == prepare noise vector ==
        if noise is None:
            noise = torch.randn_like(x_start)
        if noise.shape != x_start.shape:
            raise ValueError("The shape of noise and x_start must be equal.")

        t_rev = 1 - t
        x_t = t_rev[:, None, None] * x_start + (1 - (1 - self.sigma_min) * t_rev[:, None, None]) * noise
        return x_t, noise, t

    def training_losses(
        self,
        model_output: torch.Tensor,
        x_start: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        **kwargs
    ):
        x_start = pack(x_start, patch_size=kwargs.get("patch_size", 2))
        v_t = (1 - self.sigma_min) * noise - x_start
        loss = F.mse_loss(model_output.float(), v_t.float(), reduction="mean")
        return loss
