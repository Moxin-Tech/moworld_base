import importlib
import warnings
from abc import abstractmethod

import torch
import torch.nn as nn
from diffusers.utils.accelerate_utils import apply_forward_hook


class DiffusersAEModel(nn.Module):
    """
    Support import AutoEncoder from diffusers
    """

    def __init__(self, model_name, config):
        super().__init__()
        module = importlib.import_module("diffusers")
        automodel = getattr(module, model_name)
        self.model_name = model_name
        self.model = automodel.from_pretrained(config["from_pretrained"], torch_dtype=config.pop("dtype", None))
        self.do_sample = config.get("do_sample", True)

        # tiling
        self._tiling = False  # True: use custom tiling method; False: disable tiling or use diffusers' tiling function
        use_tiling = config.pop("enable_tiling", False)
        self.tiling_param = None
        if use_tiling:
            self.enable_tiling(tiling_param=config.get("tiling_param", None))

        # In diffusers, model.encode does not normalize latent at last, the normalization is performed in other modules.
        # MindSpeed-MM merges this step into AE.encode.
        # Mode:
        # 1. value_shift_scale: (output - shift_factor) * scaling_factor
        # 2. channel_specified_shift_scale: (output - latents_mean) / latents_std
        self.norm_latents = config.pop("norm_latents", True)
        self.norm_mode = config.pop("norm_mode", "value_shift_scale")

    def enable_tiling(self, tiling_param=None):
        if hasattr(self.model, "enable_tiling"):
            if tiling_param:
                self.model.enable_tiling(**tiling_param)
                self.tiling_param = tiling_param
            else:
                self.model.enable_tiling()
        else:
            self._tiling = True
            self.tiling_param = tiling_param if tiling_param else self.tiling_param
            warnings.warn(
                f"The autoencoder {self.model_name} in the diffusers library does not implement tiling functionality. "
                "Please ensure to call the custom tiling method to enable tiling. "
            )

    def disable_tiling(self):
        if hasattr(self.model, "disable_tiling"):
            self.model.disable_tiling()
        else:
            self._tiling = False

    @apply_forward_hook
    def encode(self, x, **kwargs):
        if self._tiling:
            output = self.tiled_encode(x, **kwargs)
        else:
            output = self.model.encode(x, return_dict=True, **kwargs)
            if self.do_sample:
                output = output.latent_dist.sample()
            else:
                output = output.latent_dist.mode()

        if self.norm_latents:
            output = self.normalize_latent(output)

        return output

    @abstractmethod
    def tiled_encode(self, x, **kwargs):
        pass

    def normalize_latent(self, x):
        if self.norm_mode == "value_shift_scale":
            if getattr(self.model.config, "shift_factor", None):
                output = (
                    x - self.model.config.shift_factor
                ) * self.model.config.scale_factor
            else:
                output = x * self.model.config.scale_factor
        elif self.norm_mode == "channel_specified_shift_scale":
            latents_mean = (
                torch.tensor(self.model.config.latents_mean).view(1, -1, 1, 1, 1).to(x)
            )  # b c t h w
            latents_std = (
                torch.tensor(self.model.config.latents_std).view(1, -1, 1, 1, 1).to(x)
            )
            output = (x - latents_mean) / latents_std
        else:
            raise NotImplementedError(
                f"norm_mode: {self.norm_mode} is not implemented."
            )
        return output

    @apply_forward_hook
    def decode(self, x, **kwargs):
        if self._tiling:
            return self.tiled_decode(x, **kwargs)
        else:
            return self.model.decode(x).sample

    @abstractmethod
    def tiled_decode(self, x, **kwargs):
        pass

