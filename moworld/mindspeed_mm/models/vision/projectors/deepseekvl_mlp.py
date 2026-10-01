# --------------------------------------------------------
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
# Copyright (c) 2023 DeepSeek
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig

from mindspeed_mm.models.common.module import MultiModalModule


class MlpProjector(MultiModalModule):
    def __init__(self, cfg):

        super().__init__(cfg)

        self.cfg = cfg

        if cfg.projector_type == "identity":
            modules = nn.Identity()

        elif cfg.projector_type == "linear":
            modules = nn.Linear(cfg.input_dim, cfg.n_embed)

        elif cfg.projector_type == "mlp_gelu":
            mlp_depth = cfg.depth
            modules = [nn.Linear(cfg.input_dim, cfg.n_embed)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed, cfg.n_embed))
            modules = nn.Sequential(*modules)

        elif cfg.projector_type == "downsample_mlp_gelu":
            mlp_depth = cfg.depth
            mlp_ratio = cfg.mlp_ratio
            modules = [nn.Linear(cfg.input_dim * cfg.downsample_ratio * cfg.downsample_ratio, cfg.n_embed * mlp_ratio)]
            for _ in range(1, mlp_depth - 1):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed * mlp_ratio))
            modules.append(nn.GELU())
            modules.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed))
            modules = nn.Sequential(*modules)

        else:
            raise ValueError(f"Unknown projector type: {cfg.projector_type}")

        if cfg.token_pooling:
            self.token_pooling_layer = nn.Linear(cfg.input_dim * 4, cfg.input_dim)

        self.layers = modules

    def forward(self, x):
        if self.cfg.token_pooling:
            batch_size, wxh, channels = x.shape
            w = h = int(wxh ** 0.5)
            x = x.view(batch_size, w, h, channels)
            x = x.permute(0, 3, 1, 2)
            patches = x.unfold(2, 2, 2).unfold(3, 2, 2)
            batch_size, channels, h_patches, w_patches, _, _ = patches.size()
            # Concatenate on channel dim
            patches = patches.contiguous().view(batch_size, channels, h_patches * w_patches, -1)

            # Through linear layer
            patches = patches.permute(0, 2, 1, 3).contiguous()
            patches = patches.view(batch_size, h_patches * w_patches, channels * 4)

            x = self.token_pooling_layer(patches)

        elif self.cfg.projector_type == 'downsample_mlp_gelu':
            bs, hw, input_dim = x.shape
            h = w = int((hw) ** 0.5)

            """compute padding"""
            if h % self.cfg.downsample_ratio:
                pad = self.cfg.downsample_ratio - h % self.cfg.downsample_ratio
            else:
                pad = 0
            x = x.reshape(bs, h, w, input_dim)
            if pad > 0:
                x = F.pad(x, (0, 0, 0, pad, 0, pad), "constant", 0)

            """4 to 1 concat"""
            x = x.permute(0, 3, 1, 2)  # B, C, H, W
            x = F.unfold(x, kernel_size=self.cfg.downsample_ratio, stride=self.cfg.downsample_ratio,
                         padding=0)  # B, C*4, HW // 4
            x = x.permute(0, 2, 1)

        return self.layers(x)


class MlpProjectorConfig(PretrainedConfig):
    model_type = "mlp_projector"
    projector_type: str = "downsample_mlp_gelu"
    input_dim: int = 1152
    n_embed: int = 2048
    depth: int = 2
    mlp_ratio: int = 1
    downsample_ratio: int = 2
    token_pooling: bool = False

    def __init__(
            self,
            projector_type: str = "downsample_mlp_gelu",
            input_dim: int = 1152,
            n_embed: int = 2048,
            depth: int = 2,
            mlp_ratio: int = 1,
            downsample_ratio: int = 2,
            **kwargs
    ):
        self.projector_type = projector_type
        self.input_dim = input_dim
        self.n_embed = n_embed
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.downsample_ratio = downsample_ratio

        super().__init__(**kwargs)


def create_deepseekvl_mlp(
        config,
        **kwargs
):
    config = config.to_dict()
    cfg = MlpProjectorConfig(
        projector_type=config.get("projector_type", "downsample_mlp_gelu"),
        input_dim=config.get("input_dim", 1152),
        n_embed=config.get("n_embed", 2048),
        depth=config.get("depth", 2),
        mlp_ratio=config.get("mlp_ratio", 1),
        downsample_ratio=config.get("downsample_ratio", 2),
    )
    model = MlpProjector(
        cfg=cfg
    )
    return model
