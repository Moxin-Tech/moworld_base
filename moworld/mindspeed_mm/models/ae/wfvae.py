import os
from collections import deque

import torch.nn as nn
import torch
from megatron.core import mpu

from mindspeed_mm.models.common.checkpoint import load_checkpoint
from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.models.common.normalize import normalize
from mindspeed_mm.models.common.attention import WfCausalConv3dAttnBlock
from mindspeed_mm.models.common.distrib import DiagonalGaussianDistribution
from mindspeed_mm.models.common.wavelet import (
    HaarWaveletTransform2D,
    HaarWaveletTransform3D,
    InverseHaarWaveletTransform2D,
    InverseHaarWaveletTransform3D
)
from mindspeed_mm.models.common.conv import Conv2d, WfCausalConv3d
from mindspeed_mm.models.common.resnet_block import ResnetBlock2D, ResnetBlock3D
from mindspeed_mm.models.common.updownsample import (
    Upsample,
    Downsample,
    Spatial2xTime2x3DDownsample,
    CachedCausal3DUpsample
)
from mindspeed_mm.utils.utils import (
    is_context_parallel_initialized,
    initialize_context_parallel,
    get_context_parallel_group,
    get_context_parallel_rank,
)
from mindspeed_mm.models.common.communications import split_tensor, collect_tensors_across_ranks

WFVAE_MODULE_MAPS = {
    "Conv2d": Conv2d,
    "ResnetBlock2D": ResnetBlock2D,
    "CausalConv3d": WfCausalConv3d,
    "AttnBlock3D": WfCausalConv3dAttnBlock,
    "ResnetBlock3D": ResnetBlock3D,
    "Downsample": Downsample,
    "HaarWaveletTransform2D": HaarWaveletTransform2D,
    "HaarWaveletTransform3D": HaarWaveletTransform3D,
    "InverseHaarWaveletTransform2D": InverseHaarWaveletTransform2D,
    "InverseHaarWaveletTransform3D": InverseHaarWaveletTransform3D,
    "Spatial2xTime2x3DDownsample": Spatial2xTime2x3DDownsample,
    "Upsample": Upsample,
    "Spatial2xTime2x3DUpsample": CachedCausal3DUpsample,
    "WfCausalConv3dAttnBlock": WfCausalConv3dAttnBlock,
}


def model_name_to_cls(model_name):
    if model_name in WFVAE_MODULE_MAPS:
        return WFVAE_MODULE_MAPS[model_name]
    else:
        raise ValueError(f"Model name {model_name} not supported")


class Encoder(MultiModalModule):

    def __init__(
            self,
            latent_dim: int = 8,
            base_channels: int = 128,
            num_resblocks: int = 2,
            energy_flow_hidden_size: int = 64,
            dropout: float = 0.0,
            use_attention: bool = True,
            atten_block: str = "WfCausalConv3dAttnBlock",
            norm_type: str = "groupnorm",
            l1_dowmsample_block: str = "Downsample",
            l1_downsample_wavelet: str = "HaarWaveletTransform2D",
            l2_dowmsample_block: str = "Spatial2xTime2x3DDownsample",
            l2_downsample_wavelet: str = "HaarWaveletTransform3D",
            enable_vae_cp: bool = False,
            training: bool = False
    ) -> None:
        super().__init__(config=None)
        self.activation = nn.SiLU()

        self.down1 = nn.Sequential(
            Conv2d(24, base_channels, kernel_size=3, stride=1, padding=1),
            *[
                ResnetBlock2D(
                    in_channels=base_channels,
                    out_channels=base_channels,
                    dropout=dropout,
                    norm_type=norm_type,
                )
                for _ in range(num_resblocks)
            ],
            model_name_to_cls(l1_dowmsample_block)(
                in_channels=base_channels,
                out_channels=base_channels,
                conv_type="WfCausalConv3d"
            ),
        )

        self.down2 = nn.Sequential(
            Conv2d(
                base_channels + energy_flow_hidden_size,
                base_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            *[
                ResnetBlock3D(
                    in_channels=base_channels * 2,
                    out_channels=base_channels * 2,
                    dropout=dropout,
                    norm_type=norm_type,
                    conv_type="WfCausalConv3d",
                    enable_vae_cp=enable_vae_cp
                )
                for _ in range(num_resblocks)
            ],
            model_name_to_cls(l2_dowmsample_block)(base_channels * 2, base_channels * 2, conv_type="WfCausalConv3d",
                                                   enable_vae_cp=enable_vae_cp),
        )
        # Connection
        if l1_downsample_wavelet.endswith("2D"):
            l1_channels = 12
        elif l1_downsample_wavelet.endswith("3D"):
            l1_channels = 24
        else:
            raise ValueError('l1_downsample_wavelet only Support `HaarWaveletTransform2D` and `HaarWaveletTransform3D`')

        self.connect_l1 = Conv2d(
            l1_channels, energy_flow_hidden_size, kernel_size=3, stride=1, padding=1
        )
        self.connect_l2 = Conv2d(
            24, energy_flow_hidden_size, kernel_size=3, stride=1, padding=1
        )
        # Mid
        mid_layers = [
            ResnetBlock3D(
                in_channels=base_channels * 2 + energy_flow_hidden_size,
                out_channels=base_channels * 4,
                dropout=dropout,
                norm_type=norm_type,
                conv_type="WfCausalConv3d",
                enable_vae_cp=enable_vae_cp
            ),
            ResnetBlock3D(
                in_channels=base_channels * 4,
                out_channels=base_channels * 4,
                dropout=dropout,
                norm_type=norm_type,
                conv_type="WfCausalConv3d",
                enable_vae_cp=enable_vae_cp
            ),
        ]
        if use_attention:
            mid_layers.insert(
                1, model_name_to_cls(atten_block)(
                    in_channels=base_channels * 4,
                    out_channels=base_channels * 4,
                    norm_type=norm_type)
            )
        self.mid = nn.Sequential(*mid_layers)

        self.norm_out = normalize(base_channels * 4, norm_type=norm_type)
        self.conv_out = WfCausalConv3d(
            base_channels * 4, latent_dim * 2, kernel_size=3, stride=1, padding=1, parallel_input=False
        )
        if not training:
            self.wavelet_tranform_in = model_name_to_cls(l2_downsample_wavelet)()
        self.wavelet_transform_l1 = model_name_to_cls(l1_downsample_wavelet)()
        self.wavelet_transform_l2 = model_name_to_cls(l2_downsample_wavelet)()

    def forward(self, coeffs):
        if not self.training:
            coeffs = split_tensor(coeffs, get_context_parallel_group(), get_context_parallel_rank(), first_padding=3)
            coeffs = self.wavelet_tranform_in(coeffs)
        else:
            wt = HaarWaveletTransform3D().to(coeffs.device, dtype=coeffs.dtype)
            coeffs = wt(coeffs)
        l1_coeffs = coeffs[:, :3]
        l1_coeffs = self.wavelet_transform_l1(l1_coeffs)
        l1 = self.connect_l1(l1_coeffs)
        l2_coeffs = self.wavelet_transform_l2(l1_coeffs[:, :3])
        l2 = self.connect_l2(l2_coeffs)

        h = self.down1(coeffs)
        h = torch.concat([h, l1], dim=1)
        h = self.down2(h)
        h = torch.concat([h, l2], dim=1)
        h = self.mid(h)

        h = self.norm_out(h)
        h = self.activation(h)
        h = torch.cat(collect_tensors_across_ranks(h, get_context_parallel_group()), dim=2)
        h = self.conv_out(h)
        return h


class Decoder(MultiModalModule):

    def __init__(
            self,
            latent_dim: int = 8,
            base_channels: int = 128,
            num_resblocks: int = 2,
            dropout: float = 0.0,
            energy_flow_hidden_size: int = 128,
            use_attention: bool = True,
            atten_block: str = "WfCausalConv3dAttnBlock",
            norm_type: str = "groupnorm",
            t_interpolation: str = "nearest",
            connect_res_layer_num: int = 2,
            l1_upsample_block: str = "Upsample",
            l1_upsample_wavelet: str = "InverseHaarWaveletTransform2D",
            l2_upsample_block: str = "Spatial2xTime2x3DUpsample",
            l2_upsample_wavelet: str = "InverseHaarWaveletTransform3D",
            enable_vae_cp: bool = False,
            training: bool = False
    ) -> None:
        super().__init__(config=None)
        self.energy_flow_hidden_size = energy_flow_hidden_size
        self.activation = nn.SiLU()
        self.conv_in = WfCausalConv3d(
            latent_dim, base_channels * 4, kernel_size=3, stride=1, padding=1
        )
        mid_layers = [
            ResnetBlock3D(
                in_channels=base_channels * 4,
                out_channels=base_channels * 4,
                dropout=dropout,
                norm_type=norm_type,
                conv_type="WfCausalConv3d"
            ),
            ResnetBlock3D(
                in_channels=base_channels * 4,
                out_channels=base_channels * 4 + energy_flow_hidden_size,
                dropout=dropout,
                norm_type=norm_type,
                conv_type="WfCausalConv3d"
            ),
        ]
        if use_attention:
            mid_layers.insert(
                1, model_name_to_cls(atten_block)(
                    in_channels=base_channels * 4,
                    out_channels=base_channels * 4,
                    norm_type=norm_type)
            )
        self.mid = nn.Sequential(*mid_layers)

        self.up2 = nn.Sequential(
            *[
                ResnetBlock3D(
                    in_channels=base_channels * 4,
                    out_channels=base_channels * 4,
                    dropout=dropout,
                    norm_type=norm_type,
                    conv_type="WfCausalConv3d"
                )
                for _ in range(num_resblocks)
            ],
            model_name_to_cls(l2_upsample_block)(
                base_channels * 4, base_channels * 4, t_interpolation=t_interpolation
            ),
            ResnetBlock3D(
                in_channels=base_channels * 4,
                out_channels=base_channels * 4 + energy_flow_hidden_size,
                dropout=dropout,
                norm_type=norm_type,
                conv_type="WfCausalConv3d"
            ),
        )
        self.up1 = nn.Sequential(
            *[
                ResnetBlock3D(
                    in_channels=base_channels * (4 if i == 0 else 2),
                    out_channels=base_channels * 2,
                    dropout=dropout,
                    norm_type=norm_type,
                    conv_type="WfCausalConv3d"
                )
                for i in range(num_resblocks)
            ],
            model_name_to_cls(l1_upsample_block)(in_channels=base_channels * 2, out_channels=base_channels * 2),
            ResnetBlock3D(
                in_channels=base_channels * 2,
                out_channels=base_channels * 2,
                dropout=dropout,
                norm_type=norm_type,
                conv_type="WfCausalConv3d",
                enable_vae_cp=enable_vae_cp
            ),
        )
        self.layer = nn.Sequential(
            *[
                ResnetBlock3D(
                    in_channels=base_channels * (2 if i == 0 else 1),
                    out_channels=base_channels,
                    dropout=dropout,
                    norm_type=norm_type,
                    conv_type="WfCausalConv3d"
                )
                for i in range(2)
            ],
        )
        # Connection
        if l1_upsample_wavelet.endswith("2D"):
            l1_channels = 12
        elif l1_upsample_wavelet.endswith("3D"):
            l1_channels = 24
        else:
            raise ValueError('l1_upsample_wavelet only Support `InverseHaarWaveletTransform2D` and `InverseHaarWaveletTransform3D`')

        self.connect_l1 = nn.Sequential(
            *[
                ResnetBlock3D(
                    in_channels=energy_flow_hidden_size,
                    out_channels=energy_flow_hidden_size,
                    dropout=dropout,
                    norm_type=norm_type,
                    conv_type="WfCausalConv3d",
                    enable_vae_cp=enable_vae_cp
                )
                for _ in range(connect_res_layer_num)
            ],
            Conv2d(energy_flow_hidden_size, l1_channels, kernel_size=3, stride=1, padding=1),
        )
        self.connect_l2 = nn.Sequential(
            *[
                ResnetBlock3D(
                    in_channels=energy_flow_hidden_size,
                    out_channels=energy_flow_hidden_size,
                    dropout=dropout,
                    norm_type=norm_type,
                    conv_type="WfCausalConv3d"
                )
                for _ in range(connect_res_layer_num)
            ],
            Conv2d(energy_flow_hidden_size, 24, kernel_size=3, stride=1, padding=1),
        )
        # Out
        self.norm_out = normalize(base_channels, norm_type=norm_type)
        self.conv_out = Conv2d(base_channels, 24, kernel_size=3, stride=1, padding=1)

        if not training:
            self.inverse_wavelet_transform_out = model_name_to_cls(l2_upsample_wavelet)()
        self.inverse_wavelet_transform_l1 = model_name_to_cls(l1_upsample_wavelet)()
        self.inverse_wavelet_transform_l2 = model_name_to_cls(l2_upsample_wavelet)()

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid(h)
        l2_coeffs = self.connect_l2(h[:, -self.energy_flow_hidden_size:])
        l2 = self.inverse_wavelet_transform_l2(l2_coeffs)
        h = self.up2(h[:, : -self.energy_flow_hidden_size])

        l1_coeffs = h[:, -self.energy_flow_hidden_size:]
        l1_coeffs = self.connect_l1(l1_coeffs)
        l1_coeffs[:, :3] = l1_coeffs[:, :3] + l2
        l1 = self.inverse_wavelet_transform_l1(l1_coeffs)

        h = self.up1(h[:, : -self.energy_flow_hidden_size])
        h = self.layer(h)
        h = self.norm_out(h)
        h = self.activation(h)
        h = self.conv_out(h)
        h[:, :3] = h[:, :3] + l1

        if not self.training:
            dec = self.inverse_wavelet_transform_out(h)
        else:
            wt = InverseHaarWaveletTransform3D().to(h.device, dtype=h.dtype)
            dec = wt(h)

        return dec


class WFVAE(MultiModalModule):

    def __init__(
            self,
            from_pretrained: str = None,
            latent_dim: int = 8,
            base_channels: int = 128,
            encoder_num_resblocks: int = 2,
            encoder_energy_flow_hidden_size: int = 64,
            decoder_num_resblocks: int = 2,
            decoder_energy_flow_hidden_size: int = 128,
            use_attention: bool = True,
            atten_block: str = "WfCausalConv3dAttnBlock",
            dropout: float = 0.0,
            norm_type: str = "groupnorm",
            t_interpolation: str = "nearest",
            vae_scale_factor: list = None,
            use_tiling: bool = False,
            connect_res_layer_num: int = 2,
            l1_dowmsample_block: str = "Downsample",
            l1_downsample_wavelet: str = "HaarWaveletTransform2D",
            l2_dowmsample_block: str = "Spatial2xTime2x3DDownsample",
            l2_downsample_wavelet: str = "HaarWaveletTransform3D",
            l1_upsample_block: str = "Upsample",
            l1_upsample_wavelet: str = "InverseHaarWaveletTransform2D",
            l2_upsample_block: str = "Spatial2xTime2x3DUpsample",
            l2_upsample_wavelet: str = "InverseHaarWaveletTransform3D",
            scale: list = None,
            shift: list = None,
            dtype: str = "fp32",
            vae_cp_size: int = 1,
            t_chunk_enc: int = 8,
            t_chunk_dec: int = 2,
            t_upsample_times=2,
            **kwargs
    ) -> None:
        super().__init__(config=None)
        # Hardcode for now
        self.t_chunk_enc = t_chunk_enc
        self.t_chunk_dec = t_chunk_dec
        self.t_upsample_times = t_upsample_times
        self.use_quant_layer = False
        self.vae_scale_factor = vae_scale_factor

        self.vae_cp_size = vae_cp_size
        if self.vae_cp_size > 0:
            if not is_context_parallel_initialized():
                initialize_context_parallel(self.vae_cp_size)

        self.enable_vae_cp = self.vae_cp_size > 1
        if self.enable_vae_cp and mpu.get_pipeline_model_parallel_world_size() > 1:
            raise ValueError("VAE-CP can not be enabled with PP.")

        self.encoder = Encoder(
            latent_dim=latent_dim,
            base_channels=base_channels,
            num_resblocks=encoder_num_resblocks,
            energy_flow_hidden_size=encoder_energy_flow_hidden_size,
            dropout=dropout,
            use_attention=use_attention,
            norm_type=norm_type,
            atten_block=atten_block,
            l1_dowmsample_block=l1_dowmsample_block,
            l1_downsample_wavelet=l1_downsample_wavelet,
            l2_dowmsample_block=l2_dowmsample_block,
            l2_downsample_wavelet=l2_downsample_wavelet,
            enable_vae_cp=self.enable_vae_cp,
            training=kwargs.get("training", False)
        )
        self.decoder = Decoder(
            latent_dim=latent_dim,
            base_channels=base_channels,
            num_resblocks=decoder_num_resblocks,
            energy_flow_hidden_size=decoder_energy_flow_hidden_size,
            dropout=dropout,
            use_attention=use_attention,
            norm_type=norm_type,
            t_interpolation=t_interpolation,
            connect_res_layer_num=connect_res_layer_num,
            atten_block=atten_block,
            l1_upsample_block=l1_upsample_block,
            l1_upsample_wavelet=l1_upsample_wavelet,
            l2_upsample_block=l2_upsample_block,
            l2_upsample_wavelet=l2_upsample_wavelet,
            training=kwargs.get("training", False)
        )
        self.set_parameters(dtype, scale, shift)
        self.configure_cache_offset(l1_dowmsample_block)

        if from_pretrained is not None:
            load_checkpoint(self, from_pretrained)

        self.enable_tiling(use_tiling)
        if self.enable_vae_cp:
            self.dp_group_nums = torch.distributed.get_world_size() // mpu.get_data_parallel_world_size()

    def set_parameters(self, dtype, scale, shift):
        self.dtype = dtype
        if scale is not None and shift is not None:
            self.register_buffer('shift', torch.tensor(shift, dtype=self.dtype)[None, :, None, None, None])
            self.register_buffer('scale', torch.tensor(scale, dtype=self.dtype)[None, :, None, None, None])
        else:
            self.register_buffer("shift", torch.zeros(1, 8, 1, 1, 1))
            self.register_buffer("scale", torch.tensor([0.18215] * 8)[None, :, None, None, None])

    # Set cache offset for trilinear lossless upsample.
    def configure_cache_offset(self, l1_dowmsample_block):
        if l1_dowmsample_block == "Downsample":
            self.temporal_uptimes = 4
            self._set_cache_offset([self.decoder.up2, self.decoder.connect_l2, self.decoder.conv_in, self.decoder.mid],
                                   1)
            self._set_cache_offset(
                [self.decoder.up2[-2:], self.decoder.up1, self.decoder.connect_l1, self.decoder.layer],
                self.t_upsample_times)
        else:
            self.temporal_uptimes = 8
            self._set_cache_offset([self.decoder.up2, self.decoder.connect_l2, self.decoder.conv_in, self.decoder.mid],
                                   1)
            self._set_cache_offset([self.decoder.up2[-2:], self.decoder.connect_l1, self.decoder.up1], 2)
            self._set_cache_offset([self.decoder.up1[-2:], self.decoder.layer], 4)

    def get_encoder(self):
        if self.use_quant_layer:
            return [self.quant_conv, self.encoder]
        return [self.encoder]

    def get_decoder(self):
        if self.use_quant_layer:
            return [self.post_quant_conv, self.decoder]
        return [self.decoder]

    def _empty_causal_cached(self, parent):
        for _, module in parent.named_modules():
            if hasattr(module, 'causal_cached'):
                module.causal_cached = deque()

    def _set_first_chunk(self, is_first_chunk=True):
        for module in self.modules():
            if hasattr(module, 'is_first_chunk'):
                module.is_first_chunk = is_first_chunk

    def _set_causal_cached(self, enable_cached=True):
        for _, module in self.named_modules():
            if hasattr(module, 'enable_cached'):
                module.enable_cached = enable_cached

    def _set_cache_offset(self, modules, cache_offset=0):
        for module in modules:
            for submodule in module.modules():
                if hasattr(submodule, 'cache_offset'):
                    submodule.cache_offset = cache_offset

    def build_chunk_start_end(self, t, decoder_mode=False):
        start_end = [[0, 1]]
        start = 1
        end = start
        while True:
            if start >= t:
                break
            end = min(t, end + (self.t_chunk_dec if decoder_mode else self.t_chunk_enc))
            start_end.append([start, end])
            start = end
        return start_end

    def encode(self, x):
        if self.enable_vae_cp and self.vae_cp_size % self.dp_group_nums == 0 and self.vae_cp_size > self.dp_group_nums:
            # loop cp
            data_list = [torch.empty_like(x) for _ in range(self.vae_cp_size)]
            data_list[get_context_parallel_rank()] = x
            torch.distributed.all_gather(data_list, x, group=get_context_parallel_group())
            data_list = data_list[::self.dp_group_nums]
            latents = []
            for data in data_list:
                latents.append(self._encode(data))
            return latents[get_context_parallel_rank() // self.dp_group_nums]
        else:
            return self._encode(x)

    def _encode(self, x):
        self._empty_causal_cached(self.encoder)
        self._set_first_chunk(True)
        dtype = x.dtype

        if self.use_tiling:
            h = self.tile_encode(x)
        else:
            h = self.encoder(x)
            if self.use_quant_layer:
                h = self.quant_conv(h)

        self._empty_causal_cached(self.encoder)
        posterior = DiagonalGaussianDistribution(h)

        if not self.training:
            return (posterior.sample() - self.shift.to(x.device, dtype=dtype)) * self.scale.to(x.device, dtype)
        else:
            return posterior

    def tile_encode(self, x):
        b, c, t, h, w = x.shape

        start_end = self.build_chunk_start_end(t)
        result = []
        for idx, (start, end) in enumerate(start_end):
            self._set_first_chunk(idx == 0)
            chunk = x[:, :, start:end, :, :]
            chunk = self.encoder(chunk)
            if self.use_quant_layer:
                chunk = self.quant_conv(chunk)
            result.append(chunk)

        return torch.cat(result, dim=2)

    def decode(self, z):
        self._empty_causal_cached(self.decoder)
        self._set_first_chunk(True)
        if not self.training:
            z = z / self.scale.to(z.device, dtype=z.dtype) + self.shift.to(z.device, dtype=z.dtype)

        if self.use_tiling:
            dec = self.tile_decode(z)
        else:
            if self.use_quant_layer:
                z = self.post_quant_conv(z)
            dec = self.decoder(z)
        return dec

    def tile_decode(self, x):
        b, c, t, h, w = x.shape
        start_end = self.build_chunk_start_end(t, decoder_mode=True)

        result = []
        for idx, (start, end) in enumerate(start_end):
            self._set_first_chunk(idx == 0)

            if idx != 0 and end + 1 < t:
                chunk = x[:, :, start:end + 1, :, :]
            else:
                chunk = x[:, :, start:end, :, :]

            if self.use_quant_layer:
                chunk = self.post_quant_conv(chunk)
            chunk = self.decoder(chunk)

            if idx != 0 and end + 1 < t:
                chunk = chunk[:, :, :-self.temporal_uptimes]
                result.append(chunk.clone())
            else:
                result.append(chunk.clone())

        return torch.cat(result, dim=2)

    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_last_layer(self):
        if hasattr(self.decoder.conv_out, "conv"):
            return self.decoder.conv_out.conv.weight
        else:
            return self.decoder.conv_out.weight

    def enable_tiling(self, use_tiling: bool = False):
        self.use_tiling = use_tiling
        self._set_causal_cached(use_tiling)

    def disable_tiling(self):
        self.enable_tiling(False)
