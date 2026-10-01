from einops import rearrange, repeat
import torch
import torch.nn as nn

from mindspeed_mm.models.common.checkpoint import load_checkpoint
from mindspeed_mm.models.common.resnet_block import ResnetBlock2D
from mindspeed_mm.models.common.attention import Conv2dAttnBlock
from mindspeed_mm.models.common.normalize import normalize
from mindspeed_mm.models.common.activations import Sigmoid


class MOVQ(nn.Module):
    def __init__(
        self,
        from_pretrained: str = None,
        double_z=False,
        z_channels=4,
        resolution=256,
        in_channels=3,
        out_ch=3,
        ch=256,
        ch_mult=None,
        num_res_blocks=2,
        attn_resolutions=None,
        dropout=0.0,
        n_embed=16384,
        embed_dim=4,
        **kwargs
    ):
        super().__init__()
        self.encoder = Encoder(
            double_z=double_z,
            z_channels=z_channels,
            resolution=resolution,
            in_channels=in_channels,
            out_ch=out_ch,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
        )
        self.quantize = VectorQuantizer(n_embed, embed_dim)
        self.quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)
        # load_checkpoint
        if from_pretrained is not None:
            load_checkpoint(self, from_pretrained)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        info = self.quantize(h)
        return info


class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=None, num_res_blocks=2,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels=3,
                 resolution=256, z_channels=4, double_z=True, use_sdp_attention=False,
                 **ignore_kwargs):
        super().__init__()
        if ch_mult is None:
            ch_mult = (1, 2, 4, 8)
        if attn_resolutions is None:
            attn_resolutions = [32]
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock2D(in_channels=block_in, out_channels=block_out, dropout=dropout, act_type="swish"))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(Conv2dAttnBlock(block_in, block_in, use_sdp_attention=use_sdp_attention))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock2D(in_channels=block_in, out_channels=block_in, dropout=dropout, act_type="swish")
        self.mid.attn_1 = Conv2dAttnBlock(block_in, block_in, use_sdp_attention=use_sdp_attention)
        self.mid.block_2 = ResnetBlock2D(in_channels=block_in, out_channels=block_in, dropout=dropout, act_type="swish")

        # end
        self.norm_out = normalize(block_in)
        self.nonlinearity = Sigmoid()
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = self.nonlinearity(h)
        h = self.conv_out(h)
        return h


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        return min_encoding_indices
