from einops import rearrange, repeat
import torch
from mindspeed_mm.models.ae.diffusers_ae_model import DiffusersAEModel


class WanVideoVAE(DiffusersAEModel):
    def __init__(self, **config):
        super().__init__(model_name="AutoencoderKLWan", config=config)
        self.upsampling_factor = 8

    def _build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + 1) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + 1) / border_width, dims=(0,))
        return x

    def _build_mask(self, data, is_bound, border_width):
        _, _, _, H, W = data.shape
        h = self._build_1d_mask(H, is_bound[0], is_bound[1], border_width[0])
        w = self._build_1d_mask(W, is_bound[2], is_bound[3], border_width[1])

        h = repeat(h, "H -> H W", H=H, W=W)
        w = repeat(w, "W -> H W", H=H, W=W)

        mask = torch.stack([h, w]).min(dim=0).values
        mask = rearrange(mask, "H W -> 1 1 1 H W")
        return mask

    def tiled_encode(self, x, **kwargs):
        _, _, T, H, W = x.shape
        size_h, size_w = self.tiling_param["tile_size"]
        stride_h, stride_w = self.tiling_param["tile_stride"]
        size_h, size_w = (
            size_h * self.upsampling_factor,
            size_w * self.upsampling_factor,
        )
        stride_h, stride_w = (
            stride_h * self.upsampling_factor,
            stride_w * self.upsampling_factor,
        )

        # Split tasks
        tasks = []
        for h in range(0, H, stride_h):
            if h - stride_h >= 0 and h - stride_h + size_h >= H:
                continue
            for w in range(0, W, stride_w):
                if w - stride_w >= 0 and w - stride_w + size_w >= W:
                    continue
                h_, w_ = h + size_h, w + size_w
                tasks.append((h, h_, w, w_))

        out_T = (T + 3) // 4
        weight = torch.zeros((1, 1, out_T, H // self.upsampling_factor, W // self.upsampling_factor)).to(x)
        values = torch.zeros((1, 16, out_T, H // self.upsampling_factor, W // self.upsampling_factor)).to(x)

        for h, h_, w, w_ in tasks:
            hidden_states_batch = x[:, :, :, h:h_, w:w_]
            hidden_states_batch = self.model.encode(hidden_states_batch).latent_dist
            hidden_states_batch = hidden_states_batch.sample() if self.do_sample else hidden_states_batch.mode()

            mask = self._build_mask(
                hidden_states_batch,
                is_bound=(h == 0, h_ >= H, w == 0, w_ >= W),
                border_width=(
                    (size_h - stride_h) // self.upsampling_factor,
                    (size_w - stride_w) // self.upsampling_factor,
                ),
            ).to(x)

            target_h = h // self.upsampling_factor
            target_w = w // self.upsampling_factor
            values[
                :,
                :,
                :,
                target_h:target_h + hidden_states_batch.shape[3],
                target_w:target_w + hidden_states_batch.shape[4],
            ] += (
                hidden_states_batch * mask
            )
            weight[
                :,
                :,
                :,
                target_h:target_h + hidden_states_batch.shape[3],
                target_w:target_w + hidden_states_batch.shape[4],
            ] += mask
        values = values / weight
        return values

    def tiled_decode(self, x, **kwargs):
        _, _, T, H, W = x.shape
        size_h, size_w = self.tiling_param["tile_size"]
        stride_h, stride_w = self.tiling_param["tile_stride"]

        # Split tasks
        tasks = []
        for h in range(0, H, stride_h):
            if h - stride_h >= 0 and h - stride_h + size_h >= H:
                continue
            for w in range(0, W, stride_w):
                if w - stride_w >= 0 and w - stride_w + size_w >= W:
                    continue
                h_, w_ = h + size_h, w + size_w
                tasks.append((h, h_, w, w_))

        out_T = T * 4 - 3
        weight = torch.zeros((1, 1, out_T, H * self.upsampling_factor, W * self.upsampling_factor)).to(x)
        values = torch.zeros((1, 3, out_T, H * self.upsampling_factor, W * self.upsampling_factor)).to(x)

        for h, h_, w, w_ in tasks:
            hidden_states_batch = x[:, :, :, h:h_, w:w_]
            hidden_states_batch = self.model.decode(hidden_states_batch).sample

            mask = self._build_mask(
                hidden_states_batch,
                is_bound=(h == 0, h_ >= H, w == 0, w_ >= W),
                border_width=(
                    (size_h - stride_h) * self.upsampling_factor,
                    (size_w - stride_w) * self.upsampling_factor,
                ),
            ).to(x)

            target_h = h * self.upsampling_factor
            target_w = w * self.upsampling_factor
            values[
                :,
                :,
                :,
                target_h:target_h + hidden_states_batch.shape[3],
                target_w:target_w + hidden_states_batch.shape[4],
            ] += (
                hidden_states_batch * mask
            )
            weight[
                :,
                :,
                :,
                target_h:target_h + hidden_states_batch.shape[3],
                target_w:target_w + hidden_states_batch.shape[4],
            ] += mask
        values = values / weight
        values = values.clamp_(-1, 1)
        return values

    def get_tiling_state(self):
        return self._tiling
