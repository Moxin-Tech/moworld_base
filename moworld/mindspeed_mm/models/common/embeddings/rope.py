import torch
from torch import nn
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding, get_pos_emb_on_this_cp_rank
from megatron.training.global_vars import get_args


class PositionGetter3D:

    def __init__(self, atten_layout="BSH"):
        self.cache_positions = {}
        self.atten_layout = atten_layout

    @staticmethod
    def check_type(param):
        if isinstance(param, torch.Tensor):
            param = param.item()
        return param

    def __call__(self, b, t, h, w, device):
        b = self.check_type(b)
        t = self.check_type(t)
        h = self.check_type(h)
        w = self.check_type(w)

        if not (b, t, h, w) in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            z = torch.arange(t, device=device)
            pos = torch.cartesian_prod(z, y, x)
            if self.atten_layout == "SBH":
                pos = pos.reshape(t * h * w, 3).transpose(0, 1).reshape(3, -1, 1).contiguous().expand(3, -1, b).clone()
            elif self.atten_layout == "BSH":
                pos = pos.reshape(t * h * w, 3).transpose(0, 1).reshape(3, 1, -1).contiguous().expand(3, b, -1).clone()
            else:
                raise ValueError(f"Unsupported layout type: {self.atten_layout}")
            poses = (pos[0].contiguous(), pos[1].contiguous(), pos[2].contiguous())
            max_poses = (int(poses[0].max()), int(poses[1].max()), int(poses[2].max()))

            self.cache_positions[b, t, h, w] = (poses, max_poses)
        pos = self.cache_positions[b, t, h, w]
        return pos


class RoPE3D(nn.Module):

    def __init__(self, freq=10000.0, interpolation_scale=(1, 1, 1)):
        super().__init__()
        self.base = freq
        self.interpolation_scale_t, self.interpolation_scale_h, self.interpolation_scale_w = interpolation_scale
        self.cache = {}

    def get_cos_sin(self, dim, seq_len, device, dtype, interpolation_scale=1):
        if (dim, seq_len, device, dtype) not in self.cache:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float().to(device) / dim))
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype) / interpolation_scale
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            freqs = torch.cat((freqs, freqs), dim=-1)
            cos = freqs.cos()  # (Seq, Dim)
            sin = freqs.sin()
            self.cache[dim, seq_len, device, dtype] = (cos, sin)
        return self.cache[dim, seq_len, device, dtype]

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope1d(self, tokens, pos1d, cos, sin):
        if pos1d.ndim != 2:
            raise AssertionError("pos1d.ndim must be 2")
        cos = F.embedding(pos1d, cos)[:, :, None, :]
        sin = F.embedding(pos1d, sin)[:, :, None, :]

        return (tokens * cos) + (self.rotate_half(tokens) * sin)

    def forward(self, tokens, positions):
        """
        input:
            tokens: batch_size x nheads x ntokens x dim
            positions: batch_size x ntokens x 3 (t, y and x position of each token)
        output:
            tokens after applying RoPE3D (batch_size x nheads x ntokens x x dim)
        """
        if tokens.size(3) % 3 != 0:
            raise AssertionError("number of dimensions should be a multiple of three")
        dim = tokens.size(3) // 3
        poses, max_poses = positions
        if len(poses) != 3 or poses[0].ndim != 2:  # [Batch, Seq, 3]
            raise AssertionError("poses shape error")
        cos_t, sin_t = self.get_cos_sin(dim, max_poses[0] + 1, tokens.device, tokens.dtype, self.interpolation_scale_t)
        cos_y, sin_y = self.get_cos_sin(dim, max_poses[1] + 1, tokens.device, tokens.dtype, self.interpolation_scale_h)
        cos_x, sin_x = self.get_cos_sin(dim, max_poses[2] + 1, tokens.device, tokens.dtype, self.interpolation_scale_w)
        # split features into three along the feature dimension, and apply rope1d on each half
        t, y, x = tokens.chunk(3, dim=-1)
        t = self.apply_rope1d(t, poses[0], cos_t, sin_t)
        y = self.apply_rope1d(y, poses[1], cos_y, sin_y)
        x = self.apply_rope1d(x, poses[2], cos_x, sin_x)
        tokens = torch.cat((t, y, x), dim=-1)
        return tokens


class DynamicRotaryEmbedding(RotaryEmbedding):
    def __init__(self,
                config,
                kv_channels,
                rotary_percent,
                rotary_interleaved=False,
                seq_len_interpolation_factor=None,
                rotary_base=10000,
                use_cpu_initialization=False,
                seq_len=None
            ):
        super().__init__(
            kv_channels=kv_channels,
            rotary_percent=rotary_percent,
            rotary_interleaved=rotary_interleaved,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            rotary_base=rotary_base,
        )
        dim = kv_channels
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)

        max_position_embeddings = config.max_position_embeddings
        factor = config.rope_scaling.factor if hasattr(config.rope_scaling, 'factor') else 1.0

        seq_len = seq_len if seq_len is not None and seq_len > max_position_embeddings else max_position_embeddings
        device = "cpu" if use_cpu_initialization else torch.cuda.current_device()

        rotary_base = rotary_base * ((factor * seq_len / max_position_embeddings) - (factor - 1)) ** (dim / (dim - 2))
        self.inv_freq = 1.0 / (rotary_base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))

    def forward(self, max_seq_len: int, offset: int = 0):
        """Forward pass of RoPE embedding.

        Args:
            max_seq_len (int): Maximum size of sequence
            offset (int, optional): _description_. Defaults to 0.

        Returns:
            Tensor: Embeddings after applying RoPE.
        """
        if self.inv_freq.device.type == 'cpu':
            # move `inv_freq` to GPU once at the first micro-batch forward pass
            self.inv_freq = self.inv_freq.to(device=torch.cuda.current_device())
        seq = (
            torch.arange(max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            + offset
        )

        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor

        freqs = torch.outer(seq, self.inv_freq)
        # first part even vector components, second part odd vector components,
        # 2 * dim in dimension size
        if not self.rotary_interleaved:
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            emb = torch.stack((freqs.view(-1, 1), freqs.view(-1, 1)), dim=-1).view(
                freqs.shape[0], -1
            )
        # emb [seq_length, .., dim]
        emb = emb[:, None, None, :]
        if parallel_state.get_context_parallel_world_size() > 1 and get_args().context_parallel_algo == "ulysses_cp_algo":
            # slice rotary_pos_emb along sequence dimension and select the partition of the current CP rank
            emb = get_pos_emb_on_this_cp_rank(emb, 0)
        return emb

    def get_rotary_seq_len(
        self,
        inference_context,
        transformer,
        transformer_input,
        transformer_config,
        inference_params=None,
    ) -> float:
        """Function to get the rotary sequence length.

        Args:
            inference_params : Used during Inference time
            transformer (TransformerBlock): The transformer block (decoder/encoder) used by the model
            transformer_input (Tensor): _description_
            transformer_config (TransformerConfig): Transformer config used by the model

        Returns:
            float: The rotary sequence length
        """
        if inference_params is not None:
            rotary_seq_len = inference_params.max_sequence_length
        else:
            if transformer.input_tensor is not None:
                rotary_seq_len = transformer.input_tensor.size(0)
            else:
                rotary_seq_len = transformer_input.size(0)

            if transformer_config.sequence_parallel:
                rotary_seq_len = torch.Tensor([rotary_seq_len]).cuda()
                torch.distributed.all_reduce(rotary_seq_len, group=parallel_state.get_tensor_model_parallel_group())
                rotary_seq_len = rotary_seq_len.item()

        if get_args().context_parallel_algo == "ulysses_cp_algo":
            rotary_seq_len *= transformer_config.context_parallel_size

        return rotary_seq_len
