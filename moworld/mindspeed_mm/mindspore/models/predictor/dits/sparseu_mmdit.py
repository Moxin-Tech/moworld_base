from typing import Any, Dict, Optional, Tuple
import torch
from diffusers.utils import is_torch_version
from mindspeed_mm.models.predictor.dits.sparseu_mmdit import BlockForwardInputs, create_custom_forward, maybe_clamp_tensor


def block_forward(
    self,
    block,
    hidden_states,
    attention_mask,
    encoder_hidden_states,
    inputs: BlockForwardInputs,
    ):
    """
    block_forward.
    mindspore is not support use_reentrant=False.
    """
    embedded_timestep = inputs.embedded_timestep
    frames = inputs.frames
    height = inputs.height
    width = inputs.width
    video_rotary_emb = inputs.video_rotary_emb
    if self.training and self.gradient_checkpointing:
        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
        hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
        create_custom_forward(block),
        hidden_states,
        attention_mask,
        encoder_hidden_states,
        embedded_timestep,
        frames,
        height,
        width,
        video_rotary_emb,
        **ckpt_kwargs
        )
    else:
        hidden_states, encoder_hidden_states = block(
        hidden_states,
        attention_mask=attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        inputs=inputs
        )
    return hidden_states, encoder_hidden_states


def sparsemmditblock_forward(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    embedded_timestep: Optional[torch.FloatTensor] = None,
    frames: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    video_rotary_emb: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> torch.FloatTensor:
    """
    sparsemmditblock_forward.
    mindspore is not support use_reentrant=False.
    """

    # 0. Prepare rope embedding
    vis_seq_length, batch_size = hidden_states.shape[:2]

    # 1. norm & scale & shift
    hidden_states = maybe_clamp_tensor(hidden_states, training=self.training)
    encoder_hidden_states = maybe_clamp_tensor(encoder_hidden_states, training=self.training)
    norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
        hidden_states, encoder_hidden_states, embedded_timestep
    )

    # 2. MM Attention
    attn_hidden_states, attn_encoder_hidden_states = self.attn1(
        norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        frames=frames,
        height=height,
        width=width,
        attention_mask=attention_mask,
        video_rotary_emb=video_rotary_emb,
    )

    weight_dtype = hidden_states.dtype
    if gate_msa.dtype != torch.float32 or enc_gate_msa.dtype != torch.float32:
        raise ValueError("Gate must be float32.")

    # 3. residual & gate
    hidden_states = hidden_states.float() + gate_msa * attn_hidden_states.float()
    hidden_states = hidden_states.to(weight_dtype)
    if not self.context_pre_only:
        encoder_hidden_states = encoder_hidden_states.float() + enc_gate_msa * attn_encoder_hidden_states.float()
        encoder_hidden_states = encoder_hidden_states.to(weight_dtype)

    # 4. norm & scale & shift
    hidden_states = maybe_clamp_tensor(hidden_states, training=self.training)
    if not self.context_pre_only:
        encoder_hidden_states = maybe_clamp_tensor(encoder_hidden_states, training=self.training)

    norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
        hidden_states, encoder_hidden_states, embedded_timestep
    )
    weight_dtype = hidden_states.dtype
    if gate_ff.dtype != torch.float32 or enc_gate_ff.dtype != torch.float32:
        raise AssertionError("Gate FFN should be float32")

    if self.double_ff:
        # 5. FFN
        vis_ff_output = self.ff(norm_hidden_states)
        # 6. residual & gate
        hidden_states = hidden_states.float() + gate_ff * vis_ff_output.float()
        hidden_states = hidden_states.to(weight_dtype)
        if self.ff_enc is not None:
            enc_ff_output = self.ff_enc(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.float() + enc_gate_ff * enc_ff_output.float()
            encoder_hidden_states = encoder_hidden_states.to(weight_dtype)
    else:
        # 5. FFN
        norm_hidden_states = torch.cat([norm_hidden_states, norm_encoder_hidden_states], dim=0)
        ff_output = self.ff(norm_hidden_states)
        # 6. residual & gate
        hidden_states = hidden_states.float() + gate_ff * ff_output[:vis_seq_length].float()
        hidden_states = hidden_states.to(weight_dtype)
        encoder_hidden_states = encoder_hidden_states.to(weight_dtype)
        if not self.context_pre_only:
            encoder_hidden_states = encoder_hidden_states.float() + enc_gate_ff * ff_output[vis_seq_length:].float()
            encoder_hidden_states = encoder_hidden_states.to(weight_dtype)

    return hidden_states, encoder_hidden_states
