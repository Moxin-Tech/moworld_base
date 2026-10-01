import torch
from ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationType
from ltx_core.model.transformer.transformer_args import TransformerArgs
from ltx_core.utils import rms_norm
from ltx_core.model.transformer.modality import Modality

def block_forward(  # noqa: PLR0915
        self,
        vx: torch.Tensor | None,
        ax: torch.Tensor | None,
        video_enabled: bool = True,
        audio_enabled: bool = True,
        video_timesteps: torch.Tensor | None = None,
        audio_timesteps: torch.Tensor | None = None,
        video_positional_embeddings: torch.Tensor | None = None,
        audio_positional_embeddings: torch.Tensor | None = None,
        video_context: torch.Tensor | None = None,
        video_context_mask: torch.Tensor | None = None,
        audio_context: torch.Tensor | None = None,
        audio_context_mask: torch.Tensor | None = None,
        video_cross_scale_shift_timestep: torch.Tensor | None = None,
        video_cross_gate_timestep: torch.Tensor | None = None,
        audio_cross_scale_shift_timestep: torch.Tensor | None = None,
        audio_cross_gate_timestep: torch.Tensor | None = None,
        video_cross_positional_embeddings: torch.Tensor | None = None,
        audio_cross_positional_embeddings: torch.Tensor | None = None,
        perturbations: BatchedPerturbationConfig | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if vx is None and ax is None:
        raise ValueError("At least one of vx or ax must be provided")

    batch_size = vx.shape[0] if vx is not None else ax.shape[0]

    if perturbations is None:
        perturbations = BatchedPerturbationConfig.empty(batch_size)

    run_vx = vx is not None and video_enabled and vx.numel() > 0
    run_ax = ax is not None and audio_enabled and ax.numel() > 0

    run_a2v = run_vx and (ax is not None and ax.numel() > 0)
    run_v2a = run_ax and (vx is not None and vx.numel() > 0)

    if run_vx:
        vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
            self.scale_shift_table, vx.shape[0], video_timesteps, slice(0, 3)
        )
        if not perturbations.all_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx):
            norm_vx = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_msa) + vshift_msa
            v_mask = perturbations.mask_like(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx, vx)
            vx = vx + self.attn1(norm_vx, pe=video_positional_embeddings) * vgate_msa * v_mask

        vx = vx + self.attn2(rms_norm(vx, eps=self.norm_eps), context=video_context, mask=video_context_mask)

        del vshift_msa, vscale_msa, vgate_msa

    if run_ax:
        ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
            self.audio_scale_shift_table, ax.shape[0], audio_timesteps, slice(0, 3)
        )

        if not perturbations.all_in_batch(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx):
            norm_ax = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_msa) + ashift_msa
            a_mask = perturbations.mask_like(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx, ax)
            ax = ax + self.audio_attn1(norm_ax, pe=audio_positional_embeddings) * agate_msa * a_mask

        ax = ax + self.audio_attn2(rms_norm(ax, eps=self.norm_eps), context=audio_context, mask=audio_context_mask)

        del ashift_msa, ascale_msa, agate_msa

    # Audio - Video cross attention.
    if run_a2v or run_v2a:
        vx_norm3 = rms_norm(vx, eps=self.norm_eps)
        ax_norm3 = rms_norm(ax, eps=self.norm_eps)

        (
            scale_ca_audio_hidden_states_a2v,
            shift_ca_audio_hidden_states_a2v,
            scale_ca_audio_hidden_states_v2a,
            shift_ca_audio_hidden_states_v2a,
            gate_out_v2a,
        ) = self.get_av_ca_ada_values(
            self.scale_shift_table_a2v_ca_audio,
            ax.shape[0],
            audio_cross_scale_shift_timestep,
            audio_cross_gate_timestep,
        )

        (
            scale_ca_video_hidden_states_a2v,
            shift_ca_video_hidden_states_a2v,
            scale_ca_video_hidden_states_v2a,
            shift_ca_video_hidden_states_v2a,
            gate_out_a2v,
        ) = self.get_av_ca_ada_values(
            self.scale_shift_table_a2v_ca_video,
            vx.shape[0],
            video_cross_scale_shift_timestep,
            video_cross_gate_timestep,
        )

        if run_a2v and not perturbations.all_in_batch(PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx):
            vx_scaled = vx_norm3 * (1 + scale_ca_video_hidden_states_a2v) + shift_ca_video_hidden_states_a2v
            ax_scaled = ax_norm3 * (1 + scale_ca_audio_hidden_states_a2v) + shift_ca_audio_hidden_states_a2v
            a2v_mask = perturbations.mask_like(PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx, vx)
            vx = vx + (
                self.audio_to_video_attn(
                    vx_scaled,
                    context=ax_scaled,
                    pe=video_cross_positional_embeddings,
                    k_pe=audio_cross_positional_embeddings,
                )
                * gate_out_a2v
                * a2v_mask
            )

        if run_v2a and not perturbations.all_in_batch(PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx):
            ax_scaled = ax_norm3 * (1 + scale_ca_audio_hidden_states_v2a) + shift_ca_audio_hidden_states_v2a
            vx_scaled = vx_norm3 * (1 + scale_ca_video_hidden_states_v2a) + shift_ca_video_hidden_states_v2a
            v2a_mask = perturbations.mask_like(PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx, ax)
            ax = ax + (
                self.video_to_audio_attn(
                    ax_scaled,
                    context=vx_scaled,
                    pe=audio_cross_positional_embeddings,
                    k_pe=video_cross_positional_embeddings,
                )
                * gate_out_v2a
                * v2a_mask
            )

        del gate_out_a2v, gate_out_v2a
        del (
            scale_ca_video_hidden_states_a2v,
            shift_ca_video_hidden_states_a2v,
            scale_ca_audio_hidden_states_a2v,
            shift_ca_audio_hidden_states_a2v,
            scale_ca_video_hidden_states_v2a,
            shift_ca_video_hidden_states_v2a,
            scale_ca_audio_hidden_states_v2a,
            shift_ca_audio_hidden_states_v2a,
        )

    if run_vx:
        vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
            self.scale_shift_table, vx.shape[0], video_timesteps, slice(3, None)
        )
        vx_scaled = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_mlp) + vshift_mlp
        vx = vx + self.ff(vx_scaled) * vgate_mlp

        del vshift_mlp, vscale_mlp, vgate_mlp

    if run_ax:
        ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
            self.audio_scale_shift_table, ax.shape[0], audio_timesteps, slice(3, None)
        )
        ax_scaled = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_mlp) + ashift_mlp
        ax = ax + self.audio_ff(ax_scaled) * agate_mlp

        del ashift_mlp, ascale_mlp, agate_mlp

    return vx, ax

def _process_transformer_blocks(
    self,
    video: TransformerArgs | None,
    audio: TransformerArgs | None,
    perturbations: BatchedPerturbationConfig,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Process transformer blocks for LTXAV."""

    vx = video.x if video is not None else None
    ax = audio.x if audio is not None else None

    # Process transformer blocks
    for block in self.transformer_blocks:
        vx, ax = block(
            vx=vx,
            ax=ax,
            video_enabled=video.enabled if video is not None else False,
            audio_enabled=audio.enabled if audio is not None else False,
            video_timesteps=video.timesteps if video is not None else None,
            audio_timesteps=audio.timesteps if audio is not None else None,
            video_positional_embeddings=video.positional_embeddings if video is not None else None,
            audio_positional_embeddings=audio.positional_embeddings if audio is not None else None,
            video_context=video.context if video is not None else None,
            video_context_mask=video.context_mask if video is not None else None,
            audio_context=audio.context if audio is not None else None,
            audio_context_mask=audio.context_mask if audio is not None else None,
            video_cross_scale_shift_timestep=video.cross_scale_shift_timestep if video is not None else None,
            video_cross_gate_timestep=video.cross_gate_timestep if video is not None else None,
            audio_cross_scale_shift_timestep=audio.cross_scale_shift_timestep if audio is not None else None,
            audio_cross_gate_timestep=audio.cross_gate_timestep if audio is not None else None,
            video_cross_positional_embeddings=video.cross_positional_embeddings if video is not None else None,
            audio_cross_positional_embeddings=audio.cross_positional_embeddings if audio is not None else None,
            perturbations=perturbations,
        )

    return vx, ax

def model_forward(
    self, video: Modality | None, audio: Modality | None, perturbations: BatchedPerturbationConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass for LTX models.
    Returns:
        Processed output tensors
    """
    if not self.model_type.is_video_enabled() and video is not None:
        raise ValueError("Video is not enabled for this model")
    if not self.model_type.is_audio_enabled() and audio is not None:
        raise ValueError("Audio is not enabled for this model")

    video_args = self.video_args_preprocessor.prepare(video) if video is not None else None
    audio_args = self.audio_args_preprocessor.prepare(audio) if audio is not None else None
    # Process transformer blocks
    video_out, audio_out = self._process_transformer_blocks(
        video=video_args,
        audio=audio_args,
        perturbations=perturbations,
    )

    # Process output
    vx = (
        self._process_output(
            self.scale_shift_table, self.norm_out, self.proj_out, video_out, video_args.embedded_timestep
        )
        if video_out is not None
        else None
    )
    ax = (
        self._process_output(
            self.audio_scale_shift_table,
            self.audio_norm_out,
            self.audio_proj_out,
            audio_out,
            audio_args.embedded_timestep,
        )
        if audio_out is not None
        else None
    )
    return vx, ax
