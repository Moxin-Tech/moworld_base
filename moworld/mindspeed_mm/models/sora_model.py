# coding=utf-8
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from logging import getLogger
from typing import Any, Mapping, Optional, Tuple

import torch
import torch_npu
from megatron.core import mpu
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.transformer.utils import (
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from torch import nn

from mindspeed_mm.data.data_utils.constants import (
    LATENTS,
    PROMPT,
    PROMPT_MASK,
    VIDEO_MASK,
)
from mindspeed_mm.models.ae import AEModel
from mindspeed_mm.models.common.communications import collect_tensors_across_ranks
from mindspeed_mm.models.diffusion import DiffusionModel
from mindspeed_mm.models.predictor import PredictModel
from mindspeed_mm.models.text_encoder import TextEncoder
from mindspeed_mm.utils.utils import unwrap_single
from mindspeed_mm.models.transformers.base_model import FSDP2Mixin, WeightInitMixin

logger = getLogger(__name__)


class SoRAModel(nn.Module, FSDP2Mixin, WeightInitMixin):
    """
    Instantiate a video generation model from config.
    SoRAModel is an assembled model, which may include text_encoder, video_encoder, predictor, and diffusion model.
    The model supports distributed training with pipeline parallelism and tensor parallelism,
    also supports encoder offload and feature cache mechanism for training efficiency optimization.

    Attributes:
        config: Core configuration for Multi-Modal Model with sub-configs for inner modules.
        task: Training task type of the model, default is `t2v`, optional: `t2v`, `i2v`.
        is_enable_lora: Flag to enable LoRA fine-tuning for target modules.
        interleaved_steps: Encoder offload interval step.
        enable_encoder_dp: Flag to enable data parallelism for encoder module.
        cache: Feature cache for encoder offload and DP scenario, store latent/prompt/mask features.
        index: Step index counter for feature cache access in interleaved training.
        i2v_keys: Cache keys for image-to-video task extra features.
        pre_process: Flag to mark current rank as pipeline first stage for feature encoding.
        post_process: Flag to mark current rank as pipeline last stage for loss computation.
        input_tensor: Input tensor passed from previous pipeline stage.
        load_video_features: Flag to load pre-encoded video latent features instead of raw video tensor.
        load_text_features: Flag to load pre-encoded text hidden states instead of token ids.
        ae: AutoEncoder model for video encode/decode, eval mode with no grad.
        text_encoder: Text encoder model for text feature extraction, eval mode with no grad.
        text_encoder_num: Number of text encoder branches in the multi-text encoder setup.
        offload_cpu: Flag to enable CPU offload for text encoder to save GPU memory.
        diffusion: Diffusion core model for video latent space generation.
        predictor: Predictor model for noise prediction in diffusion process.
        share_embeddings_and_output_weights: Flag to disable embedding weight sharing for megatron compatibility.
    """
    def __init__(self, config):
        """Initialize the assembled multi-modal video generation SoRAModel from config.

        Args:
            config (dict): the general config for Multi-Modal Model derived from model.json
            {
                "ae": {...},
                "text_encoder": {...},
                "predictor": {...},
                "diffusion": {...},
                "load_video_features":False,
                ...
            }
        """
        super().__init__()
        args = get_args()
        self.config = core_transformer_config_from_args(args)

        # training task
        self.task = getattr(config, "task", "t2v")
        self.is_enable_lora = bool(getattr(args, "lora_target_modules", None))

        # encoder inference interleaves with DIT training, encoder_offload_interval = 1 means don't offload
        self.interleaved_steps = getattr(config, "encoder_offload_interval", 1)
        self.enable_encoder_dp = getattr(config, "enable_encoder_dp", False)
        self.cache = {}
        self.index = 0
        self.i2v_keys = None
        if self.enable_encoder_dp and mpu.get_pipeline_model_parallel_world_size() > 1:
            raise AssertionError("Encoder DP cannot be used with PP")

        # build inner module
        self.pre_process = mpu.is_pipeline_first_stage()
        self.post_process = mpu.is_pipeline_last_stage()
        self.input_tensor = None

        if mpu.get_pipeline_model_parallel_rank() == 0:
            self.load_video_features = config.load_video_features
            self.load_text_features = config.load_text_features
            if not self.load_video_features:
                print_rank_0(f"init AEModel....")
                self.ae = AEModel(config.ae).eval()
                self.ae.requires_grad_(False)
            if not self.load_text_features:
                print_rank_0(f"init TextEncoder....")
                self.text_encoder = TextEncoder(config.text_encoder).eval()
                self.text_encoder.requires_grad_(False)
                self.text_encoder_num = len(self.text_encoder.text_encoders) \
                    if isinstance(self.text_encoder.text_encoders, nn.ModuleList) else 1
                self.offload_cpu = self.interleaved_steps > 1

        self.diffusion = DiffusionModel(config.diffusion).get_model()
        self.predictor = PredictModel(config.predictor).get_model()

        # to avoid grad all-reduce and reduce-scatter in megatron, since SoRAModel has no embedding layer.
        self.share_embeddings_and_output_weights = False

    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor
        self.predictor.set_input_tensor(input_tensor)

    def forward(self, video, prompt_ids, video_mask=None, prompt_mask=None, skip_encode=False, **kwargs):
        """
        video: raw video tensors, or ae encoded latent
        prompt_ids: tokenized input_ids, or encoded hidden states
        video_mask: mask for video/image
        prompt_mask: mask for prompt(text)
        skip_encode: get feature from the cache in some steps
        control_camera_latents_input: precomputed camera control latents (for wan2.2-control-camera)
        """
        args = get_args()
        if self.pre_process:
            with torch.no_grad():
                if not skip_encode:
                    self.index = 0
                    i2v_results = None

                    # Visual Encode
                    if self.load_video_features:
                        latents = video
                    else:
                        if self.task == "t2v":
                            latents, _ = self.ae.encode(video)
                        elif self.task in ("i2v", "control-camera"):
                            latents, i2v_results = self.ae.encode(video, **kwargs)
                        else:
                            raise NotImplementedError(f"Task {self.task} is not Implemented!")

                    # Text Encode
                    if self.load_text_features:
                        prompt = prompt_ids
                        if isinstance(prompt_ids, list) or isinstance(prompt_ids, tuple):
                            prompt = [p.npu() for p in prompt]
                    else:
                        prompt, prompt_mask = self.text_encoder.encode(prompt_ids, prompt_mask,
                                                                       offload_cpu=self.offload_cpu, **kwargs)

            # In feature mode, i2v_vae_feature / control_camera_latents_input arrive via kwargs.
            # Extract them so they can flow through the cache when interleaved_steps > 1.
            if i2v_results is None and self.task in ("i2v", "control-camera"):
                i2v_results = {}
                for feat_key in ("i2v_vae_feature", "control_camera_latents_input"):
                    if feat_key in kwargs and kwargs[feat_key] is not None:
                        i2v_results[feat_key] = kwargs.pop(feat_key)

            # Compute camera control latents before caching so they are included in i2v_results.
            # This must happen before init_cache because cam_from_worlds/intrinsics cannot be
            # cached directly (they are not 5D tensors and may include non-tensor values).
            if self.task == "control-camera" and hasattr(self.predictor, 'control_adapter') and self.predictor.control_adapter is not None:
                if i2v_results is not None and 'control_camera_latents_input' in i2v_results:
                    pass  # already extracted from kwargs above
                elif 'control_camera_latents_input' in kwargs and kwargs['control_camera_latents_input'] is not None:
                    # Pop from kwargs and add to i2v_results for caching
                    if i2v_results is None:
                        i2v_results = {}
                    i2v_results['control_camera_latents_input'] = kwargs.pop('control_camera_latents_input')
                else:
                    # End-to-end mode: compute camera latents from camera parameters
                    camera_keys = ['camera_control_direction', 'cam_from_worlds', 'intrinsics', 'ori_h', 'ori_w', 'start_frame', 'end_frame']
                    camera_kwargs = {k: kwargs.pop(k, None) for k in camera_keys}
                    # In interleaved mode these are lists; take first item to derive shape info.
                    ref_latents = latents[0] if isinstance(latents, list) else latents
                    num_latent_frames = ref_latents.shape[2]
                    num_frames = (num_latent_frames - 1) * 4 + 1  # inverse of num_latent_frames = (num_frames-1)//4+1
                    height = ref_latents.shape[3] * 8  # VAE spatial scale factor
                    width = ref_latents.shape[4] * 8
                    _to_int = lambda v: int(v.flatten()[0].item()) if hasattr(v, 'numel') and v.numel() > 0 else int(v.item()) if hasattr(v, 'item') else int(v) if v is not None else None
                    # Determine whether we have per-step lists (interleaved) or single tensors
                    is_interleaved = isinstance(latents, list)
                    if is_interleaved:
                        computed = []
                        for step_idx in range(len(latents)):
                            step_cam = self._compute_single_camera_latents(
                                camera_kwargs, step_idx, num_frames, height, width,
                                _to_int, ref_latents.device, ref_latents.dtype,
                            )
                            if step_cam is not None:
                                computed.append(step_cam)
                        if computed:
                            if i2v_results is None:
                                i2v_results = {}
                            i2v_results['control_camera_latents_input'] = computed
                    else:
                        step_cam = self._compute_single_camera_latents(
                            camera_kwargs, None, num_frames, height, width,
                            _to_int, latents.device, latents.dtype,
                        )
                        if step_cam is not None:
                            if i2v_results is None:
                                i2v_results = {}
                            i2v_results['control_camera_latents_input'] = step_cam

            # Gather the results after encoding of ae and text_encoder
            if self.enable_encoder_dp or self.interleaved_steps > 1:
                if self.index == 0:
                    self.init_cache(latents, prompt, video_mask, prompt_mask, i2v_results)
                latents, prompt, video_mask, prompt_mask, i2v_results = self.get_feature_from_cache()
            else:
                kwargs.update({"shape": latents.shape})

            if self.task in ("i2v", "control-camera") and i2v_results:
                for k, v in i2v_results.items():
                    if isinstance(v, torch.Tensor):
                        v = v.to(device=latents.device, dtype=latents.dtype)
                    kwargs[k] = v

            noised_latents, noise, timesteps = self.diffusion.q_sample(latents, model_kwargs=kwargs, mask=video_mask)
            predictor_input_latent, predictor_timesteps, predictor_prompt = noised_latents, timesteps, prompt
            predictor_video_mask, predictor_prompt_mask = video_mask, prompt_mask

        else:
            if not hasattr(self.predictor, "pipeline_set_prev_stage_tensor"):
                raise ValueError(f"PP has not been implemented for {self.predictor_cls} yet. ")
            predictor_input_list, training_loss_input_list = self.predictor.pipeline_set_prev_stage_tensor(
                self.input_tensor, extra_kwargs=kwargs)
            predictor_input_latent, predictor_timesteps, predictor_prompt, predictor_video_mask, predictor_prompt_mask \
                = predictor_input_list
            latents, noised_latents, timesteps, noise, video_mask = training_loss_input_list

        output = self.predictor(
            predictor_input_latent,
            timestep=predictor_timesteps,
            prompt=predictor_prompt,
            video_mask=predictor_video_mask,
            prompt_mask=predictor_prompt_mask,
            **kwargs,
        )

        if self.post_process:
            loss = self.compute_loss(
                output if isinstance(output, torch.Tensor) else output[0],
                latents,
                noised_latents,
                timesteps,
                noise,
                video_mask,
                **kwargs
            )
            return [loss]

        return self.predictor.pipeline_set_next_stage_tensor(
            input_list=[latents, noised_latents, timesteps, noise, video_mask],
            output_list=output,
            extra_kwargs=kwargs)

    def _compute_single_camera_latents(
            self, camera_kwargs, step_idx, num_frames, height, width, _to_int, device, dtype,
    ):
        """Compute camera control latents for a single step (or the only step in non-interleaved mode)."""
        from mindspeed_mm.models.predictor.dits.wan_camera_controller import compute_camera_latents

        def _get_step_val(v, idx):
            if v is None:
                return None
            if isinstance(v, list):
                if idx is None:
                    return v[0] if len(v) == 1 else v
                return v[idx] if idx < len(v) else v[-1]
            return v

        camera_dir = _get_step_val(camera_kwargs.get('camera_control_direction'), step_idx)
        cam_from_worlds = _get_step_val(camera_kwargs.get('cam_from_worlds'), step_idx)
        intrinsics_val = _get_step_val(camera_kwargs.get('intrinsics'), step_idx)
        ori_h = _to_int(_get_step_val(camera_kwargs.get('ori_h'), step_idx))
        ori_w = _to_int(_get_step_val(camera_kwargs.get('ori_w'), step_idx))
        start_frame = _to_int(_get_step_val(camera_kwargs.get('start_frame'), step_idx)) or 0
        end_frame = _to_int(_get_step_val(camera_kwargs.get('end_frame'), step_idx)) or num_frames

        has_direction = camera_dir is not None and not (
            isinstance(camera_dir, list) and all(v is None for v in camera_dir)
        )
        has_extrinsics = cam_from_worlds is not None

        if not has_direction and not has_extrinsics:
            return None

        if has_extrinsics and isinstance(cam_from_worlds, torch.Tensor) and cam_from_worlds.ndim >= 3:
            cam_from_worlds = cam_from_worlds[0]
        if intrinsics_val is not None and isinstance(intrinsics_val, torch.Tensor) and intrinsics_val.ndim >= 3:
            intrinsics_val = intrinsics_val[0]

        return compute_camera_latents(
            self.predictor.control_adapter,
            camera_control_direction=camera_dir,
            num_frames=num_frames, height=height, width=width,
            ori_h=ori_h, ori_w=ori_w,
            start_frame=start_frame, end_frame=end_frame,
            cam_from_worlds=cam_from_worlds,
            intrinsics=intrinsics_val,
            device=device, dtype=dtype,
        )

    def compute_loss(
            self, model_output, latents, noised_latents, timesteps, noise, video_mask, **kwargs
    ):
        """compute diffusion loss"""
        loss_dict = self.diffusion.training_losses(
            model_output=model_output,
            x_start=latents,
            x_t=noised_latents,
            noise=noise,
            t=timesteps,
            mask=video_mask,
            **kwargs
        )
        return loss_dict

    def train(self, mode=True):
        self.predictor.train()

    def state_dict_for_save_lora_checkpoint(self, state_dict):
        state_dict_ = dict()
        for key in state_dict:
            if 'lora' in key:
                state_dict_[key] = state_dict[key]
        return state_dict_

    def state_dict_for_save_checkpoint(self, prefix="", keep_vars=False):
        """Customized state_dict"""
        state_dict = self.predictor.state_dict(prefix=prefix, keep_vars=keep_vars)
        if self.is_enable_lora:
            state_dict = self.state_dict_for_save_lora_checkpoint(state_dict)
        return state_dict

    def state_dict(self, destination: Optional[dict] = None, **kwargs):
        """Customized state_dict for fsdp2"""
        if destination is None:
            destination = {}
        destination.update(self.predictor.state_dict())
        return destination

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        """Customized load."""
        if not isinstance(state_dict, Mapping):
            raise TypeError(f"Expected state_dict to be dict-like, got {type(state_dict)}.")

        missing_keys, unexpected_keys = self.predictor.load_state_dict(state_dict, False)

        if missing_keys is not None:
            logger.info(f"Missing keys in state_dict: {missing_keys}.")
        if unexpected_keys is not None:
            logger.info(f"Unexpected key(s) in state_dict: {unexpected_keys}.")
        return None

    def init_cache(self, latents, prompt, video_mask, prompt_mask, i2v_results=None):
        """
        Initialize feature cache in the first step of one round when encoder dp or encoder interleave offload is enabled.
        example with latents and prompt:
        input:
            latents [step_1, step_2..., step_n]
            prompt  [step_n][encoder_1, encoder_2, ...encoder_n]
        cache:
            latents [step_n][rank1_data, rank2_data, ...]"
            prompt  [step_n][encoder_n][rank1_data, rank2_data, ...]
        """
        # empty cache
        self.cache = {}
        group = mpu.get_tensor_and_context_parallel_group()
        for key, value in [(LATENTS, latents), (VIDEO_MASK, video_mask)]:
            if value is None or len(value) < 0:
                continue
            self.cache[key] = [[item] for item in value] if not self.enable_encoder_dp \
                else collect_tensors_across_ranks(value, group=group, dynamic_shape=False)

        for key, value in [(PROMPT, prompt), (PROMPT_MASK, prompt_mask)]:
            if value is None or len(value) < 0:
                continue
            if not self.enable_encoder_dp:
                self.cache[key] = [[item] for item in value]
                continue
            self.cache[key] = [[[] for _ in range(self.text_encoder_num)] for _ in range(self.interleaved_steps)]
            for encoder_idx in range(self.text_encoder_num):
                # Features from the same text encoder have identical shapes, concat to reduce communication overhead.
                encoder_step_tensors = torch.stack([value[step][encoder_idx] for step in range(self.interleaved_steps)])
                collected_tensors = collect_tensors_across_ranks(encoder_step_tensors, group=group, dynamic_shape=False)

                for step_idx in range(self.interleaved_steps):
                    for collected_tensor in collected_tensors:
                        self.cache[key][step_idx][encoder_idx].append(
                            collected_tensor[step_idx:step_idx + 1].squeeze(0).contiguous())

        # handle extra i2v cache, source i2v_results：{"key_n":[step_1_data, step_2_date ... step_n_data]}
        if self.task not in ("i2v", "control-camera") or not i2v_results:
            return

        self.i2v_keys = i2v_results.keys() if self.i2v_keys is None else self.i2v_keys
        for i2v_key, value in i2v_results.items():
            if not self.enable_encoder_dp:
                self.cache[i2v_key] = [[item] for item in value]
                continue
            self.cache[i2v_key] = collect_tensors_across_ranks(value, group=group, dynamic_shape=False)

    def get_feature_from_cache(self):
        """Get from the cache while several features have been already encoded and cached."""
        divisor = mpu.get_tensor_and_context_parallel_world_size() if self.enable_encoder_dp else 1
        step_idx = self.index // divisor
        rank_idx = self.index % divisor

        latents = unwrap_single(self.cache[LATENTS][step_idx][rank_idx] if LATENTS in self.cache else None)
        video_mask = unwrap_single(self.cache[VIDEO_MASK][step_idx][rank_idx] if VIDEO_MASK in self.cache else None)
        prompt = unwrap_single([self.cache[PROMPT][step_idx][encoder_idx][rank_idx] \
                                for encoder_idx in range(self.text_encoder_num)] if PROMPT in self.cache else None)
        prompt_mask = unwrap_single([self.cache[PROMPT_MASK][step_idx][encoder_idx][rank_idx] \
                                     for encoder_idx in range(self.text_encoder_num)] if PROMPT_MASK in self.cache else None)

        i2v_results = {}
        if self.task in ("i2v", "control-camera"):
            for key in self.i2v_keys:
                i2v_results[key] = unwrap_single(self.cache[key][step_idx][rank_idx] if key in self.cache else None)

        self.index += 1
        return latents, prompt, video_mask, prompt_mask, i2v_results

    def sharded_state_dict(
        self,
        prefix: str = '',
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Default implementation for sharded state dict for distributed checkpointing.

        General definition of sharded_state_dict simply calls `sharded_state_dict_default`
        (which call sharded_state_dict method if possible or a default implementation otherwise)
        recursively on all submodules.

        Args:
            prefix (str): prefix for the state dict keys
            sharded_offsets (Tuple[Tuple[int, int, int]], optional): sharding already
                applied (e.g. PP related) by sup-modules. Passed along to ShardedTensor
            metadata (dict, optional): metadata passed recursively to sharded_state_dict methods

        Returns:
            dict: dictionary of state dict keys mapped to ShardedTensors
        """
        sharded_state_dict = {}
        # Save parameters

        self._save_to_state_dict(sharded_state_dict, '', keep_vars=True)

        sharded_state_dict = make_sharded_tensors_for_checkpoint(
            sharded_state_dict, prefix, sharded_offsets=sharded_offsets, extra_state_suffix=""
        )
        # Recurse into submodules
        for name, module in self.named_children():
            sharded_state_dict.update(
                sharded_state_dict_default(module, f'{prefix}{name}.', sharded_offsets, metadata)
            )

        return sharded_state_dict
