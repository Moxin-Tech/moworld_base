# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
from logging import getLogger
from typing import Any, Mapping

import copy
import torch
import torch_npu

from megatron.core import mpu
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from torch import nn

from mindspeed_mm.models.ae import AEModel
from mindspeed_mm.models.diffusion import DiffusionModel
from mindspeed_mm.models.predictor import PredictModel
from mindspeed_mm.models.text_encoder import TextEncoder

logger = getLogger(__name__)


class SoRADPOModel(nn.Module):
    """
    The hyper model wraps multiple models required in reinforcement learning into a single model,
    maintaining the original distributed perspective unchanged.
    """

    def __init__(self, config):
        super().__init__()
        args = get_args()
        self.config = core_transformer_config_from_args(args)
        self.task = getattr(config, "task", "t2v")
        self.pre_process = mpu.is_pipeline_first_stage()
        self.post_process = mpu.is_pipeline_last_stage()
        self.input_tensor = None
        # to avoid grad all-reduce and reduce-scatter in megatron, since SoRAModel has no embedding layer.
        self.share_embeddings_and_output_weights = False
        self._model_provider(config)

    def _model_provider(self, config):
        """Builds the model."""
        print_rank_0("building SoRADPOModel related modules ...")
        args = get_args()
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

        self.diffusion = DiffusionModel(config.diffusion).get_model()
        # copy config
        predictor_config_ref = copy.deepcopy(config.predictor)
        predictor_config = copy.deepcopy(config.predictor)
        self.reference = PredictModel(predictor_config_ref).get_model().eval()
        self.reference.requires_grad_(False)
        self.actor = PredictModel(predictor_config).get_model()

        print_rank_0("finish building SoRADPOModel related modules ...")
        return None

    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor
        self.actor.set_input_tensor(input_tensor)

    def forward(self, video, video_lose, prompt_ids, video_mask=None, prompt_mask=None, **kwargs):
        """
        video: high-scoring raw video tensors, or ae encoded latent
        video_lose: low-scoring raw video tensors, or ae encoded latent
        prompt_ids: tokenized input_ids, or encoded hidden states
        video_mask: mask for video/image
        prompt_mask: mask for prompt(text)
        """
        args = get_args()
        if self.pre_process:
            with torch.no_grad():
                i2v_results = None
                # Visual Encode
                if self.load_video_features:
                    latents = video
                    latents_lose = video_lose
                else:
                    if self.task == "t2v":
                        latents, _ = self.ae.encode(video)
                        latents_lose, _ = self.ae.encode(video_lose)
                    elif self.task == "i2v":
                        latents, i2v_results = self.ae.encode(video, **kwargs)
                        # The first frame of the two videos are the same, so i2v_results should be the same.
                        latents_lose, _ = self.ae.encode(video_lose, **kwargs)
                    else:
                        raise NotImplementedError(f"Task {self.task} is not Implemented!")

                 # Text Encode
                if self.load_text_features:
                    prompts = prompt_ids
                    prompt_mask = prompt_mask
                else:
                    prompts, prompt_mask = self.text_encoder.encode(prompt_ids, prompt_mask, **kwargs)

            noised_latents_win, noise, timesteps = self.diffusion.q_sample(latents, model_kwargs=kwargs, mask=video_mask)
            noised_latents_lose, _, _ = self.diffusion.q_sample(latents_lose, noise=noise, t=timesteps, model_kwargs=kwargs, mask=video_mask)
            noised_latents = torch.cat((noised_latents_win, noised_latents_lose), dim=0)
            noise = torch.cat((noise, noise), dim=0)
            timesteps = timesteps.repeat(2)
            latents = torch.cat((latents, latents_lose), dim=0)
            # text is the same
            if isinstance(prompts, list):
                prompt = [torch.cat((prompt, prompt), dim=0) for prompt in prompts]
                prompt_mask = [torch.cat((mask, mask), dim=0) for mask in prompt_mask]
            else:
                prompt = torch.cat((prompts, prompts), dim=0)
                prompt_mask = torch.cat((prompt_mask, prompt_mask), dim=0)

            if i2v_results is not None:
                for k, v in i2v_results.items():
                    kwargs[k] = torch.cat((v, v), dim=0)

            predictor_input_latent, predictor_timesteps, predictor_prompt = noised_latents, timesteps, prompt
            predictor_video_mask, predictor_prompt_mask = video_mask, prompt_mask
            predictor_input_latent_ref, predictor_timesteps_ref, predictor_prompt_ref = noised_latents, timesteps, prompt
            predictor_video_mask_ref, predictor_prompt_mask_ref = video_mask, prompt_mask
            kwargs_ref = kwargs
            score = kwargs["score"]
            score_lose = kwargs["score_lose"]
        else:
            if not hasattr(self.actor, "pipeline_set_prev_stage_tensor"):
                raise ValueError(f"PP has not been implemented for {self.actor_cls} yet. ")
            kwargs_ref = copy.deepcopy(kwargs)
            ori_keys = kwargs_ref.keys()
            predictor_input_list, training_loss_input_list, score_list = self.actor.pipeline_set_prev_stage_tensor(
                self.input_tensor, extra_kwargs=kwargs)
            new_keys = kwargs.keys()
            extra_keys = new_keys - ori_keys
            # get extra kwargs for forward func
            for key in extra_keys:
                if isinstance(kwargs[key], torch.Tensor):
                    extra_input_list = kwargs.pop(key)
                    kwargs[key], kwargs_ref[key] = torch.chunk(extra_input_list, 2, dim=0)
                else:
                    kwargs_ref[key] = kwargs[key]
            # prev stage output
            predictor_input_latent_list, predictor_timesteps_list, predictor_prompt_list, _, predictor_prompt_mask_list = predictor_input_list
            predictor_input_latent, predictor_input_latent_ref = torch.chunk(predictor_input_latent_list, 2, dim=0)
            predictor_timesteps, predictor_timesteps_ref = torch.chunk(predictor_timesteps_list, 2, dim=0)
            predictor_prompt, predictor_prompt_ref = torch.chunk(predictor_prompt_list, 2, dim=0)
            predictor_video_mask, predictor_video_mask_ref = None, None
            predictor_prompt_mask, predictor_prompt_mask_ref = torch.chunk(predictor_prompt_mask_list, 2, dim=0)
            # values to calculate loss.
            latents, noised_latents, timesteps, noise, video_mask = training_loss_input_list
            # score
            score, score_lose = score_list

        with torch.no_grad():
            refer_output = self.reference(
                predictor_input_latent_ref,
                timestep=predictor_timesteps_ref,
                prompt=predictor_prompt_ref,
                video_mask=predictor_video_mask_ref,
                prompt_mask=predictor_prompt_mask_ref,
                **kwargs_ref,
            )

        actor_output = self.actor(
            predictor_input_latent,
            timestep=predictor_timesteps,
            prompt=predictor_prompt,
            video_mask=predictor_video_mask,
            prompt_mask=predictor_prompt_mask,
            **kwargs,
        )

        if self.post_process:
            if isinstance(refer_output, tuple):
                refer_output = refer_output[0]
            if isinstance(actor_output, tuple):
                actor_output = actor_output[0]
            output = torch.cat((actor_output, refer_output), dim=0)
            return [output, score, score_lose, latents, noised_latents, timesteps, noise]

        output = []
        for index, _ in enumerate(actor_output):
            output.append(torch.cat((actor_output[index], refer_output[index]), dim=0))

        output = output + [score, score_lose]

        return self.actor.pipeline_set_next_stage_tensor(
            input_list=[latents, noised_latents, timesteps, noise, video_mask],
            output_list=output,
            extra_kwargs=kwargs)

    def state_dict_for_save_checkpoint(self, prefix="", keep_vars=False):
        """Customized state_dict"""
        state_dict = self.actor.state_dict(prefix=prefix, keep_vars=keep_vars)
        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        """Customized load."""
        if not isinstance(state_dict, Mapping):
            raise TypeError(f"Expected state_dict to be dict-like, got {type(state_dict)}.")

        missing_keys, unexpected_keys = self.actor.load_state_dict(state_dict, False)
        if missing_keys is not None:
            logger.info(f"Actor missing keys in state_dict: {missing_keys}.")
        if unexpected_keys is not None:
            logger.info(f"Actor unexpected key(s) in state_dict: {unexpected_keys}.")

        missing_keys_ref, unexpected_keys_ref = self.reference.load_state_dict(state_dict, False)
        if missing_keys_ref is not None:
            logger.info(f"Reference missing keys in state_dict: {missing_keys_ref}.")
        if unexpected_keys_ref is not None:
            logger.info(f"Reference unexpected key(s) in state_dict: {unexpected_keys_ref}.")
        return None
