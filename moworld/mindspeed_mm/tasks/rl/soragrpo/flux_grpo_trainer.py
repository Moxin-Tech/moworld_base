# Copyright (c) [2025] [FastVideo Team]
# Copyright (c) [2025] [ByteDance Ltd. and/or its affiliates.]
# SPDX-License-Identifier: [Apache License 2.0]
#
# This file has been modified by [ByteDance Ltd. and/or its affiliates.] in 2025.
#
# Original file was released under [Apache License 2.0], with the full license text
# available at [https://github.com/hao-ai-lab/FastVideo/blob/main/LICENSE].
#
# This modified file is released under the same license.

import os

import torch
import torch.distributed as dist
from diffusers.image_processor import VaeImageProcessor
from tqdm.auto import tqdm

from mindspeed_mm.tasks.rl.soragrpo.sora_grpo_trainer import SoraGRPOTrainer
from mindspeed_mm.tasks.rl.soragrpo.flux_grpo_model import FluxGRPOModel


class FluxGRPOTrainer(SoraGRPOTrainer):
    def model_provider(self, args):
        return FluxGRPOModel(args, device=self.device)

    def grpo_one_step(self, sample, perm, sigma_schedule, index):
        args = self.args
        latents = sample["latents"][:, index]
        pre_latents = sample["next_latents"][:, index]
        encoder_hidden_states = sample["encoder_hidden_states"]
        pooled_prompt_embeds = sample["pooled_prompt_embeds"]
        text_ids = sample["text_ids"]
        image_ids = sample["image_ids"]
        transformer = self.hyper_model.diffuser
        timesteps = sample["timesteps"][:, index]
        transformer.train()
        with torch.autocast("cuda", torch.bfloat16):
            pred = transformer(
                hidden_states=latents,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timesteps / 1000,
                guidance=torch.tensor(
                    [3.5],
                    device=latents.device,
                    dtype=torch.bfloat16
                ),
                txt_ids=text_ids.repeat(encoder_hidden_states.shape[1], 1),  # B, L
                pooled_projections=pooled_prompt_embeds,
                img_ids=image_ids.squeeze(0),
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
        config = {}
        config["grpo"] = True
        config["sde_solver"] = True
        config["eta"] = args.eta
        config["index"] = perm
        z, pred_original, log_prob = self.grpo_step(pred, latents.to(torch.float32), sigma_schedule,
                                                    pre_latents.to(torch.float32), config)
        return log_prob

    def sample_reference(self, dataloader):
        (
            encoder_hidden_states,
            pooled_prompt_embeds,
            text_ids,
            caption,
        ) = next(dataloader)
        args = self.args
        if args.use_group:
            def repeat_tensor(tensor):
                if tensor is None:
                    return None
                return torch.repeat_interleave(tensor, args.num_generations, dim=0)

            encoder_hidden_states = repeat_tensor(encoder_hidden_states)
            pooled_prompt_embeds = repeat_tensor(pooled_prompt_embeds)
            text_ids = repeat_tensor(text_ids)
            if isinstance(caption, str):
                caption = [caption] * args.num_generations
            elif isinstance(caption, list):
                caption = [
                    item
                    for item in caption
                    for _ in range(args.num_generations)
                ]
            else:
                raise ValueError(f"Unsupported caption type: {type(caption)}")
        reward, all_latents, all_log_probs, sigma_schedule, all_image_ids = self.sample_reference_model(
            args,
            caption,
            encoder_hidden_states,
            pooled_prompt_embeds,
            text_ids
        )
        batch_size = all_latents.shape[0]
        timestep_value = [int(sigma * 1000) for sigma in sigma_schedule][:args.sampling_steps]
        timestep_values = [timestep_value[:] for _ in range(batch_size)]
        device = all_latents.device
        timesteps = torch.tensor(timestep_values, device=all_latents.device, dtype=torch.long)
        samples = {
            "timesteps": timesteps.detach().clone()[:, :-1],
            "latents": all_latents[
                       :, :-1
                       ][:, :-1],  # each entry is the latent before timestep t
            "next_latents": all_latents[
                            :, 1:
                            ][:, :-1],  # each entry is the latent after timestep t
            "log_probs": all_log_probs[:, :-1],
            "rewards": reward.to(torch.float32),
            "image_ids": all_image_ids,
            "text_ids": text_ids,
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }

        gathered_reward = self.gather_tensor(samples["rewards"])
        if dist.get_rank() == 0:
            print("gathered_hps_reward", gathered_reward)
            print("gathered_hps_reward_mean=", gathered_reward.mean().item())
            with open(args.hps_reward_save, 'a') as f:
                f.write(f"{gathered_reward.mean().item()}\n")
        # Calculate advantage
        if args.use_group:
            n = len(samples["rewards"]) // (args.num_generations)
            advantages = torch.zeros_like(samples["rewards"])

            for i in range(n):
                start_idx = i * args.num_generations
                end_idx = (i + 1) * args.num_generations
                group_rewards = samples["rewards"][start_idx:end_idx]
                group_mean = group_rewards.mean()
                group_std = group_rewards.std() + 1e-8
                advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std

            samples["advantages"] = advantages
        else:
            advantages = (samples["rewards"] - gathered_reward.mean()) / (gathered_reward.std() + 1e-8)
            samples["advantages"] = advantages

        perms = torch.stack(
            [
                torch.randperm(len(samples["timesteps"][0]))
                for _ in range(batch_size)
            ]
        ).to(device)
        for key in ["timesteps", "latents", "next_latents", "log_probs"]:
            samples[key] = samples[key][
                torch.arange(batch_size).to(device)[:, None],
                perms,
            ]
        samples_batched = {
            k: v.unsqueeze(1)
            for k, v in samples.items()
        }
        # dict of lists -> list of dicts for easier iteration
        samples_batched_list = [dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())]
        train_timesteps = int(len(samples["timesteps"][0]) * args.timestep_fraction)
        return samples_batched_list, train_timesteps, sigma_schedule, perms

    def sample_reference_model(self, args, caption, encoder_hidden_states, pooled_prompt_embeds, text_ids):
        transformer = self.hyper_model.diffuser
        vae = self.hyper_model.ae
        reward_model = self.hyper_model.reward['model']
        tokenizer = self.hyper_model.reward['processor']
        preprocess_val = self.hyper_model.reward['preprocess_val']
        device = self.device
        w, h, t = args.w, args.h, args.t
        sample_steps = args.sampling_steps
        sigma_schedule = torch.linspace(1, 0, args.sampling_steps + 1)

        sigma_schedule = self.sd3_time_shift(args.shift, sigma_schedule)

        FluxGRPOTrainer.assert_eq(
            len(sigma_schedule),
            sample_steps + 1,
            "sigma_schedule must have length sample_steps + 1",
        )

        B = encoder_hidden_states.shape[0]
        batch_size = args.sample_batch_size
        batch_indices = torch.chunk(torch.arange(B), B // batch_size)

        SPATIAL_DOWNSAMPLE = 8
        IN_CHANNELS = 16
        latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE
        if args.init_same_noise:
            input_latents = torch.randn(
                (1, IN_CHANNELS, latent_h, latent_w),  # （c,t,h,w)
                dtype=torch.bfloat16,
            ).repeat(batch_size, 1, 1, 1).to(device)

        all_latents = []
        all_log_probs = []
        all_rewards = []
        all_image_ids = []

        for batch_idx in batch_indices:
            if not args.init_same_noise:
                input_latents = torch.randn(
                    (len(batch_idx), IN_CHANNELS, latent_h, latent_w),  # （c,t,h,w)
                    device=device,
                    dtype=torch.bfloat16,
                )
            progress_bar = tqdm(range(0, sample_steps), desc="Sampling Progress")
            image_ids = self.prepare_latent_image_ids(len(batch_idx), latent_h // 2, latent_w // 2, device,
                                                      torch.bfloat16)
            with torch.no_grad():
                pack_input_latents = self.pack_latents(input_latents, len(batch_idx), IN_CHANNELS, latent_h, latent_w)

                sample_input = {
                    "pack_input_latents": pack_input_latents,
                    "sigma_schedule": sigma_schedule,
                    "encoder_hidden_states": encoder_hidden_states[batch_idx],
                    "pooled_prompt_embeds": pooled_prompt_embeds[batch_idx],
                    "text_ids": text_ids[batch_idx],
                    "image_ids": image_ids
                }
                z, latents, batch_latents, batch_log_probs = self.run_sample_step(args, progress_bar, transformer,
                                                                                  sample_input)

            for _ in range(batch_size):
                all_image_ids.append(image_ids)
            all_latents.append(batch_latents)
            all_log_probs.append(batch_log_probs)

            vae.enable_tiling()
            rank = int(os.environ["RANK"])

            with torch.inference_mode():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    latents = self.unpack_latents(latents, h, w, 8)
                    latents = (latents / 0.3611) + 0.1159
                    image = vae.decode(latents, return_dict=False)[0]
                    image_processor = VaeImageProcessor(16)
                    batch_decoded_images = image_processor.postprocess(image)

            for idx, image in zip(batch_idx, batch_decoded_images):
                image.save(f"./images/flux_{rank}_{idx}.png")

            batch_caption = [caption[i] for i in batch_idx]
            if args.use_hpsv2:
                with torch.inference_mode():
                    for i, decoded_image in enumerate(batch_decoded_images):
                        image_path = decoded_image
                        image = preprocess_val(image_path).unsqueeze(0).to(device=device, non_blocking=True)
                        # Process the prompt
                        text = tokenizer([batch_caption[i]]).to(device=device, non_blocking=True)
                        # Calculate the HPS
                        with torch.amp.autocast('cuda'):
                            outputs = reward_model(image, text)
                            image_features, text_features = outputs["image_features"], outputs["text_features"]
                            logits_per_image = image_features @ text_features.T
                            hps_score = torch.diagonal(logits_per_image)
                        all_rewards.append(hps_score)

        all_latents = torch.cat(all_latents, dim=0)
        all_log_probs = torch.cat(all_log_probs, dim=0)
        all_rewards = torch.cat(all_rewards, dim=0)
        all_image_ids = torch.stack(all_image_ids, dim=0)

        return all_rewards, all_latents, all_log_probs, sigma_schedule, all_image_ids

    def run_sample_step(self, args, progress_bar, transformer, sample_input):
        z = sample_input["pack_input_latents"]
        sigma_schedule = sample_input["sigma_schedule"]
        encoder_hidden_states = sample_input["encoder_hidden_states"]
        pooled_prompt_embeds = sample_input["pooled_prompt_embeds"]
        text_ids = sample_input["text_ids"]
        image_ids = sample_input["image_ids"]

        all_latents = [z]
        all_log_probs = []
        for i in progress_bar:  # Add progress bar
            sigma = sigma_schedule[i]
            timestep_value = int(sigma * 1000)
            timesteps = torch.full([encoder_hidden_states.shape[0]], timestep_value, device=z.device,
                                   dtype=torch.long)
            transformer.eval()
            with torch.autocast("cuda", torch.bfloat16):
                expanded_text_ids = text_ids.unsqueeze(1)
                expanded_text_ids = expanded_text_ids.repeat(1, encoder_hidden_states.shape[1], 1)
                pred = transformer(
                    hidden_states=z,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timesteps / 1000,
                    guidance=torch.tensor(
                        [3.5],
                        device=z.device,
                        dtype=torch.bfloat16
                    ),
                    txt_ids=expanded_text_ids[0],  # B, L
                    pooled_projections=pooled_prompt_embeds,
                    img_ids=image_ids,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
            config = {}
            config["grpo"] = True
            config["sde_solver"] = True
            config["eta"] = args.eta
            config["index"] = i
            z, pred_original, log_prob = self.grpo_step(pred, z.to(torch.float32), sigma_schedule, None, config)
            z.to(torch.bfloat16)
            all_latents.append(z)
            all_log_probs.append(log_prob)
        latents = pred_original
        all_latents = torch.stack(all_latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
        all_log_probs = torch.stack(all_log_probs, dim=1)  # (batch_size, num_steps, 1)
        return z, latents, all_latents, all_log_probs

    def pack_latents(self, latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    def unpack_latents(self, latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents

    def prepare_latent_image_ids(self, batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)
