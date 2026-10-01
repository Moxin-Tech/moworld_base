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
import json
import os
import random

import torch
from torch.utils.data import Dataset


class LatentDataset(Dataset):
    def __init__(
            self, json_path, num_latent_t, cfg_rate,
    ):
        self.json_path = json_path
        self.cfg_rate = cfg_rate
        self.datase_dir_path = os.path.dirname(json_path)
        self.prompt_embed_dir = os.path.join(self.datase_dir_path, "prompt_embed")
        self.pooled_prompt_embeds_dir = os.path.join(
            self.datase_dir_path, "pooled_prompt_embeds"
        )
        self.text_ids_dir = os.path.join(
            self.datase_dir_path, "text_ids"
        )
        with open(self.json_path, "r") as f:
            self.data_anno = json.load(f)
        self.num_latent_t = num_latent_t
        self.uncond_prompt_embed = torch.zeros(256, 4096).to(torch.float32)
        self.uncond_prompt_mask = torch.zeros(256).bool()
        self.lengths = [
            data_item["length"] if "length" in data_item else 1
            for data_item in self.data_anno
        ]

    def __getitem__(self, idx):
        prompt_embed_file = self.data_anno[idx]["prompt_embed_path"]
        pooled_prompt_embeds_file = self.data_anno[idx]["pooled_prompt_embeds_path"]
        text_ids_file = self.data_anno[idx]["text_ids"]
        if random.random() < self.cfg_rate:
            prompt_embed = self.uncond_prompt_embed
        else:
            prompt_embed = torch.load(
                os.path.join(self.prompt_embed_dir, prompt_embed_file),
                map_location="cpu",
                weights_only=True,
            )
            pooled_prompt_embeds = torch.load(
                os.path.join(
                    self.pooled_prompt_embeds_dir, pooled_prompt_embeds_file
                ),
                map_location="cpu",
                weights_only=True,
            )
            text_ids = torch.load(
                os.path.join(
                    self.text_ids_dir, text_ids_file
                ),
                map_location="cpu",
                weights_only=True,
            )
        return prompt_embed, pooled_prompt_embeds, text_ids, self.data_anno[idx]['caption']

    def __len__(self):
        return len(self.data_anno)


def latent_collate_function(batch):
    prompt_embeds, pooled_prompt_embeds, text_ids, caption = zip(*batch)
    prompt_embeds = torch.stack(prompt_embeds, dim=0)
    pooled_prompt_embeds = torch.stack(pooled_prompt_embeds, dim=0)
    text_ids = torch.stack(text_ids, dim=0)
    return prompt_embeds, pooled_prompt_embeds, text_ids, caption
