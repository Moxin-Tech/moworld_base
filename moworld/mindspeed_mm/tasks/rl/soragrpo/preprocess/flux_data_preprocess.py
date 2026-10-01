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
import json
import torch

from diffusers import FluxPipeline
from tqdm import tqdm
import torch.distributed as dist

import mindspeed.megatron_adaptor
from mindspeed_mm.tasks.rl.soragrpo.preprocess.data_preprocess import DataPreprocess


class FluxDataPreprocess(DataPreprocess):
    def __init__(self):
        super().__init__()

    def preprocess(self):
        args = self.args
        local_rank = self.rank
        world_size = self.world_size
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "prompt_embed"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "text_ids"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "pooled_prompt_embeds"), exist_ok=True)
        pipe = FluxPipeline.from_pretrained(args.load, torch_dtype=torch.bfloat16).to(self.device)
        json_data = []
        for _, data in tqdm(enumerate(self.dataloader), disable=local_rank != 0):
            with torch.inference_mode():
                for idx, video_name in enumerate(data["filename"]):
                    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                        prompt=data["caption"], prompt_2=data["caption"]
                    )
                    prompt_embed_path = os.path.join(args.output_dir, "prompt_embed", video_name + ".pt")
                    pooled_prompt_embeds_path = os.path.join(args.output_dir, "pooled_prompt_embeds",
                                                             video_name + ".pt")
                    text_ids_path = os.path.join(args.output_dir, "text_ids", video_name + ".pt")
                    # save latent
                    torch.save(prompt_embeds[idx], prompt_embed_path)
                    torch.save(pooled_prompt_embeds[idx], pooled_prompt_embeds_path)
                    torch.save(text_ids[idx], text_ids_path)
                    item = {}
                    item["prompt_embed_path"] = video_name + ".pt"
                    item["text_ids"] = video_name + ".pt"
                    item["pooled_prompt_embeds_path"] = video_name + ".pt"
                    item["caption"] = data["caption"][idx]
                    json_data.append(item)
        dist.barrier()
        local_data = json_data
        gathered_data = [None] * world_size
        dist.all_gather_object(gathered_data, local_data)
        if local_rank == 0:
            all_json_data = [
                item
                for sublist in gathered_data
                for item in sublist
            ]
            with open(os.path.join(args.output_dir, "videos2caption.json"), "w") as f:
                json.dump(all_json_data, f, indent=4)


if __name__ == '__main__':
    data_preprocess = FluxDataPreprocess()
    data_preprocess.preprocess()
