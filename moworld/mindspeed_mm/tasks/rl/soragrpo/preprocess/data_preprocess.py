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

import re
import os
import argparse

from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DistributedSampler
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader


class DataPreprocess(ABC):
    def __init__(self):
        self.rank = int(os.environ["RANK"])
        dist.init_process_group("hccl")
        torch.cuda.set_device(self.rank)
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.device = torch.cuda.current_device()
        self.args = self.get_args()
        self.dataloader = self._init_dataloader()

    def _init_dataloader(self):
        args = self.args
        rank = self.rank
        world_size = self.world_size
        train_dataset = PromptDataset(args.prompt_dir, args)
        sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
        return DataLoader(
            train_dataset,
            sampler=sampler,
            batch_size=args.dataload_batch_size,
            num_workers=args.dataloader_num_workers,
        )

    @abstractmethod
    def preprocess(self):
        raise NotImplementedError("Subclasses must implement this method")

    def get_args(self):
        parser = argparse.ArgumentParser()
        # dataset & dataloader
        parser.add_argument("--load", type=str)
        parser.add_argument("--model_type", type=str)
        # text encoder & vae & diffusion model
        parser.add_argument(
            "--dataloader_num_workers",
            type=int,
            default=1,
            help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
        )
        parser.add_argument(
            "--dataload_batch_size",
            type=int,
            default=1,
            help="Batch size (per device) for the preprocess dataloader.",
        )
        parser.add_argument("--text_encoder_name", type=str)
        parser.add_argument("--cache_dir", type=str, default="./cache_dir")
        parser.add_argument(
            "--output_dir",
            type=str,
            default=None,
        )
        parser.add_argument("--vae_debug", action="store_true")
        parser.add_argument("--prompt_dir", type=str, default="./empty.txt")
        parser.add_argument("--sample_num", type=int, default=None)
        return parser.parse_args()


class PromptDataset(Dataset):
    def __init__(self, txt_path, args):
        self.txt_path = txt_path
        self.args = args
        with open(self.txt_path, "r", encoding="utf-8") as f:
            self.train_dataset = [line for line in f.read().splitlines() if not self.contains_chinese(line)]
        if args.sample_num is not None:
            self.train_dataset = self.train_dataset[:args.sample_num]

    def __getitem__(self, idx):
        return dict(caption=(self.train_dataset[idx]), latents=[], filename=str(idx))

    def __len__(self):
        return len(self.train_dataset)

    def contains_chinese(self, text):
        return bool(re.search(r'[\u4e00-\u9fff]', text))
