# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import os
import random
import traceback
import pyarrow.parquet as pq
import pyarrow.fs as pf
import torch
import torch.distributed as dist
from PIL import Image, ImageFile, PngImagePlugin

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class DistributedIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset_name, local_rank=0, world_size=1, num_workers=8):
        self.dataset_name = dataset_name
        self.local_rank = local_rank
        self.world_size = world_size
        self.num_workers = num_workers
        self.rng = random.Random()
        self.data_paths = None

    def get_data_paths(self, *args, **kwargs):
        raise NotImplementedError

    def set_epoch(self, seed=42):
        if self.data_paths is None:
            return

        if isinstance(self.data_paths[0], tuple):
            data_paths = sorted(self.data_paths, key=lambda x: (x[0], x[1]))
        elif isinstance(self.data_paths[0], str):
            data_paths = sorted(self.data_paths)
        else:
            raise ValueError(f"Unknown data_paths type: {type(self.data_paths[0])}")

        self.rng.seed(seed)
        self.rng.shuffle(data_paths)

        num_files_per_rank = len(data_paths) // self.world_size
        local_start = self.local_rank * num_files_per_rank
        local_end = (self.local_rank + 1) * num_files_per_rank
        self.num_files_per_rank = num_files_per_rank
        self.data_paths_per_rank = data_paths[local_start:local_end]

    def get_data_paths_per_worker(self):
        if self.data_paths is None:
            return None

        info = torch.utils.data.get_worker_info()
        if info is None:
            # Single worker: Use all files assigned to the rank
            return self.data_paths_per_rank, 0

        worker_id = info.id
        num_files_per_worker = self.num_files_per_rank // info.num_workers
        start = num_files_per_worker * worker_id
        end = num_files_per_worker * (worker_id + 1)
        data_paths_per_worker = self.data_paths_per_rank[start:end]

        return data_paths_per_worker[::-1], worker_id

    def __iter__(self):
        raise NotImplementedError


class T2IIterableDataset(DistributedIterableDataset):
    def __init__(
            self, dataset_name, transform, tokenizer, data_dir_list, num_used_data,
            local_rank=0, world_size=1, num_workers=8, data_status=None,
    ):
        """
        data_dir_list: list of data directories contains parquet files
        num_used_data: list of number of sampled data paths for each data directory
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.data_paths = self.get_data_paths(data_dir_list, num_used_data)
        self.set_epoch()

    def get_data_paths(self, data_dir_list, num_sampled_data_paths, rank=0, world_size=1):
        num_data_dirs = len(data_dir_list)
        if world_size > 1:
            chunk_size = (num_data_dirs + world_size - 1) // world_size
            start_idx = rank * chunk_size
            end_idx = min(start_idx + chunk_size, num_data_dirs)
            local_data_dir_list = data_dir_list[start_idx:end_idx]
            local_num_sampled_data_paths = num_sampled_data_paths[start_idx:end_idx]
        else:
            local_data_dir_list = data_dir_list
            local_num_sampled_data_paths = num_sampled_data_paths

        local_data_paths = []
        for data_dir, num_data_path in zip(local_data_dir_list, local_num_sampled_data_paths):
            files = os.listdir(data_dir)
            data_paths_per_dir = [
                os.path.join(data_dir, name)
                for name in files
                if name.endswith(".parquet")
            ]
            repeat = num_data_path // len(data_paths_per_dir)
            data_paths_per_dir = data_paths_per_dir * (repeat + 1)
            local_data_paths.extend(data_paths_per_dir[:num_data_path])

        if world_size > 1:
            gather_list = [None] * world_size
            dist.all_gather_object(gather_list, local_data_paths)

            combined_chunks = []
            for chunk_list in gather_list:
                if chunk_list is not None:
                    combined_chunks.extend(chunk_list)
        else:
            combined_chunks = local_data_paths

        return combined_chunks

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            parquet_start_id = self.data_status[worker_id][0]
            row_group_start_id = self.data_status[worker_id][1]
            row_start_id = self.data_status[worker_id][2] + 1
        else:
            parquet_start_id = 0
            row_group_start_id = 0
            row_start_id = 0
        transform_stride = self.transform.stride

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at parquet#{parquet_start_id}, rg#{row_group_start_id}, row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[parquet_start_id:]
            for parquet_idx, parquet_file_path in enumerate(data_paths_per_worker_, start=parquet_start_id):
                fs = pf.LocalFileSystem()
                with fs.open_input_file(parquet_file_path) as f:
                    fr = pq.ParquetFile(f)
                    row_group_ids = list(range(fr.num_row_groups))
                    row_group_ids_ = row_group_ids[row_group_start_id:]

                    for row_group_id in row_group_ids_:
                        df = fr.read_row_group(row_group_id).to_pandas()
                        df = df.iloc[row_start_id:]

                        for row_idx, row in df.iterrows():
                            num_tokens = 0
                            try:
                                image_byte = row['image']
                                image = Image.open(io.BytesIO(image_byte)).convert('RGB')
                            except Exception as e:
                                print(f'Error: {e} in rg#{row_group_id}, {parquet_file_path}')
                                continue
                            image_tensor = self.transform(image)
                            height, width = image_tensor.shape[1:]
                            num_tokens += width * height // transform_stride ** 2

                            try:
                                caption_dict = row['captions']
                                caption_dict = json.loads(caption_dict)
                            except Exception as e:
                                print(f'Error: {e} in rg#{row_group_id}, {parquet_file_path}')
                                continue

                            caps_token = [self.tokenizer.encode(v) for _, v in caption_dict.items()]
                            if len(caps_token) == 0:
                                print(f'no caption in rg#{row_group_id}, {parquet_file_path}')
                                caption_token = self.tokenizer.encode(' ')
                            else:
                                caption_token = random.choice(caps_token)

                            sequence_plan, text_ids_list = [], []
                            text_ids = caption_token
                            num_tokens += len(caption_token)
                            text_ids_list.append(text_ids)
                            sequence_plan.append({
                                'type': 'text',
                                'enable_cfg': 1,
                                'loss': 0,
                                'special_token_loss': 0,
                                'special_token_label': None,
                            })

                            sequence_plan.append({
                                'type': 'vae_image',
                                'enable_cfg': 0,
                                'loss': 1,
                                'special_token_loss': 0,
                                'special_token_label': None,
                            })

                            sample = dict(
                                image_tensor_list=[image_tensor],
                                text_ids_list=text_ids_list,
                                num_tokens=num_tokens,
                                sequence_plan=sequence_plan,
                                data_indexes={
                                    "data_indexes": [parquet_idx, row_group_id, row_idx],
                                    "worker_id": worker_id,
                                    "dataset_name": self.dataset_name,
                                }
                            )
                            yield sample

                        row_start_id = 0
                    row_group_start_id = 0
            parquet_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")


class SftJSONLIterableDataset(DistributedIterableDataset):
    def __init__(
            self, dataset_name, transform, tokenizer,
            jsonl_path_list, data_dir_list, num_used_data,
            local_rank=0, world_size=1, num_workers=8, data_status=None,
            shuffle_lines=False, shuffle_seed=0,
    ):
        """
        jsonl_path_list: list of jsonl file paths
        data_dir_list: list of image directories containing the images of each jsonl file
        num_used_data: list of number of sampled data points for each jsonl
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.data_paths = self.get_data_paths(
            jsonl_path_list,
            data_dir_list,
            num_used_data,
            shuffle_lines,
            shuffle_seed,
        )
        self.set_epoch()

    def get_data_paths(
            self,
            jsonl_path_list,
            data_dir_list,
            num_used_data,
            shuffle_lines,
            shuffle_seed,
    ):
        data_paths = []
        for jsonl_path, image_dir, num_data_point in zip(
                jsonl_path_list, data_dir_list, num_used_data
        ):
            with open(jsonl_path, 'r') as f:
                raw_data = f.readlines()
            if shuffle_lines:
                self.rng.seed(shuffle_seed)
                self.rng.shuffle(raw_data)
            raw_data = raw_data[:num_data_point]
            data_paths.extend([(json_data, image_dir) for json_data in raw_data])
        return data_paths

    def change_format(self, data, num_images):
        elements = []
        for conversation in data['conversations']:
            if conversation['from'] == 'human':
                if '<image>' not in conversation['value']:
                    elements.append({
                        'type': 'text',
                        'has_loss': 0,
                        'text': conversation['value'],
                    })
                else:
                    text_list = conversation['value'].split('<image>')
                    for idx, text in enumerate(text_list):
                        if text.strip() != '':
                            elements.append({
                                'type': 'text',
                                'has_loss': 0,
                                'text': text.strip(),
                            })
                        if (idx != len(text_list) - 1) and (idx < num_images):
                            elements.append({'type': 'image', })
            elif conversation['from'] == 'gpt':
                elements.append({
                    'type': 'text',
                    'has_loss': 1,
                    'text': conversation['value'],
                })
        return elements

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            row_start_id = self.data_status[worker_id] + 1
        else:
            row_start_id = 0
        transform_stride = self.transform.stride

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            for row_idx, (data, image_dir) in enumerate(data_paths_per_worker_, start=row_start_id):
                num_tokens = 0
                image_tensor_list = []
                text_ids_list = []
                sequence_plan = []

                data_item = json.loads(data)
                raw_images = None
                if 'image' in data_item:
                    if isinstance(data_item.get('image'), list):
                        raw_images = [
                            Image.open(os.path.join(image_dir, image)).convert('RGB')
                            for image in data_item['image']
                        ]
                    else:
                        raw_images = [
                            Image.open(os.path.join(image_dir, data_item['image'])).convert('RGB')
                        ]

                if raw_images:
                    for raw_image in raw_images:
                        image_tensor = self.transform(raw_image, img_num=len(raw_images))
                        image_tensor_list.append(image_tensor)
                        height, width = image_tensor.shape[1:]
                        num_tokens += width * height // transform_stride ** 2

                elements = self.change_format(data_item, len(image_tensor_list))

                for item in elements:
                    if item['type'] == 'text':
                        text_data = item['text']
                        text_ids = self.tokenizer.encode(text_data)
                        if len(text_ids) > 0:
                            text_ids_list.append(text_ids)
                            num_tokens += len(text_ids)
                            current_plan = {
                                'type': 'text',
                                'enable_cfg': 0,
                                'loss': item['has_loss'],
                                'special_token_loss': 0,
                                'special_token_label': None,
                            }
                            sequence_plan.append(current_plan)
                    elif item['type'] == 'image':
                        current_plan = {
                            'type': 'vit_image',
                            'enable_cfg': 0,
                            'loss': 0,
                            'special_token_loss': 0,
                            'special_token_label': None,
                        }
                        sequence_plan.append(current_plan)

                has_loss = [item['loss'] for item in sequence_plan]
                if sum(has_loss) == 0:
                    print(f'No loss defined, skipped.')
                    continue

                yield dict(
                    image_tensor_list=image_tensor_list,
                    text_ids_list=text_ids_list,
                    sequence_plan=sequence_plan,
                    num_tokens=num_tokens,
                    data_indexes={
                        "data_indexes": row_idx,
                        "worker_id": worker_id,
                        "dataset_name": self.dataset_name,
                    }
                )

            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
