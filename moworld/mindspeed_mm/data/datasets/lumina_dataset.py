import os
import copy
import json
import traceback
import logging
import warnings
from time import sleep
from pathlib import Path
from typing import List, Union, Optional

import torch.distributed as dist
from torch.utils.data import Dataset

from mindspeed_mm.utils.utils import Registry
from mindspeed_mm.data.data_utils.lumina_item_processor import ItemProcessor

logger = logging.getLogger(__name__)


class LuminaConversationDataset(Dataset):
    def __init__(
        self,
        basic_param: dict,
        preprocess_parameters: dict,
        tokenizer_config: Optional[Union[dict, List[dict]]] = None,
        **kwargs,
    ):
        self.config = basic_param.pop("data_config", {})
        # Create item processor with configuration
        self.item_processor = ItemProcessor.create(**preprocess_parameters, tokenizer_config=tokenizer_config)
        self.meta_collection, self.annotations_collection = self._collect_annotations()

    def __len__(self):
        return sum([_["len"] for _ in self.meta_collection])

    def _collect_annotations(self):
        meta_collection = []
        annotations_collection = []

        meta, annotations = self._load_meta(self.config)
        meta_collection.append(meta)
        annotations_collection.append(annotations)

        return meta_collection, annotations_collection

    def _load_meta(self, meta):
        if "type" not in meta:
            meta["type"] = "default"

        meta_path, meta_type = meta["path"], meta["type"]
        meta_ext = os.path.splitext(meta_path)[-1]
        if meta_ext == ".json":
            with open(meta_path) as f:
                annotations = json.load(f)
        elif meta_ext == ".jsonl":
            annotations = []
            with open(meta_path) as f:
                for i, line in enumerate(f):
                    try:
                        annotations.append(json.loads(line))
                    except json.decoder.JSONDecodeError as e:
                        logger.error(f"Error decoding the following jsonl line ({i}):\n{line.rstrip()}")
                        raise e
        else:
            raise NotImplementedError(
                f'Unknown meta file extension: "{meta_ext}". '
                f"Currently, .json, .jsonl are supported. "
                "If you are using a supported format, please set the file extension so that the proper parsing "
                "routine can be called."
            )
        logger.info(f"{meta_path}, type{meta_type}: len {len(annotations)}")

        meta["len"] = len(annotations)
        meta["item_len_list"] = [self.item_processor.predict_item_token_length(_) for _ in annotations]
        return meta, annotations

    def __getitem__(self, index):
        meta_idx, idx_in_meta = self.tie_index_to_meta(index)
        try:
            return self.get_item_func(meta_idx, idx_in_meta)
        except Exception as e:
            logger.info(
                f"Item {index} errored, annotation:\n"
                f"{self.annotations_collection[meta_idx][idx_in_meta]}\n"
                f"Error:\n"
                f"{traceback.format_exc()}"
            )
            if idx_in_meta != 0:
                return self[index - 1]
            else:
                return self[index + self.meta_collection[meta_idx]["len"] - 1]

    def tie_index_to_meta(self, idx: int):
        # Initialize the starting index
        start_idx = 0
        # Iterate through the list of dictionaries
        for i, meta in enumerate(self.meta_collection):
            # Calculate the ending index for the current collection
            end_idx = start_idx + meta["len"]
            # Check if the given index falls within the current collection
            if start_idx <= idx < end_idx:
                # Calculate the new index within the current collection
                new_index = idx - start_idx
                return i, new_index
            # Update the starting index for the next collection
            start_idx = end_idx
        # If the index is out of range of all collections, raise an error
        raise IndexError("Index out of range")

    def get_item_func(self, meta_idx, idx_in_meta):
        data_item = self.annotations_collection[meta_idx][idx_in_meta]
        data_item = copy.deepcopy(data_item)
        return self.item_processor.process_item(data_item, training_mode=True)
