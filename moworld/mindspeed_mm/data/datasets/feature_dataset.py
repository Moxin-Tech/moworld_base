from typing import Union, List, Optional
import copy
import os

import torch

from mindspeed_mm.data.datasets.t2v_dataset import T2VDataset
from mindspeed_mm.data.datasets.mm_base_dataset import MMBaseDataset
from mindspeed_mm.data.data_utils.constants import (
    PROMPT_IDS,
    PROMPT_MASK,
    VIDEO,
    FILE_INFO,
    FILE_REJECTED_INFO,
    VIDEO_REJECTED,
    SCORE,
    SCORE_REJECTED
)


# Define the structure of the output data for the T2V dataset
T2VOutputData = {
    VIDEO: [],
    PROMPT_IDS: [],
    PROMPT_MASK: []
}


class FeatureDataset(MMBaseDataset):
    def __init__(
        self,
        basic_param: dict
    ):
        """
        Initialize the FeatureDataset class, which is designed to handle feature data for multimodal models.

        Args:
            basic_param (dict): A dictionary containing basic configuration parameters for the dataset.
        """
        super().__init__(**basic_param)

    def __getitem__(self, index: int) -> dict:
        """
        Get the data sample at the specified index.

        Args:
            index (int): The index of the data sample to retrieve.

        Returns:
            dict: A dictionary containing the processed data sample with video features, masks, and text prompts.
        """
        sample = self.data_samples[index]

        # Initialize the output data structure
        examples = copy.deepcopy(T2VOutputData)
        feature_file_path = sample[FILE_INFO]
        if self.data_folder:
            feature_file_path = os.path.join(self.data_folder, feature_file_path)

        # Load feature data from the specified file
        feature_data = self.get_data_from_feature_data(feature_file_path)

        # Extract video features
        examples[VIDEO] = feature_data.pop(VIDEO, None)

        # Extract text prompt IDs and masks
        examples[PROMPT_IDS] = feature_data.pop(PROMPT_IDS, None)
        examples[PROMPT_MASK] = feature_data.pop(PROMPT_MASK, None)

        # DPO (Direct Preference Optimization) handling
        if FILE_REJECTED_INFO in sample.keys():
            rejected_feature_file_path = os.path.join(self.data_folder, sample[FILE_REJECTED_INFO])
            rejected_feature_data = self.get_data_from_feature_data(rejected_feature_file_path)
            examples[VIDEO_REJECTED] = rejected_feature_data.pop(VIDEO, None)
            examples[SCORE] = sample[SCORE]
            examples[SCORE_REJECTED] = sample[SCORE_REJECTED]

        # Add any remaining keys from feature_data to examples
        for key in feature_data.keys():
            examples[key] = feature_data[key]

        return examples


    def get_data_from_feature_data(self, feature_path: str) -> dict:
        """
        Load feature data from a specified file path.

        Args:
            feature_path (str): The path to the feature data file.

        Returns:
            dict: A dictionary containing the loaded feature data.
        """
        if feature_path.endswith((".pt", ".pth")):
            return torch.load(feature_path, map_location=torch.device('cpu'), weights_only=False)
        raise NotImplementedError("Unsupported file format. Only .pt/.pth files are currently supported.")
