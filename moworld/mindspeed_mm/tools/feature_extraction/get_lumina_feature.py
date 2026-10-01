import json
import os
from typing import Dict, Any, Tuple, List

import torch
import torch.distributed

import mindspeed.megatron_adaptor
from megatron.core import mpu
from megatron.training import get_args, print_rank_0
from megatron.training.initialize import initialize_megatron, set_jit_fusion_options
from tqdm import tqdm

from mindspeed_mm.configs.config import merge_mm_args, mm_extra_args_provider
from mindspeed_mm.data import build_mm_dataloader, build_mm_dataset
from mindspeed_mm.data.data_utils.constants import FILE_INFO

from mindspeed_mm.tools.profiler import Profiler
from mindspeed_mm.utils.utils import get_device, get_dtype, is_npu_available


# NPU (Ascend) specific setup if available
if is_npu_available():
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    torch.npu.config.allow_internal_format = False


class FeatureExtractor:
    """
    Distributed feature extractor for multimodal data (text + image)

    This class handles:
    1. Distributed environment setup using Megatron
    2. Data loading and preprocessing
    3. Feature extraction using vqvae (image) and text tokenizer models
    4. Saving extracted features to disk
    5. Metadata management for extracted features
    """

    def __init__(self):
        """Initialize the feature extraction pipeline"""
        # Initialize distributed environment (Megatron)
        self._initialize_distributed()

        # Get save path from configuration
        self.save_path = self.args.mm.tool.sorafeature.save_path
        self.features_dir = os.path.join(self.save_path, "features")
        self.data_info_path = os.path.join(self.save_path, 'data.jsonl')

        # Only rank 0 creates directories to avoid race conditions
        if self.rank == 0:
            os.makedirs(self.features_dir, exist_ok=True)
            print_rank_0(f"Created features directory at: {self.features_dir}")

        # clear data.jsonl
        with open(self.data_info_path, "w", encoding="utf-8") as f:
            pass

        # Configure PyTorch for optimal performance
        set_jit_fusion_options()
        torch.set_grad_enabled(False)

        self.device = get_device("npu")

        # Prepare data pipeline (dataset and dataloader)
        self.dataset, self.dataloader = self._prepare_data()
        torch.distributed.barrier()

    def _initialize_distributed(self):
        """Initialize Megatron distributed training environment"""
        # Initialize Megatron with multimodal-specific arguments
        initialize_megatron(extra_args_provider=mm_extra_args_provider, args_defaults={})

        # Get and merge arguments
        args = get_args()
        merge_mm_args(args)
        self.args = get_args()

        # Store rank and world size for distributed operations
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

        print_rank_0(f"Initialized distributed environment (rank {self.rank}/{self.world_size})")

    def extract_all(self):
        """Main method to extract features from all data samples"""
        total_samples = len(self.dataset)
        print_rank_0(f"Starting feature extraction. Total samples: {total_samples}")

        # Initialize counters and profiler
        counter = 0
        profiler = self._init_profiler()
        if profiler:
            profiler.start()

        try:
            # Process all batches in the dataloader
            for index, batch in tqdm(enumerate(self.dataloader)):
                # Extract features from current batch
                file_names, tokens, labels = batch

                batch_size = len(file_names)
                # Save features for each sample in the batch
                for i in range(batch_size):
                    self._save_sample_features(
                        file_name=file_names[i],
                        tokens=tokens[i],
                        labels=labels[i],
                        index=index
                    )

                # Update profiler if enabled
                if profiler:
                    profiler.step()

        except Exception as e:
            print_rank_0(f"Feature extraction failed: {str(e)}")
            raise
        finally:
            # Clean up profiler
            if profiler:
                profiler.stop()

    def _init_profiler(self):
        """Initialize performance profiler if enabled in configuration"""
        if hasattr(self.args.mm.tool, "profile"):
            print_rank_0("Initializing performance profiler")
            return Profiler(self.args.mm.tool.profile)
        return None

    def _save_sample_features(
        self,
        file_name: str,
        tokens: Tuple[int],
        labels: Tuple[int],
        index: int,
    ):
        """Save extracted features for a single sample to disk"""
        pt_name = self._generate_safe_filename(file_name)
        save_path = os.path.join(self.features_dir, pt_name)
        # Prepare data dictionary
        data_to_save = {"token": tokens, "label": labels}
        # Save to file
        torch.save(data_to_save, save_path)
        record = {"file": save_path, "len": len(tokens)}
        # Write dataset metadata information
        with open(self.data_info_path, "a") as f:
            record_str = json.dumps(record) + "\n"
            f.write(record_str)

    def _prepare_data(self) -> Tuple[Any, Any]:
        """Prepare dataset and data loader"""
        # Build dataset
        dataset = build_mm_dataset(self.args.mm.data.dataset_param)

        # Build dataloader
        dataloader = build_mm_dataloader(
            dataset,
            self.args.mm.data.dataloader_param,
            process_group=mpu.get_data_parallel_group(),
            dataset_param=self.args.mm.data.dataset_param,
        )

        print_rank_0(f"Prepared dataset with {len(dataset)} samples")
        return dataset, dataloader

    @staticmethod
    def _generate_safe_filename(file_path: str) -> str:
        """
        Generate a safe filename without special characters

        Example:
            Input: "/path/to/image/0.jpg"
            Output: "0_jpg.pkl"
        """
        # Extract base name
        base_name = os.path.basename(file_path)
        # Replace dots with underscores to avoid extension issues
        safe_name = base_name.replace(".", "_") + ".pt"
        return safe_name


if __name__ == "__main__":
    # Initialize and run feature extraction
    print_rank_0("Starting feature extraction process")
    extractor = FeatureExtractor()
    extractor.extract_all()
    print_rank_0("Feature extraction completed successfully")
