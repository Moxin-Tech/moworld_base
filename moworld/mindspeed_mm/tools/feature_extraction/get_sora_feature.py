import json
import os
import time
import copy
from typing import Dict, Any, Tuple, List, Union, Optional

import mindspeed.megatron_adaptor
import torch
import torch.distributed
from megatron.core import mpu
from megatron.training import get_args, print_rank_0
from megatron.training.initialize import initialize_megatron, set_jit_fusion_options
from numpy import save
from tqdm import tqdm

from mindspeed_mm.configs.config import merge_mm_args, mm_extra_args_provider
from mindspeed_mm.data import build_mm_dataloader, build_mm_dataset
from mindspeed_mm.data.data_utils.constants import (
    FILE_INFO,
    PROMPT_IDS,
    PROMPT_MASK,
    VIDEO,
    VIDEO_MASK,
)
from mindspeed_mm.models.ae import AEModel
from mindspeed_mm.models.text_encoder import TextEncoder
from mindspeed_mm.tools.profiler import Profiler
from mindspeed_mm.utils.utils import get_device, get_dtype, is_npu_available


# NPU (Ascend) specific setup if available
if is_npu_available():
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    torch.npu.config.allow_internal_format = False


class FeatureExtractor:
    """
    Distributed feature extractor for multimodal data (video + text)

    This class handles:
    1. Distributed environment setup using Megatron
    2. Data loading and preprocessing
    3. Feature extraction using autoencoder (video) and text encoder models
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

        # Only rank 0 creates directories to avoid race conditions
        if self.rank == 0:
            os.makedirs(self.features_dir, exist_ok=True)
            print_rank_0(f"Created features directory at: {self.features_dir}")

        # Configure PyTorch for optimal performance
        set_jit_fusion_options()
        torch.set_grad_enabled(False)

        self.device = get_device("npu")
        self.ae_dtype = get_dtype(self.args.mm.model.ae.dtype)

        # Prepare data pipeline (dataset and dataloader)
        self.dataset, self.dataloader = self._prepare_data()

        # Write dataset metadata information
        self._write_data_info()
        torch.distributed.barrier()

        # Initialize models (autoencoder and text encoder)
        self.vae, self.text_encoder = self._prepare_models()

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

        print(f"Initialized distributed environment (rank {self.rank}/{self.world_size})")

    def _write_data_info(self):
        """
        Write dataset metadata information (JSONL file)
        """
        if self.rank != 0:
            return

        print_rank_0("Writing dataset metadata information...")
        data_info_path = os.path.join(self.save_path, 'data.jsonl')

        with open(data_info_path, 'w', encoding="utf-8") as json_file:
            # Determine data storage format from configuration
            storage_mode = self.args.mm.data.dataset_param.basic_parameters.data_storage_mode

            if storage_mode == "combine":
                source_file_key = "path"
            elif storage_mode == "standard":
                source_file_key = FILE_INFO
            else:
                raise NotImplementedError(f"Unsupported storage mode: {storage_mode}")

            # Process each data sample in the dataset
            for data_sample in self.dataset.data_samples:
                # Get original file path
                file_name = data_sample[source_file_key]

                # Generate safe filename for feature storage
                pt_name = self._generate_safe_filename(file_name)

                # Create metadata entry
                data_info = copy.deepcopy(data_sample)
                data_info[FILE_INFO] = f"features/{pt_name}"

                # Write to JSONL file
                json_file.write(json.dumps(data_info) + '\n')

        print_rank_0(f"Dataset metadata written to: {data_info_path}")

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
            for _, batch in tqdm(enumerate(self.dataloader)):
                # Extract features from current batch
                file_names, latents, latents_dict, prompt, prompt_mask = self._extract_single(batch)
                batch_size = latents.shape[0]
                counter += batch_size

                # Save features for each sample in the batch
                for i in range(batch_size):
                    self._save_sample_features(
                        file_name=file_names[i],
                        latent=latents[i],
                        prompt=prompt,
                        prompt_mask=prompt_mask,
                        sample_idx=i,
                        latents_dict=latents_dict
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

    def _extract_single(
        self,
        batch: Dict[str, Any]
    ) -> Tuple[List[str], torch.Tensor, Dict[str, Any], Any, Any]:
        """
        Extract features from a batch of data

        Returns:
            file_names: List of original file names
            latents: Extracted video features (tensor)
            latents_dict: Additional video features (dict)
            prompt: Extracted text features
            prompt_mask: Text attention masks
        """
        if not batch:
            raise ValueError("Received empty batch")

        video = batch.pop(VIDEO).to(self.device, dtype=self.ae_dtype)
        prompt_ids = batch.pop(PROMPT_IDS)
        prompt_mask = batch.pop(PROMPT_MASK)
        file_names = batch.pop(FILE_INFO)

        # Extract video features using autoencoder
        latents, latents_dict = self.vae.encode(video, **batch)

        # Extract text features using text encoder
        prompt, prompt_mask = self.text_encoder.encode(prompt_ids, prompt_mask)

        return file_names, latents, latents_dict, prompt, prompt_mask

    def _save_sample_features(
        self,
        file_name: str,
        latent: torch.Tensor,
        prompt: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
        prompt_mask: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
        sample_idx: int,
        latents_dict: Optional[Dict[str, Any]] = None
    ):
        """Save extracted features for a single sample to disk"""
        pt_name = self._generate_safe_filename(file_name)
        save_path = os.path.join(self.features_dir, pt_name)

        # Prepare data dictionary
        data_to_save = {
            VIDEO: latent.cpu(),  # Move to CPU before saving
            PROMPT_IDS: self._extract_prompt_component(prompt, sample_idx),
            PROMPT_MASK: self._extract_prompt_component(prompt_mask, sample_idx)
        }

        # Add i2v additional latents if present
        if latents_dict:
            for key, value in latents_dict.items():
                item = value[sample_idx]
                # Move tensors to CPU, leave other types as-is
                data_to_save[key] = item.cpu() if isinstance(item, torch.Tensor) else item

        # Save to file
        torch.save(data_to_save, save_path)

    def _extract_prompt_component(
        self,
        prompt: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
        idx: int
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Extract prompt component for a specific sample index"""
        if isinstance(prompt, (list, tuple)):
            # Handle multi prompts
            return [p[idx].cpu() for p in prompt]
        # Handle single prompt
        return prompt[idx].cpu()

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

    def _prepare_models(self) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """Initialize and configure models for inference"""
        # Initialize autoencoder (video feature extractor)
        vae = AEModel(self.args.mm.model.ae)
        vae = vae.to(self.device, dtype=self.ae_dtype).eval()

        # Initialize text encoder
        text_encoder = TextEncoder(self.args.mm.model.text_encoder)
        text_encoder = text_encoder.to(self.device).eval()

        print_rank_0("Models initialized and moved to evaluation mode")
        return vae, text_encoder

    @staticmethod
    def _generate_safe_filename(file_path: str) -> str:
        """
        Generate a safe filename without special characters

        Example:
            Input: "/path/to/video.mp4"
            Output: "video_mp4.pt"
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
