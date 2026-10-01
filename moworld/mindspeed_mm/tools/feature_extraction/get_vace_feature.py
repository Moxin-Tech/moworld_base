import copy
import hashlib
import json
import os
import random
from typing import List, Optional, Union, Dict, Any, Tuple
import mindspeed.megatron_adaptor
import torch
import torch.distributed
from PIL import Image
from datasets import tqdm
from einops import rearrange
from megatron.core import mpu
from megatron.training import get_args, print_rank_0
from megatron.training.initialize import initialize_megatron, set_jit_fusion_options
from numpy import save
from mindspeed_mm.configs.config import merge_mm_args, mm_extra_args_provider
from mindspeed_mm.data import build_mm_dataloader, build_mm_dataset
from mindspeed_mm.data.data_utils.constants import (
    FILE_INFO,
    PROMPT_IDS,
    PROMPT_MASK,
    VIDEO,
    VIDEO_MASK,
)
from mindspeed_mm.data.data_utils.transform_pipeline import get_transforms
from mindspeed_mm.data.datasets.t2v_dataset import T2VDataset
from mindspeed_mm.models.ae import AEModel
from mindspeed_mm.models.text_encoder import TextEncoder
from mindspeed_mm.tools.profiler import Profiler
from mindspeed_mm.utils.utils import get_device, get_dtype, is_npu_available
from mindspeed_mm.tools.feature_extraction.get_sora_feature import FeatureExtractor

if is_npu_available():
    import torch_npu
    from torch_npu.contrib import transfer_to_npu

    torch.npu.config.allow_internal_format = False


class VACEDataset(T2VDataset):
    def __getitem__(self, index):
        example = {}
        sample = self.data_samples[index]
        input_video = sample['video']
        src_video = sample["src_video"]
        src_mask = sample["src_video_mask"]
        image_size = []
        if src_mask is not None and src_video is not None:
            src_video, src_mask, input_video = self.video_reader(src_video), self.video_reader(
                src_mask), self.video_reader(input_video)
            src_video, src_mask, input_video, _, _, _ = self.video_processer(src_video, src_mask, input_video)
            # (T,C,H,W) -> (C,T,H,W)
            src_mask = src_mask.permute(1, 0, 2, 3)
            src_mask = torch.clamp((src_mask[:1, :, :, :] + 1) / 2, min=0, max=1)
            example["src_video"] = src_video.permute(1, 0, 2, 3)
            example["src_video_mask"] = src_mask
            example["video"] = input_video.permute(1, 0, 2, 3)
            image_size = src_video.shape[2:]
        elif src_video is not None:
            src_video, input_video = self.video_reader(src_video), self.video_reader(input_video)
            src_video, input_video, _, _, _ = self.video_processer(src_video, input_video)
            example["src_video"] = src_video.permute(1, 0, 2, 3)
            example["video"] = input_video.permute(1, 0, 2, 3)
            image_size = src_video.shape[2:]
        elif src_video is None and src_mask is None:
            input_video = self.video_reader(input_video)
            input_video, _, _, _ = self.video_processer(input_video)
            example["video"] = input_video.permute(1, 0, 2, 3)
            image_size = input_video.shape[2:]

        if sample["src_ref_images"]:
            images = []
            for image_path in sample["src_ref_images"]:
                self.image_processer.image_transforms = get_transforms(is_video=False,
                                                                       train_pipeline=self.train_pipeline,
                                                                       image_size=image_size)
                self.image_processer.is_image = True
                ref_img = self.image_processer(image_path)
                images.append(ref_img)
            example["src_ref_images"] = images

        text = sample["cap"]
        if not isinstance(text, list):
            text = [text]
        text = [random.choice(text)]
        if self.use_text_processer:
            prompt_ids, prompt_mask = self.get_text_processer(text)
            example[PROMPT_IDS], example[PROMPT_MASK] = prompt_ids[0], prompt_mask[0]
        else:
            example["text"] = text

        file_path = text[0].encode(encoding="UTF-8")
        file_path = hashlib.md5(file_path).hexdigest()
        example[FILE_INFO] = file_path
        return example


class VACEFeatureExtractor(FeatureExtractor):
    def _prepare_data(self):
        args = get_args()
        task = args.mm.model.task if hasattr(args.mm.model, "task") else "vace"

        dataset_param = args.mm.data.dataset_param.to_dict()
        dataset = VACEDataset(
            basic_param=dataset_param["basic_parameters"],
            vid_img_process=dataset_param["preprocess_parameters"],
            **dataset_param
        )
        dataloader = build_mm_dataloader(
            dataset,
            args.mm.data.dataloader_param,
            process_group=mpu.get_data_parallel_group(),
            dataset_param=args.mm.data.dataset_param,
        )

        return dataset, dataloader

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

            if storage_mode == "combine" or storage_mode == "vace":
                source_file_key = "path"
            elif storage_mode == "standard":
                source_file_key = FILE_INFO
            else:
                raise NotImplementedError(f"Unsupported storage mode: {storage_mode}")

            for data_sample in self.dataset.data_samples:
                file_name = copy.deepcopy(data_sample["cap"])
                file_name = file_name.encode(encoding="UTF-8")
                file_name = hashlib.md5(file_name).hexdigest()
                pt_name = self._generate_safe_filename(file_name)
                data_info = copy.deepcopy(data_sample)
                data_info[FILE_INFO] = f"features/{pt_name}"

                json_file.write(json.dumps(data_info, ensure_ascii=False) + '\n')

        print_rank_0(f"Dataset metadata written to: {data_info_path}")

    def _extract_single(
            self,
            batch: Dict[str, Any]
    ) -> Tuple[List[str], torch.Tensor, Dict[str, Any], torch.Tensor, Any, Any]:
        """
        Extract features from a batch of data

        Returns:
            file_names: List of original file names
            video_latents: Extracted video features (tensor)
            video_latents_dict: Additional video features (dict)
            vace_context: Extracted vace features (tensor)
            prompt: Extracted text features
            prompt_mask: Text attention masks
        """
        if not batch:
            raise ValueError("Received empty batch")

        # Extract video features using autoencoder
        video = batch.pop("video").to(self.device, dtype=self.ae_dtype)
        video_latents, latents_dict = self.vae.encode(video, **batch)

        vace_reference_image = None
        vace_reference_latents = None
        if "src_ref_images" in batch:
            vace_reference_image = torch.cat(batch["src_ref_images"], dim=2).to(dtype=self.ae_dtype, device=self.device)
            vace_reference_latents, _ = self.vae.encode(vace_reference_image, **batch)
            vace_reference_latents = vace_reference_latents.to(dtype=self.ae_dtype, device=self.device)
            video_latents = torch.concat([vace_reference_latents, video_latents], dim=2)
        num_frames, height, width = video.shape[2], video.shape[3], video.shape[4]
        # Extract vace_context features using autoencoder
        if "src_video" not in batch:
            vace_video = torch.zeros((1, 3, num_frames, height, width), dtype=self.ae_dtype, device=self.device)
        else:
            vace_video = batch["src_video"].to(dtype=self.ae_dtype, device=self.device)
        if "src_video_mask" not in batch:
            vace_video_mask = torch.ones_like(vace_video, dtype=self.ae_dtype, device=self.device)
        else:
            vace_video_mask = batch["src_video_mask"].to(dtype=self.ae_dtype, device=self.device)
        inactive = vace_video * (1 - vace_video_mask) + 0 * vace_video_mask
        reactive = vace_video * vace_video_mask + 0 * (1 - vace_video_mask)
        inactive, _ = self.vae.encode(inactive, **batch)
        reactive, _ = self.vae.encode(reactive, **batch)
        vace_video_latents = torch.concat((inactive, reactive), dim=1)
        vace_mask_latents = rearrange(vace_video_mask[0, 0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8)
        vace_mask_latents = torch.nn.functional.interpolate(vace_mask_latents, size=(
            (vace_mask_latents.shape[2] + 3) // 4, vace_mask_latents.shape[3], vace_mask_latents.shape[4]),
                                                            mode='nearest-exact')

        if "src_ref_images" in batch:
            vace_reference_latents = torch.concat((vace_reference_latents, torch.zeros_like(vace_reference_latents)),
                                                  dim=1)
            vace_video_latents = torch.concat((vace_reference_latents, vace_video_latents), dim=2)
            vace_mask_latents = torch.concat(
                (torch.zeros_like(vace_mask_latents[:, :, :vace_reference_latents.shape[2]]), vace_mask_latents), dim=2)

        vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)

        prompt_ids = batch.pop(PROMPT_IDS)
        prompt_mask = batch.pop(PROMPT_MASK)
        file_names = batch.pop(FILE_INFO)
        # Extract text features using text encoder
        prompt, prompt_mask = self.text_encoder.encode(prompt_ids, prompt_mask)
        single_output = [file_names, video_latents, latents_dict, vace_context, prompt, prompt_mask]
        return single_output

    def extract_all(self):
        """Main method to extract features from all data samples"""

        total_samples = len(self.dataset)
        print_rank_0(f"Starting feature extraction. Total samples: {total_samples}")

        counter = 0
        profiler = self._init_profiler()
        if profiler:
            profiler.start()

        try:
            for _, batch in tqdm(enumerate(self.dataloader)):
                single_output = self._extract_single(batch)
                file_names, latents, latents_dict, vace_context, prompt, prompt_mask = single_output
                batch_size = latents.shape[0]
                counter += batch_size
                for i in range(batch_size):
                    self._save_vace_sample_features(
                        file_name=file_names[i],
                        latent=latents[i],
                        vace_context=vace_context[i],
                        prompt=prompt,
                        prompt_mask=prompt_mask,
                        sample_idx=i,
                        latents_dict=latents_dict
                    )

                if profiler:
                    profiler.step()

        except Exception as e:
            print_rank_0(f"Feature extraction failed: {str(e)}")
            raise
        finally:
            if profiler:
                profiler.stop()

    def _save_vace_sample_features(
            self,
            file_name: str,
            latent: torch.Tensor,
            vace_context: torch.Tensor,
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
            PROMPT_MASK: self._extract_prompt_component(prompt_mask, sample_idx),
            "vace_context": vace_context.cpu()
        }

        if latents_dict:
            for key, value in latents_dict.items():
                item = value[sample_idx]
                # Move tensors to CPU, leave other types as-is
                data_to_save[key] = item.cpu() if isinstance(item, torch.Tensor) else item

        torch.save(data_to_save, save_path)


if __name__ == "__main__":
    print_rank_0("Starting feature extraction process")
    extractor = VACEFeatureExtractor()
    extractor.extract_all()
    print_rank_0("Feature extraction completed successfully")
