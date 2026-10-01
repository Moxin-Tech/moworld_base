import random
from typing import Union, Optional, List
import torch
import torchvision.transforms as transforms
import numpy as np
import PIL.Image
from PIL import Image
from transformers import CLIPImageProcessor


def numpy_to_pil(images: np.ndarray) -> List[PIL.Image.Image]:
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def black_image(width, height):
    b_image = Image.new("RGB", (width, height), (0, 0, 0))
    return b_image


class HunyuanVideoI2VProcessor:
    """
    The I2V Processor of HunyuanVideo:

    """

    def __init__(self, config):
        self.sematic_cond_drop_p = config.get("sematic_cond_drop_p", 0)
        processor_path = config.get("processor_path", None)
        self.processor = CLIPImageProcessor.from_pretrained(processor_path)

    @staticmethod
    def get_cond_latents(latents, vae):
        """get conditioned latent by decode and encode the first frame latents"""
        first_image_latents = latents[:, :, 0, ...] if len(latents.shape) == 5 else latents
        first_image_latents = 1 / vae.scaling_factor * first_image_latents
        first_images = vae.decode(first_image_latents.unsqueeze(2))
        first_images = first_images.squeeze(2)
        first_images = (first_images / 2 + 0.5).clamp(0, 1)
        first_images = first_images.cpu().permute(0, 2, 3, 1).float().numpy()
        first_images = numpy_to_pil(first_images)

        image_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
        first_images_pixel_values = [image_transform(image) for image in first_images]
        first_images_pixel_values = (
            torch.cat(first_images_pixel_values).unsqueeze(0).unsqueeze(2).to(latents.device)
        )

        cond_latents = vae.encode(first_images_pixel_values.to(latents.dtype))

        return cond_latents

    @staticmethod
    def get_cond_images(latents, vae, is_uncond=False):
        """get conditioned images by decode the first frame latents"""
        sematic_image_latents = (
            latents[:, :, 0, ...] if len(latents.shape) == 5 else latents
        )
        sematic_image_latents = 1 / vae.scaling_factor * sematic_image_latents
        semantic_images = vae.decode(sematic_image_latents.unsqueeze(2))
        semantic_images = semantic_images.squeeze(2)
        semantic_images = (semantic_images / 2 + 0.5).clamp(0, 1)
        semantic_images = semantic_images.cpu().permute(0, 2, 3, 1).float().numpy()
        semantic_images = numpy_to_pil(semantic_images)
        if is_uncond:
            semantic_images = [black_image(img.size[0], img.size[1]) for img in semantic_images]

        return torch.tensor(np.array(semantic_images))

    def __call__(self, vae_model, videos, video_latents, **kwargs):
        cond_latents = self.get_cond_latents(video_latents, vae_model)
        is_uncond = (
            torch.tensor(1).to(torch.int64)
            if random.random() < self.sematic_cond_drop_p
            else torch.tensor(0).to(torch.int64)
        )
        semantic_images = self.get_cond_images(video_latents, vae_model, is_uncond=is_uncond)
        pixel_values = self.processor(semantic_images, return_tensors="pt")["pixel_values"].to(video_latents.device)

        return {"cond_latents": cond_latents, "semantic_images": semantic_images, "pixel_values": pixel_values}
