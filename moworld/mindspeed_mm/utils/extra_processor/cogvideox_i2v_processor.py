import random
import torch


class CogVideoXI2VProcessor:
    """
    The I2V Processor of CogVideoX:
    1. add noise to first frame
    2. encode the first frame
    3. random dropout image latent

    Args:
        config (dict): the processor config
        {
            "noised_image_all_concat": False,
            "noised_image_dropout": 0.05,
            "noised_image_input": True
        }
    """

    def __init__(self, config):
        self.noised_image_all_concat = config.get("noised_image_all_concat", False)
        self.noised_image_dropout = config.get("noised_image_dropout", 0.05)
        self.noised_image_input = config.get("noised_image_input", True)

    @staticmethod
    def add_noise_to_image(image):
        sigma = torch.normal(mean=-3.0, std=0.5, size=(image.shape[0],)).to(image.device)
        sigma = torch.exp(sigma).to(image.dtype)
        image_noise = torch.randn_like(image) * sigma[:, None, None, None, None]
        image = image + image_noise
        return image

    def __call__(self, vae_model, videos, video_latents, images=None, **kwargs):
        if images is None:
            images = videos[:, 0:1]
        images = images.to(videos.dtype)

        if random.random() < self.noised_image_dropout:
            image_latents = torch.zeros_like(video_latents)
        else:
            if self.noised_image_input:
                images = self.add_noise_to_image(images)
            image_latents = vae_model.encode(images, enable_cp=False)
            if self.noised_image_all_concat:
                image_latents = image_latents.repeat(1, 1, video_latents.size(2), 1, 1)
            else:
                image_latents = torch.concat(
                    [
                        image_latents,
                        torch.zeros_like(video_latents[:, :, 1:])
                    ],
                    dim=2
                )

        return {"masked_video": image_latents}
