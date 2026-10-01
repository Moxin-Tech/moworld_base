import torch


class StepVideoI2VProcessor:
    """
    The I2V Processor of StepVideo:
    1. add noise to first frame
    2. encode the first frame
    3. random dropout image latent
    """

    def __init__(self, config):
        super().__init__()

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
        images = self.add_noise_to_image(images)
        img_emb = vae_model.encode(images).repeat(videos.size(0), 1, 1, 1, 1).to(videos.device)
        padding_tensor = torch.zeros_like(video_latents[:, 1:])
        condition_hidden_states = torch.cat([img_emb, padding_tensor], dim=1)
        return {"image_latents": condition_hidden_states.to(videos.dtype)}
