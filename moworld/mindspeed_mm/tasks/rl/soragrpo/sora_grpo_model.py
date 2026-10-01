import os
from abc import ABC

import torch
from torch import nn


class SoraGRPOModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.ae = None
        self.diffuser = None
        self.reward = None
        self.text_encoder = None

    def initialize_reward_model(self, args, device):
        reward_model_config = args.mm.model.reward
        if reward_model_config.model_id == "CLIP-ViT-H-14":
            return self.initialize_hps_model(device, reward_model_config)
        else:
            raise ValueError("reward model id is wrong.")

    def initialize_hps_model(self, device, reward_model_config):
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

        def initialize_model(device):
            model_dict = {}
            pretrained = os.path.join(reward_model_config.ckpt_dir, reward_model_config.pretrained)
            model, preprocess_train, preprocess_val = create_model_and_transforms(
                reward_model_config.model_name,
                pretrained,
                precision='amp',
                device=device,
                jit=False,
                force_quick_gelu=False,
                force_custom_text=False,
                force_patch_dropout=False,
                force_image_size=None,
                pretrained_image=False,
                image_mean=None,
                image_std=None,
                light_augmentation=True,
                aug_cfg={},
                output_dict=True,
                with_score_predictor=False,
                with_region_predictor=False
            )
            model_dict['model'] = model
            model_dict['preprocess_val'] = preprocess_val
            return model_dict

        model_dict = initialize_model(device)
        model = model_dict['model']
        preprocess_val = model_dict['preprocess_val']
        cp = os.path.join(reward_model_config.ckpt_dir, reward_model_config.load_pt)
        checkpoint = torch.load(cp, map_location=f'cuda:{device}')
        model.load_state_dict(checkpoint['state_dict'])
        processor = get_tokenizer(reward_model_config.model_name)
        reward_model = model.to(device)
        reward_model.eval()
        model_dict['processor'] = processor
        model_dict['reward_model'] = reward_model
        model_dict['preprocess_val'] = preprocess_val
        return model_dict

    def get_split_modules(self):
        raise NotImplementedError("Subclasses must implement this method")
