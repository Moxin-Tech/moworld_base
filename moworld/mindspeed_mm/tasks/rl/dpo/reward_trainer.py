# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
# Copyright 2025 The KwaiVGI team. All rights reserved.

from copy import deepcopy
from functools import partial

import math
import torch
import torch.nn.functional as F

from megatron.core.enums import ModelType
from megatron.training import get_args, print_rank_0
from megatron.training.checkpointing import load_checkpoint
from megatron.training.global_vars import set_args
from megatron.training.training import get_model
from megatron.training.utils import average_losses_across_data_parallel_group
from mindspeed_mm.models.reward_model import Qwen2VLRewardModelBT
from mindspeed_mm.tasks.finetune.lora.utils import is_enable_lora
from mindspeed_mm.tasks.rl.dpo.dpo_trainer import DPOTrainer
from mindspeed_mm.utils.transformer_model_config import get_model_config
from mindspeed_mm.data.data_utils.func_utils.convert import load_reward_tokenizer


class PartialEmbeddingUpdater:
    """Function: Update only the embeddings of special tokens, while freezing the embeddings of regular tokens."""

    def __init__(self):
        # The list of special token IDs that require updates.
        self.special_token_ids = None
        self.orig_embeds_params = None
        self.vocab_size = None

    def get_model_args(self, special_token_ids, enable_partial_update):
        self.special_token_ids = special_token_ids
        self.enable_partial_update = enable_partial_update

    def setup(self, model):
        """Pre-training Initialization: Backup the initial weights of the model's input embedding layer. """
        self.device = torch.cuda.current_device()
        input_embeddings = model.text_decoder.embedding.word_embeddings
        self.orig_embeds_params = input_embeddings.weight.clone().detach()
        self.orig_embeds_params = self.orig_embeds_params.to(self.device)
        self.vocab_size = self.orig_embeds_params.shape[0]

    def __call__(self, model, *kwargs):
        """After each training step, execute: restore the embedding weights of the regular tokens. """
        if self.special_token_ids and self.enable_partial_update:
            # Generate "Recovery Index"
            index_no_updates = torch.ones((self.vocab_size), dtype=torch.bool, device=self.device)
            index_no_updates[self.special_token_ids] = False

            # Restore regular token embedding (disable gradient)
            with torch.no_grad():
                input_embeddings = model.text_decoder.embedding.word_embeddings
                input_embeddings.weight[index_no_updates] = self.orig_embeds_params[index_no_updates]


class VideoVLMRewardTrainer(DPOTrainer):
    """
    A trainer class for Video Reward Model.

    This class provides methods for model initialize, computing losses and metrics, and training.
    """

    def __init__(
        self,
        train_valid_test_dataset_provider,
        model_type,
        process_non_loss_data_func=None,
        extra_args_provider=None,
        args_defaults=None,
    ):
        """
        Initializes the VideoVLMReward instance.

        Sets up the instance variables for the model provider, actual micro batch size,
        and initializes the VideoVLMReward model.
        """
        self.partialEmbeddingUpdater = PartialEmbeddingUpdater()
        super().__init__(
            train_valid_test_dataset_provider,
            model_type,
            process_non_loss_data_func,
            extra_args_provider,
            args_defaults,
            call_backs=[self.partialEmbeddingUpdater]
        )
        self.disable_dropout()

    def model_provider(self, pre_process=True, post_process=True):
        """Builds the model."""
        args = get_args()
        print_rank_0("building VideoReward model ...")
        self.loss_type = args.mm.model.loss_type
        self.loss_dtype = torch.bfloat16 if args.mm.model.text_decoder.bf16 else torch.float32

        vlm_config = deepcopy(args.mm.model)
        data_config = deepcopy(args.mm.data)

        if not isinstance(data_config, dict):
            data_config = data_config.to_dict()
        preprocess_param = data_config['dataset_param']['preprocess_parameters']

        special_token_ids = None
        token_embedding_length = None
        tokenizer_module = load_reward_tokenizer(preprocess_param)
        tokenizer, processor = tokenizer_module['tokenizer'], tokenizer_module['processor']

        if preprocess_param['split_special_tokens']:
            special_tokens = ["<|VQ_reward|>", "<|MQ_reward|>", "<|TA_reward|>"]
            tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
            special_token_ids = tokenizer.convert_tokens_to_ids(special_tokens)
            token_embedding_length = len(tokenizer)

        tokenizer_padding_side = "right"
        pad_token_id = tokenizer.pad_token_id
        model_args = {"special_token_ids": special_token_ids, "token_embedding_length": token_embedding_length,
                      "tokenizer_padding_side": tokenizer_padding_side, "pad_token_id": pad_token_id}

        vlm_config.pre_process = pre_process
        vlm_config.post_process = post_process
        vlm_config.reward_process = True

        if vlm_config.image_encoder and vlm_config.text_decoder:
            vlm_config.image_encoder.vision_encoder = get_model_config(vlm_config.image_encoder.vision_encoder)
            vlm_config.image_encoder.vision_projector = get_model_config(vlm_config.image_encoder.vision_projector)
            vlm_config.text_decoder = get_model_config(vlm_config.text_decoder)

            model = Qwen2VLRewardModelBT(config=vlm_config, extra_config=model_args)

            model.freeze(freeze_image_encoder=getattr(vlm_config.image_encoder.vision_encoder, 'freeze', False),
                         freeze_image_projection=getattr(vlm_config.image_encoder.vision_projector, 'freeze', False),
                         freeze_text_decoder=getattr(vlm_config.text_decoder, 'freeze', False))

        else:
            raise AttributeError("image_encoder config or text_decoder config not exist!")

        self.token_embedding_length = token_embedding_length
        enable_partial_update = getattr(vlm_config.text_decoder, 'word_embeddings_only_update_special', False)
        self.partialEmbeddingUpdater.get_model_args(special_token_ids, enable_partial_update)
        return model

    def disable_dropout(self):
        """
        disable dropout
        """
        args_ = get_args()
        args_.attention_dropout = 0.0
        args_.hidden_dropout = 0.0
        args_.retro_encoder_hidden_dropout = 0.0
        args_.retro_encoder_attention_dropout = 0.0
        set_args(args_)

    def _convert_A_B_to_chosen_rejected(self, rewards_A, rewards_B, scores_A, scores_B, chosen_label, label_dim=None):
        """
        Inputs:
            rewards_A, rewards_B: [B, N]
            scores_A, scores_B: [B, N]
            chosen_label: [B, N]
        Outputs:
            rewards_chosen, rewards_rejected: [B, N]
            scores_chosen, scores_rejected: [B, N]
            nontied_mask: [B, N] (preference labels that is not tied)
            valid_mask: [B, N]  (all valid labels)
        """
        chosen_mask = (chosen_label == 1)
        rejected_mask = (chosen_label != 1)
        if label_dim is not None:
            N = chosen_label.size(1)
            chosen_mask = chosen_mask[:, label_dim].unsqueeze(1).expand(-1, N)
            rejected_mask = rejected_mask[:, label_dim].unsqueeze(1).expand(-1, N)

        rewards_chosen = torch.where(chosen_mask, rewards_A, rewards_B)
        rewards_rejected = torch.where(rejected_mask, rewards_A, rewards_B)
        scores_chosen = torch.where(chosen_mask, scores_A, scores_B)
        scores_rejected = torch.where(rejected_mask, scores_A, scores_B)

        nontied_mask = ((chosen_label == 1) | (chosen_label == -1)).float()
        if label_dim is not None:
            nontied_mask = nontied_mask[:, label_dim].unsqueeze(1).expand(-1, N)

        valid_mask = (chosen_label != 22).float()
        if label_dim is not None:
            valid_mask = valid_mask[:, label_dim].unsqueeze(1).expand(-1, N)

        return rewards_chosen, rewards_rejected, scores_chosen, scores_rejected, nontied_mask, valid_mask

    @staticmethod
    def get_batch(data_iterator):
        """Generate a batch."""
        if data_iterator is not None:
            batch = next(data_iterator)
        else:
            raise ValueError("Data iterator is None. Unable to retrieve batch.")

        device = torch.cuda.current_device()
        batch['input_ids_A'] = batch['input_ids_A'].to(device)
        batch['attention_mask_A'] = batch['attention_mask_A'].to(device)
        batch['pixel_values_A'] = batch['pixel_values_A'].to(device)
        batch['image_grid_thw_A'] = batch['image_grid_thw_A'].to(device)
        batch['input_ids_B'] = batch['input_ids_B'].to(device)
        batch['attention_mask_B'] = batch['attention_mask_B'].to(device)
        batch['pixel_values_B'] = batch['pixel_values_B'].to(device)
        batch['image_grid_thw_B'] = batch['image_grid_thw_B'].to(device)
        batch['A_scores'] = torch.tensor(batch['A_scores']).to(device)
        batch['B_scores'] = torch.tensor(batch['B_scores']).to(device)
        batch['chosen_label'] = torch.tensor(batch['chosen_label']).to(device)

        return batch

    def loss_func(self, rewards_chosen, rewards_rejected, nontied_mask, valid_mask, inputs, output_tensor):
        rewards_A, rewards_B = output_tensor[0], output_tensor[1]
        metrics = {}

        if self.loss_type == "bt":
            # Bradley-Terry model
            loss = -F.logsigmoid(rewards_chosen - rewards_rejected)
            out_mask = nontied_mask
        elif self.loss_type == "margin":
            # Bradley-Terry model with margin
            loss = -F.logsigmoid(rewards_chosen - rewards_rejected - inputs["margin"])
            out_mask = nontied_mask
        elif self.loss_type == "constant_margin":
            # Bradley-Terry model with constant margin
            loss = -F.logsigmoid(rewards_chosen - rewards_rejected - 0.57)
            out_mask = nontied_mask
        elif self.loss_type == "scaled":
            # Bradley-Terry model with scaled margin
            loss = (-(inputs["margin"] + 0.0) * F.logsigmoid(rewards_chosen - rewards_rejected))
            out_mask = nontied_mask
        elif self.loss_type == "reg":
            # regression loss
            rewards = torch.stack([rewards_A, rewards_B], dim=1)
            scores = torch.stack([inputs["A_scores"], inputs["B_scores"]], dim=1)
            out_mask = scores != 0.0
            scores = (scores - 3.0)     # rescale
            loss = F.mse_loss(rewards, scores, reduction="none")
        elif self.loss_type == "btt":
            # Bradley-Terry-With-Ties model
            k = 5.0
            log_k = math.log(k)
            log_k2_sub_1 = math.log(k ** 2 - 1)
            bt_loss = -F.logsigmoid(rewards_chosen - rewards_rejected - log_k)
            same_loss = -F.logsigmoid(rewards_chosen - rewards_rejected - log_k) \
                        - F.logsigmoid(rewards_rejected - rewards_chosen - log_k) \
                        - log_k2_sub_1
            loss = bt_loss * nontied_mask + same_loss * (1 - nontied_mask)
            out_mask = valid_mask
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented.")

        loss = loss * out_mask

        loss = loss.mean()
        # Reduce loss for logging.
        metrics['loss'] = average_losses_across_data_parallel_group([loss])
        for key in metrics.keys():
            metrics[key] = average_losses_across_data_parallel_group([metrics[key]])

        return loss, metrics

    def forward_step(self, data_iterator, model):
        batch = self.get_batch(data_iterator)

        input_ids_A = batch['input_ids_A']
        attention_mask_A = batch['attention_mask_A']
        pixel_values_A = batch['pixel_values_A']
        image_grid_thw_A = batch['image_grid_thw_A']
        input_ids_B = batch['input_ids_B']
        attention_mask_B = batch['attention_mask_B']
        pixel_values_B = batch['pixel_values_B']
        image_grid_thw_B = batch['image_grid_thw_B']
        A_scores = batch['A_scores']
        B_scores = batch['B_scores']
        chosen_label = batch['chosen_label']

        rewards_A = model(input_ids=input_ids_A, pixel_values=pixel_values_A, image_grid_thw=image_grid_thw_A,
                          attention_mask=attention_mask_A).to(self.loss_dtype)
        rewards_B = model(input_ids=input_ids_B, pixel_values=pixel_values_B, image_grid_thw=image_grid_thw_B,
                          attention_mask=attention_mask_B).to(self.loss_dtype)

        rewards_chosen, rewards_rejected, scores_chosen, scores_rejected, nontied_mask, valid_mask = self._convert_A_B_to_chosen_rejected(
            rewards_A, rewards_B, A_scores, B_scores, chosen_label
        )
        batch["margin"] = scores_chosen - scores_rejected

        output_tensor = torch.stack([rewards_A, rewards_B], dim=0)

        return output_tensor, partial(self.loss_func, rewards_chosen, rewards_rejected, nontied_mask, valid_mask, batch)
