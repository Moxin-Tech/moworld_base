from logging import getLogger

import torch
from torch import nn

from megatron.core import mpu
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args

from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.models.predictor.dits.lumina_mgpt2 import ChameleonForConditionalGeneration

logger = getLogger(__name__)


class Lumina(MultiModalModule):
    def __init__(self, config):
        super().__init__(config=config)
        args = get_args()
        self.config = core_transformer_config_from_args(args)
        if not isinstance(config, dict):
            config = config.to_dict()

        self.model = ChameleonForConditionalGeneration(**config["predictor"])
        self.z_loss_weight = config.pop("z_loss_weight", 1e-5)
        self.dtype = config.pop("dtype", torch.bfloat16)

    def forward(self, input_ids, labels, **kwargs):
        max_tokens = max([len(_) for _ in input_ids])
        max_tokens = min(max_tokens, self.model.max_position_embeddings)
        input_ids = [_[:max_tokens] for _ in input_ids]
        labels = [_[:max_tokens] for _ in labels]

        input_ids = [example + [0] * (max_tokens - len(example)) for example in input_ids]
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=self.model.device)

        labels = [label + [-100] * (max_tokens - len(label)) for label in labels]
        labels = torch.tensor(labels, dtype=torch.int64, device=self.model.device)

        with torch.autocast("npu", dtype=self.dtype):
            result = self.model(input_ids=input_ids, labels=labels, **kwargs)

        # loss
        loss = result[0]
        additional_loss_dict = {}
        if self.z_loss_weight > 0:
            logits: torch.Tensor = result[1]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            valid_mask = shift_labels >= 0
            z_loss = torch.logsumexp(shift_logits, dim=-1).pow(2)[valid_mask].mean()
            additional_loss_dict["z_loss"] = (z_loss, self.z_loss_weight)
        return loss, additional_loss_dict

    def state_dict_for_save_checkpoint(self, prefix="", keep_vars=False):
        """Customized state_dict"""
        print("lumina state dict for save checkpoint")
        state_dict = self.model.state_dict(prefix=prefix, keep_vars=keep_vars)
        return state_dict

    def load_state_dict(self, state_dict, strict=False):
        """Customized load."""
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, False)
        if missing_keys is not None:
            logger.info(f"Missing keys in state_dict: {missing_keys}.")
        if unexpected_keys is not None:
            logger.info(f"Unexpected key(s) in state_dict: {unexpected_keys}.")
        return None

    def get_fsdp_wrap_module_list(self):
        modules = [*list(self.model.model.layers), self.model.lm_head, self.model.model.embed_tokens]
        return [('model', modules)]
