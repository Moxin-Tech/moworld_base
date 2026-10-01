# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#               2025 Alibaba Inc (authors: Xiang Lyu, Yabin Li, Qihua, Shengqiang Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
import random

import os
from functools import partial
import logging
from typing import Optional, List, Generator

import torch
from torch import nn
from transformers import Qwen2ForCausalLM
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from transformers.modeling_outputs import ModelOutput
from transformers.utils import auto_docstring

from mindspeed_mm.fsdp.models.cosyvoice3.common import IGNORE_ID, th_accuracy, make_pad_mask, ras_sampling
from mindspeed_mm.fsdp.models.base_model import BaseModel
from mindspeed_mm.fsdp.utils.device import get_device_type, IS_NPU_AVAILABLE
from mindspeed_mm.fsdp.utils.register import model_register


class Qwen2Encoder(torch.nn.Module):
    def __init__(self, pretrain_path):
        super().__init__()
        self.model = Qwen2ForCausalLM.from_pretrained(pretrain_path)

    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor):
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T)
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=masks,
            output_hidden_states=True,
            return_dict=True,
        )
        return outs.hidden_states[-1], masks.unsqueeze(1)

    def forward_one_step(self, xs, masks, cache=None):
        input_masks = masks[:, -1, :]
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=input_masks,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
            past_key_values=cache,
        )
        xs = outs.hidden_states[-1]
        new_cache = outs.past_key_values
        return xs, new_cache


@model_register.register("cosyvoice3_lm")
class CosyVoice3LM(torch.nn.Module, BaseModel):
    def __init__(self, config):
        super().__init__()
        self.llm_input_size = config.llm_input_size
        self.llm_output_size = config.llm_output_size
        self.speech_token_size = config.speech_token_size
        # 2. build speech token language model related modules
        self.sos = config.speech_token_size + 0
        self.eos_token = config.speech_token_size + 1
        self.task_id = config.speech_token_size + 2
        self.fill_token = config.speech_token_size + 3

        self.llm = Qwen2Encoder(config.llm_encoder)
        self.llm_decoder = nn.Linear(config.llm_output_size, config.speech_token_size + 200, bias=False)
        self.criterion_ce = LabelSmoothingLoss(
            size=config.speech_token_size + 200,
            padding_idx=IGNORE_ID,
            smoothing=config.lsm_weight,
            normalize_length=config.length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(config.speech_token_size + 200, config.llm_input_size)

        # 4. sampling method
        self.sampling = partial(
            ras_sampling,
            top_p=config.sampling.get("top_p", 0.8),
            top_k=config.sampling.get("top_k", 25),
            win_size=config.sampling.get("win_size", 10),
            tau_r=config.sampling.get("tau_r", 0.1),
        )
        self.mix_ratio = config.mix_ratio

        # 5. vllm related
        self.stop_token_ids = [config.speech_token_size + i for i in range(200)]
        self.vllm_output_queue = {}

    def encode(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ):
        encoder_out, encoder_mask = self.text_encoder(text, text_lengths, decoding_chunk_size=1, num_decoding_left_chunks=-1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        encoder_out = self.text_encoder_affine_layer(encoder_out)
        return encoder_out, encoder_out_lens

    def pad_unpad_sequence(self, sos_emb, embedding, text_token, text_token_len, task_id_emb, speech_token, speech_token_len):
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
        lm_input = [torch.concat([sos_emb.squeeze(dim=0), embedding[i], text_token[i], task_id_emb.squeeze(dim=0), speech_token[i]], dim=0)
                    for i in range(len(text_token))]
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
        return lm_input, lm_input_len

    def sampling_ids(
        self,
        weighted_scores: torch.Tensor,
        decoded_tokens: List,
        sampling: int,
        ignore_eos: bool = True,
    ):
        num_trials, max_trials = 0, 100
        while True:
            top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
            if (not ignore_eos) or (top_ids < self.speech_token_size):
                break
            num_trials += 1
            if num_trials > max_trials:
                raise RuntimeError('sampling reaches max_trials {} and still get eos when ignore_eos is True, check your input!'.format(max_trials))
        return top_ids

    def prepare_lm_input_target(
        self,
        sos_emb,
        text_token,
        text_token_emb,
        text_token_len,
        task_id_emb,
        speech_token,
        speech_token_emb,
        speech_token_len,
        instruct_token=None,
        instruct_token_emb=None,
        instruct_token_len=None
        ):
        lm_target, lm_input = [], []
        text_token = unpad_sequence(text_token, text_token_len, batch_first=True)
        speech_token = unpad_sequence(speech_token, speech_token_len, batch_first=True)
        text_token_emb = unpad_sequence(text_token_emb, text_token_len, batch_first=True)
        speech_token_emb = unpad_sequence(speech_token_emb, speech_token_len, batch_first=True)
        # NOTE add instruct_token in CosyVoice3
        if instruct_token is not None and instruct_token_emb is not None and instruct_token_len is not None:
            instruct_token = unpad_sequence(instruct_token, instruct_token_len, batch_first=True)
            instruct_token_emb = unpad_sequence(instruct_token_emb, instruct_token_len, batch_first=True)
        else:
            instruct_token = [torch.empty(0).to(text_token[0])] * len(text_token)
            instruct_token_emb = [torch.empty(0, 896).to(text_token_emb[0])] * len(text_token)
            instruct_token_len = torch.zeros(len(text_token)).to(text_token_len)
        for i in range(len(text_token)):
            # bistream sequence
            if random.random() < 0.5 and speech_token_len[i] / text_token_len[i] > self.mix_ratio[1] / self.mix_ratio[0]:
                this_lm_target, this_lm_input = [IGNORE_ID], [sos_emb.squeeze(dim=0)]
                this_lm_target += [IGNORE_ID] * instruct_token_len[i]
                this_lm_input.append(instruct_token_emb[i])
                for j in range(((text_token_len[i] + 1) / self.mix_ratio[0]).ceil().int().item()):
                    this_text_token = text_token[i][j * self.mix_ratio[0]: (j + 1) * self.mix_ratio[0]].tolist()
                    this_speech_token = speech_token[i][j * self.mix_ratio[1]: (j + 1) * self.mix_ratio[1]].tolist()
                    if len(this_text_token) == self.mix_ratio[0]:
                        this_lm_target += [IGNORE_ID] * (self.mix_ratio[0] - 1)
                        this_lm_target += this_speech_token
                        this_lm_target.append(self.fill_token)
                        this_lm_input.append(text_token_emb[i][j * self.mix_ratio[0]: (j + 1) * self.mix_ratio[0]])
                        this_lm_input.append(speech_token_emb[i][j * self.mix_ratio[1]: (j + 1) * self.mix_ratio[1]])
                    else:
                        this_lm_target += [-1] * len(this_text_token)
                        this_lm_target += speech_token[i][j * self.mix_ratio[1]:].tolist()
                        this_lm_target.append(self.eos_token)
                        this_lm_input.append(text_token_emb[i][j * self.mix_ratio[0]:])
                        this_lm_input.append(task_id_emb.squeeze(dim=0))
                        this_lm_input.append(speech_token_emb[i][j * self.mix_ratio[1]:])
                this_lm_target, this_lm_input = torch.tensor(this_lm_target), torch.concat(this_lm_input, dim=0)
            # unistream sequence
            else:
                this_lm_target = torch.tensor([IGNORE_ID] * (1 + instruct_token_len[i] + text_token_len[i]) + speech_token[i].tolist() + [self.eos_token])
                this_lm_input = torch.concat([sos_emb.squeeze(dim=0), instruct_token_emb[i], text_token_emb[i], task_id_emb.squeeze(dim=0), speech_token_emb[i]], dim=0)
            lm_target.append(this_lm_target)
            lm_input.append(this_lm_input)
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
        lm_target = pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID)
        return lm_target, lm_input, lm_input_len

    def forward(
        self,
        text_token,
        text_token_len,
        speech_token,
        speech_token_len,
        instruct_token,
        instruct_token_len,
        **kwargs,
    ):
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        # 1. encode text_token
        text_token_emb = self.llm.model.model.embed_tokens(text_token)
        instruct_token_emb = self.llm.model.model.embed_tokens(instruct_token)

        # 3. sos and task_id
        sos_emb = self.speech_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.speech_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 2. encode speech_token
        speech_token_emb = self.speech_embedding(speech_token)

        # 3. prepare llm_input/target
        lm_target, lm_input, lm_input_len = self.prepare_lm_input_target(sos_emb, text_token, text_token_emb, text_token_len, task_id_emb,
                                                                         speech_token, speech_token_emb, speech_token_len, instruct_token, instruct_token_emb, instruct_token_len)
        lm_target = lm_target.to(device=get_device_type())

        # 4. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len)
        logits = self.llm_decoder(lm_output)
        loss = self.criterion_ce(logits, lm_target)
        acc = th_accuracy(logits.view(-1, self.speech_token_size + 200), lm_target, ignore_label=IGNORE_ID)
        return CosyVoice3LMOutputWithPast(
            loss=loss,
            acc=acc,
            logits=logits,
            last_hidden_states=lm_output,
        )

    @torch.inference_mode()
    def inference_wrapper(self, lm_input, sampling, min_len, max_len, uuid):
        out_tokens = []
        cache = None
        for i in range(max_len):
            y_pred, cache = self.llm.forward_one_step(lm_input,
                                                        masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
                                                        cache=cache)
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False)
            if top_ids in self.stop_token_ids:
                break
            # in stream mode, yield token one by one
            yield top_ids
            out_tokens.append(top_ids)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)

    @torch.inference_mode()
    def inference_bistream(
        self,
        text: Generator,
        prompt_text: torch.Tensor,
        prompt_text_len: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        prompt_speech_token_len: torch.Tensor,
        embedding: torch.Tensor,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:

        device = prompt_text.device
        # 1. prepare input
        sos_emb = self.llm_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=prompt_text.dtype).to(device)
        lm_input = torch.concat([sos_emb], dim=1)

        # 2. iterate text
        out_tokens = []
        cache = None
        # NOTE init prompt_text as text_cache as it is basically impossible prompt_speech_token/prompt_text < 15/5
        text_cache = self.llm.model.model.embed_tokens(prompt_text)
        next_fill_index = (int(prompt_speech_token.shape[1] / self.mix_ratio[1]) + 1) * self.mix_ratio[1] - prompt_speech_token.shape[1]
        for this_text in text:
            text_cache = torch.concat([text_cache, self.llm.model.model.embed_tokens(this_text)], dim=1)
            # prompt_speech_token_emb not empty, try append to lm_input
            while prompt_speech_token_emb.size(1) != 0:
                if text_cache.size(1) >= self.mix_ratio[0]:
                    lm_input_text, lm_input_speech = text_cache[:, :self.mix_ratio[0]], prompt_speech_token_emb[:, :self.mix_ratio[1]]
                    logging.info('append {} text token {} speech token'.format(lm_input_text.size(1), lm_input_speech.size(1)))
                    lm_input = torch.concat([lm_input, lm_input_text, lm_input_speech], dim=1)
                    text_cache, prompt_speech_token_emb = text_cache[:, self.mix_ratio[0]:], prompt_speech_token_emb[:, self.mix_ratio[1]:]
                else:
                    logging.info('not enough text token to decode, wait for more')
                    break
            # no prompt_speech_token_emb remain, can decode some speech token
            if prompt_speech_token_emb.size(1) == 0:
                if (len(out_tokens) != 0 and out_tokens[-1] == self.fill_token) or (len(out_tokens) == 0 and lm_input.size(1) == 1):
                    logging.info('get fill token, need to append more text token')
                    if text_cache.size(1) >= self.mix_ratio[0]:
                        lm_input_text = text_cache[:, :self.mix_ratio[0]]
                        logging.info('append {} text token'.format(lm_input_text.size(1)))
                        if len(out_tokens) != 0 and out_tokens[-1] == self.fill_token:
                            lm_input = lm_input_text
                        else:
                            lm_input = torch.concat([lm_input, lm_input_text], dim=1)
                        text_cache = text_cache[:, self.mix_ratio[0]:]
                    else:
                        logging.info('not enough text token to decode, wait for more')
                        continue
                while True:
                    seq_len = lm_input.shape[1] if cache is None else lm_input.shape[1] + cache[0][0].size(2)
                    y_pred, cache = self.llm.forward_one_step(lm_input,
                                                              masks=torch.tril(torch.ones((1, seq_len, seq_len), device=lm_input.device)).to(torch.bool),
                                                              cache=cache)
                    logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
                    if next_fill_index != -1 and len(out_tokens) == next_fill_index:
                        top_ids = self.fill_token
                        next_fill_index += (self.mix_ratio[1] + 1)
                    else:
                        top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True)
                    if top_ids == self.fill_token:
                        next_fill_index = len(out_tokens) + self.mix_ratio[1] + 1
                        logging.info('fill_token index {} next fill_token index {}'.format(len(out_tokens), next_fill_index))
                    out_tokens.append(top_ids)
                    if top_ids >= self.speech_token_size:
                        if top_ids == self.fill_token:
                            break
                        else:
                            raise ValueError('should not get token {}'.format(top_ids))
                    yield top_ids
                    lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)

        # 3. final decode
        lm_input = torch.concat([lm_input, text_cache, task_id_emb], dim=1)
        logging.info('no more text token, decode until met eos')
        while True:
            seq_len = lm_input.shape[1] if cache is None else lm_input.shape[1] + cache[0][0].size(2)
            y_pred, cache = self.llm.forward_one_step(lm_input,
                                                      masks=torch.tril(torch.ones((1, seq_len, seq_len), device=lm_input.device)).to(torch.bool),
                                                      cache=cache)
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=False)
            out_tokens.append(top_ids)
            if top_ids >= self.speech_token_size:
                if top_ids == self.eos_token:
                    break
                else:
                    raise ValueError('should not get token {}'.format(top_ids))
            # in stream mode, yield token one by one
            yield top_ids
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)

    @torch.inference_mode()
    def inference(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        prompt_text: torch.Tensor,
        prompt_text_len: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        prompt_speech_token_len: torch.Tensor,
        embedding: torch.Tensor,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
        uuid: str = '',
    ) -> Generator[torch.Tensor, None, None]:
        device = text.device
        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        text = self.llm.model.model.embed_tokens(text)

        # 3. concat llm_input
        sos_emb = self.speech_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.speech_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
        lm_input = torch.concat([sos_emb, text, task_id_emb, prompt_speech_token_emb], dim=1)

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. step by step decode
        for token in self.inference_wrapper(lm_input, sampling, min_len, max_len, uuid):
            yield token

    @classmethod
    def from_pretrained(cls, config):
        model = cls._from_config(config)
        if getattr(config, 'model_name_or_path') and os.path.exists(config.model_name_or_path):
            state_dict = torch.load(config.model_name_or_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        else:
            print("Invalid path or missing checkpoint attribute in config. will initial weights from random.")

        return model

    @classmethod
    def _from_config(cls, config):
        model = cls(config)
        return model


class LabelSmoothingLoss(nn.Module):
    """Label-smoothing loss.

    In a standard CE loss, the label's data distribution is:
    [0,1,2] ->
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    In the smoothing version CE Loss,some probabilities
    are taken from the true label prob (1.0) and are divided
    among other labels.

    e.g.
    smoothing=0.1
    [0,1,2] ->
    [
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
    ]

    Args:
        size (int): the number of class
        padding_idx (int): padding class id which will be ignored for loss
        smoothing (float): smoothing rate (0.0 means the conventional CE)
        normalize_length (bool):
            normalize loss by sequence length if True
            normalize loss by batch size if False
    """

    def __init__(
        self,
        size: int,
        padding_idx: int,
        smoothing: float,
        normalize_length: bool = False
    ):
        """Construct an LabelSmoothingLoss object."""
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.normalize_length = normalize_length

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss between x and target.

        The model outputs and data labels tensors are flatten to
        (batch*seqlen, class) shape and a mask is applied to the
        padding part which should not be calculated for loss.

        Args:
            x (torch.Tensor): prediction (batch, seqlen, class)
            target (torch.Tensor):
                target signal masked with self.padding_id (batch, seqlen)
        Returns:
            loss (torch.Tensor) : The KL loss, scalar float value
        """
        batch_size = x.size(0)
        x = x.view(-1, self.size)
        target = target.view(-1)
        # use zeros_like instead of torch.no_grad() for true_dist,
        # since no_grad() can not be exported by JIT
        true_dist = torch.zeros_like(x)
        true_dist.fill_(self.smoothing / (self.size - 1))
        ignore = target == self.padding_idx  # (B,)
        total = len(target) - ignore.sum().item()
        target = target.masked_fill(ignore, 0)  # avoid -1 index
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)
        denom = total if self.normalize_length else batch_size
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for CosyVoice3 language model outputs.
    """
)
class CosyVoice3LMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    acc
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    """

    loss: Optional[torch.FloatTensor] = None
    acc: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    last_hidden_states: Optional[torch.FloatTensor] = None


if IS_NPU_AVAILABLE:
    from mindspeed_mm.fsdp.models.cosyvoice3.npu_patch import apply_cosyvoice_npu_patch

    apply_cosyvoice_npu_patch()
