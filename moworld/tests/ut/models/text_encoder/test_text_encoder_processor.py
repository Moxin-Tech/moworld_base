from dataclasses import dataclass
from typing import Dict, AnyStr, Type
import mindspeed.megatron_adaptor
import transformers

from mindspeed_mm.models.text_encoder import TextEncoder, Tokenizer
from mindspeed_mm.models.text_encoder.hunyuan_mllm_text_encoder import HunyuanMLLmModel
from tests.ut.utils import TestConfig, judge_expression


@dataclass
class TestModelInput:
    tokenizer_dict: Dict
    text_encoder_dict: Dict
    text: AnyStr
    target_model_type: Type
    target_output_min: float
    target_att_sum: int


class TestTestEncoder:
    _execution_device = "npu"

    def _init_model(self, tokenizer_dict, text_encoder_dict):
        """
        init tokenizer and text encoder
        """
        tokenizer_config = TestConfig(tokenizer_dict)
        tokenizer = Tokenizer(tokenizer_config).get_tokenizer()

        text_encoder_config = TestConfig(text_encoder_dict)
        text_encoder = TextEncoder(text_encoder_config).to(self._execution_device)
        return tokenizer, text_encoder

    @staticmethod
    def _run_text_encoder(text, tokenizer, text_encoder):
        tokenizer_output = tokenizer(text, return_tensors='pt')
        output, att_mask = text_encoder.encode(input_ids=tokenizer_output["input_ids"], mask=tokenizer_output["attention_mask"])
        return output, att_mask

    @staticmethod
    def _judge_result(output, att_mask, target_output_min, target_att_sum):
        judge = (output.min().item() == target_output_min and att_mask.sum().item() == target_att_sum)
        judge_expression(judge)

    @staticmethod
    def _judge_model(model, target_model_type):
        judge_expression(isinstance(model.get_model(), target_model_type))

    def _test_model(self, test_model_input: TestModelInput):
        tokenizer, text_encoder = self._init_model(test_model_input.tokenizer_dict, test_model_input.text_encoder_dict)
        self._judge_model(text_encoder, test_model_input.target_model_type)
        output, attn_mask = self._run_text_encoder(test_model_input.text, tokenizer, text_encoder)
        self._judge_result(output, attn_mask, test_model_input.target_output_min, test_model_input.target_att_sum)

    def test_t5(self):
        """
        test t5 text encoder processor
        """
        text_encoder_dict = {
            "hub_backend": "hf",
            "model_id": "T5",
            "dtype": "bf16",
            "from_pretrained": "/home/ci_resource/models/text_encoder_mini/t5_mini",
        }
        tokenizer_dict = {
            "hub_backend": "hf",
            "autotokenizer_name": "AutoTokenizer",
            "from_pretrained": "/home/ci_resource/models/text_encoder_mini/t5_mini",
        }

        test_input = TestModelInput(
            tokenizer_dict=tokenizer_dict,
            text_encoder_dict=text_encoder_dict,
            text="This is a T5 example",
            target_model_type=transformers.models.t5.modeling_t5.T5EncoderModel,
            target_output_min=-2.078125,
            target_att_sum=8,
        )
        self._test_model(test_input)

    def test_mt5(self):
        """
        test mt5 text encoder processor
        """
        text_encoder_dict = {
                "hub_backend": "hf",
                "model_id": "MT5",
                "dtype": "bf16",
                "from_pretrained": "/home/ci_resource/models/text_encoder_mini/mt5_mini",
        }
        tokenizer_dict = {
                "hub_backend": "hf",
                "autotokenizer_name": "AutoTokenizer",
                "from_pretrained": "/home/ci_resource/models/text_encoder_mini/mt5_mini",
        }

        test_input = TestModelInput(
            tokenizer_dict=tokenizer_dict,
            text_encoder_dict=text_encoder_dict,
            text="This is a mt5 example",
            target_model_type=transformers.models.mt5.modeling_mt5.MT5EncoderModel,
            target_output_min=-3.078125,
            target_att_sum=9,
        )
        self._test_model(test_input)

    def test_umt5(self):
        """
        test umt5 text encoder processor
        """
        text_encoder_dict = {
                "hub_backend": "hf",
                "model_id": "UMT5",
                "dtype": "bf16",
                "from_pretrained": "/home/ci_resource/models/text_encoder_mini/umt5_mini",
        }
        tokenizer_dict = {
                "hub_backend": "hf",
                "autotokenizer_name": "AutoTokenizer",
                "from_pretrained": "/home/ci_resource/models/text_encoder_mini/umt5_mini",
        }

        test_input = TestModelInput(
            tokenizer_dict=tokenizer_dict,
            text_encoder_dict=text_encoder_dict,
            text="This is a umt5 example",
            target_model_type=transformers.models.umt5.modeling_umt5.UMT5EncoderModel,
            target_output_min=-1.1640625,
            target_att_sum=8,
        )
        self._test_model(test_input)

    def test_clip(self):
        """
        test clip text encoder processor
        """
        text_encoder_dict = {
            "hub_backend": "hf",
            "model_id": "CLIP",
            "dtype": "float32",
            "from_pretrained": "/home/ci_resource/models/stable-diffusion-xl-base-1.0",
            "subfolder": "text_encoder"
        }
        tokenizer_dict = {
            "hub_backend": "hf",
            "autotokenizer_name": "AutoTokenizer",
            "from_pretrained": "/home/ci_resource/models/stable-diffusion-xl-base-1.0",
            "subfolder": "tokenizer"
        }

        test_input = TestModelInput(
            tokenizer_dict=tokenizer_dict,
            text_encoder_dict=text_encoder_dict,
            text="This is a CLIP example",
            target_model_type=transformers.models.clip.modeling_clip.CLIPTextModel,
            target_output_min=-28.09918212890625,
            target_att_sum=7,
        )
        self._test_model(test_input)

    def test_hunyuanMLLm(self):
        """
        test hunyuanMLLm text encoder, a custom text encoder
        """
        text_encoder_dict = {
            "model_id": "HunyuanMLLmModel",
            "dtype": "fp16",
            "from_pretrained": "/home/ci_resource/models/text_encoder_mini/llava-llama-3-8b-text-encoder-tokenizer_mini_trans_453",
            "hub_backend": "hf",
            "use_attention_mask": True,
            "hidden_state_skip_layer": 2,
            "output_key": "hidden_states",
            "template_id": "hyv-llm-encode-video",
            "template_file_path": "/home/ci_resource/models/hunyuanvideo_t2v/template.json"
        }

        tokenizer_dict = {
            "autotokenizer_name": "hunyuanMLLmTokenizer",
            "hub_backend": "hf",
            "from_pretrained": "/home/ci_resource/models/text_encoder_mini/llava-llama-3-8b-text-encoder-tokenizer_mini_trans_453",
            "model_max_length": 256,
            "template_id": "hyv-llm-encode-video",
            "template_file_path": "/home/ci_resource/models/hunyuanvideo_t2v/template.json"
        }

        test_input = TestModelInput(
            tokenizer_dict=tokenizer_dict,
            text_encoder_dict=text_encoder_dict,
            text="This is a hunyuanMLLm example",
            target_model_type=HunyuanMLLmModel,
            target_output_min=-0.489501953125,
            target_att_sum=11,
        )
        self._test_model(test_input)
