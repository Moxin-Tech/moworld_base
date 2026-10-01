# coding=utf-8
"""
Implementation of FunASR model for FSDP training framework
Using the actual FunASR nano model implementation
"""

import logging
import os

import yaml
from funasr import AutoModel
from mindspeed.fsdp.utils.log import print_rank
from mindspeed_mm.fsdp.utils.device import IS_NPU_AVAILABLE, get_device_type
from mindspeed_mm.fsdp.utils.register import model_register

logger = logging.getLogger(__name__)


def get_funasr_model(model_args, model_parallel_applier):
    model_dir = model_args.model_name_or_path
    config_path = os.path.join(model_dir, "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.yaml not found in {model_dir}")

    with open(config_path) as f:
        model_config = yaml.safe_load(f)

    model_type = model_config.get("model")
    if not model_type:
        raise ValueError(f"'model' field missing in {config_path}")

    kwargs = {
        "model": model_type,
        "model_path": model_dir,
        "trust_remote_code": model_args.trust_remote_code,
        "device": get_device_type(),
    }
    model_instance = AutoModel(**kwargs)
    model = model_instance.model

    tokenizer = model_instance.kwargs.get("tokenizer")
    frontend = model_instance.kwargs.get("frontend")
    if tokenizer is None:
        raise ValueError("Tokenizer not found in model instance. Ensure config.yaml has 'tokenizer' field.")
    if frontend is None:
        logging.warning("Frontend not found in model instance. Audio features may not be extracted correctly.")

    if model_args.audio_encoder_conf["freeze"]:
        for param in model.audio_encoder.parameters():
            param.requires_grad = False
        model.audio_encoder.eval()
    else:
        for param in model.audio_encoder.parameters():
            param.requires_grad = True

    if model_args.llm_conf["freeze"]:
        for param in model.llm.parameters():
            param.requires_grad = False
        model.llm.eval()
    else:
        for param in model.llm.parameters():
            param.requires_grad = True

    if model_args.audio_adaptor_conf["freeze"]:
        for param in model.audio_adaptor.parameters():
            param.requires_grad = False
        model.audio_adaptor.eval()
    else:
        for param in model.audio_adaptor.parameters():
            param.requires_grad = True

    if model_args.ctc_decoder_conf["freeze"]:
        for param in model.ctc_decoder.parameters():
            param.requires_grad = False
        model.ctc_decoder.eval()
    else:
        for param in model.ctc_decoder.parameters():
            param.requires_grad = True

    model = model_parallel_applier(model)
    for i, (name, param) in enumerate(model.named_parameters()):
        print_rank(logger.info, f"  Param: {name}, dtype={param.dtype}, device={param.device}, requires_grad={param.requires_grad}, shape: {param.shape}, numel: {param.numel()}, sum={param.sum().item():.6f}, mean={param.mean().item():.6f}")

    if IS_NPU_AVAILABLE:
        from mindspeed_mm.fsdp.models.funasr.npu_patch import apply_funasr_npu_patch

        apply_funasr_npu_patch()

    return model, tokenizer, frontend

model_register.register("funasr")(get_funasr_model)
