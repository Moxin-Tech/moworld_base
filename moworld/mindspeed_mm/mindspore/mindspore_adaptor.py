from functools import wraps
from packaging import version

import mindspore
import torch
import transformers
from mindspeed.patch_utils import MindSpeedPatchesManager as aspm
from mindspeed.mindspore.ops.npu_rotary_position_embedding import npu_rotary_position_embedding

from mindspeed_mm.mindspore.data.datasets.utils import process_in_cpu_wrapper
from mindspeed_mm.mindspore.data.data_utils.func_utils.convert import preprocess_dataset
from mindspeed_mm.mindspore.models.common.communications import _gather
from mindspeed_mm.mindspore.utils.transformer_model_config import get_model_config
from mindspeed_mm.mindspore.models.predictor.dits.sparseu_mmdit import block_forward, sparsemmditblock_forward


def ms_linear_wrapper(fn):
    @wraps(fn)
    def linear_wrapper(inp, weight, bias=None):
        if {inp.dtype, weight.dtype} == {mindspore.float32, mindspore.bfloat16}:
            return fn(inp.to(mindspore.float32), weight.to(mindspore.float32), bias.to(mindspore.float32)).to(weight.dtype)
        return fn(inp, weight, bias)
    return linear_wrapper


def ms_matmul_wrapper(fn):
    @wraps(fn)
    def matmul_wrapper(inp, other, *args, **kwargs):
        if {inp.dtype, other.dtype} == {mindspore.float32, mindspore.bfloat16}:
            return fn(inp.to(mindspore.float32), other.to(mindspore.float32), *args, **kwargs).to(inp.dtype)
        return fn(inp, other, *args, **kwargs)
    return matmul_wrapper


def ms_scatter_add_wrapper(fn):
    @wraps(fn)
    def scatter_add_wrapper(self, dim, index, src):
        if not index.is_contiguous():
            index = index.contiguous()
        return fn(self, dim, index, src)
    return scatter_add_wrapper


def masked_scatter_(self, mask, updates):
    origin_dtype = None
    if self.dtype in (mindspore.float16, mindspore.bfloat16):
        origin_dtype = self.dtype
        self = self.to(mindspore.float32)
    if updates.dtype in (mindspore.float16, mindspore.bfloat16):
        updates = updates.to(mindspore.float32)
    self = mindspore.ops.MaskedScatter()(self, mask, updates)
    if origin_dtype is not None:
        self = self.to(origin_dtype)
    return self


def apply_mindspore_patch():
    aspm.register_patch('mindspeed_mm.data.datasets.qwen2vl_dataset.get_qwen2vl_dataset', process_in_cpu_wrapper)  # process dataset on cpu
    aspm.register_patch('torch.Tensor.masked_scatter', masked_scatter_)
    aspm.register_patch('mindspeed_mm.data.data_utils.func_utils.convert.SupervisedDatasetProcessor.preprocess_dataset', preprocess_dataset)
    aspm.register_patch('mindspeed_mm.utils.transformer_model_config.get_model_config', get_model_config)
    aspm.register_patch('mindspeed_mm.models.common.communications._gather', _gather)

    # patch llava
    aspm.register_patch(
        'mindspeed.ops.npu_rotary_position_embedding.npu_rotary_position_embedding',
        npu_rotary_position_embedding, force_patch=True
    )

    # patch glm
    if version.parse(transformers.__version__) >= version.parse('4.54.0.dev0'):
        from mindspeed_mm.mindspore.third_party.transformers.masking_utils import sdpa_mask_older_torch
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        ALL_MASK_ATTENTION_FUNCTIONS._global_mapping['sdpa'] = sdpa_mask_older_torch

    # patch opensoraplan1.5t2v
    aspm.register_patch('mindspeed_mm.models.predictor.dits.sparseu_mmdit.SparseUMMDiT.block_forward', block_forward)
    aspm.register_patch('mindspeed_mm.models.predictor.dits.sparseu_mmdit.SparseMMDiTBlock.forward',
                        sparsemmditblock_forward)
    # patch matmul&&linear input requir same stype
    aspm.register_patch('torch.nn.functional.linear', ms_linear_wrapper)
    aspm.register_patch('mindspore.mint.matmul', ms_matmul_wrapper)

    # qwen25 omni hang issue
    from mindspeed_mm.mindspore.data.data_utils.func_utils.mm_plugin import process_messages
    aspm.register_patch('mindspeed_mm.data.data_utils.func_utils.mm_plugin.Qwen2OmniPlugin.process_messages', process_messages)
    from mindspeed_mm.mindspore.third_party.accelerate.state import PartialState_prepare_backend_wrapper, PartialState_set_device
    # Assign a value to os.environ["LOCAL_RANK"] to obtain the rank ID.
    aspm.register_patch('accelerate.state.PartialState._prepare_backend', PartialState_prepare_backend_wrapper)
    # Avoid the "set_device" error.
    aspm.register_patch('accelerate.state.PartialState.set_device', PartialState_set_device)
    # torch.stft not support, use _np_extract_fbank_features
    from mindspeed_mm.mindspore.third_party.transformers.models.whisper.feature_extraction_whisper import _torch_extract_fbank_features_wrapper
    aspm.register_patch('transformers.models.whisper.feature_extraction_whisper.WhisperFeatureExtractor._torch_extract_fbank_features', _torch_extract_fbank_features_wrapper)

    # fix qwen3vl data process issue
    aspm.register_patch('datasets.arrow_dataset.Pool', mindspore.multiprocessing.Pool)
    # fix scatter_add_ contiguous issue for CANN8.5
    aspm.register_patch('mindspore.common.Tensor.scatter_add_', ms_scatter_add_wrapper)

    # opt Qwenvl3 preprocessing performance
    if version.parse(transformers.__version__) >= version.parse('4.57.0.dev0'):
        from mindspeed_mm.mindspore.third_party.transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import _preprocess
        aspm.register_patch('transformers.models.qwen2_vl.image_processing_qwen2_vl_fast.Qwen2VLImageProcessorFast._preprocess', _preprocess)

    # opt image normalizing performance
    from mindspeed_mm.mindspore.third_party.torchvision.transformers.v2.functional._misc import patch_normalize_image
    patch_normalize_image(aspm)

    aspm.apply_patches()

apply_mindspore_patch()
