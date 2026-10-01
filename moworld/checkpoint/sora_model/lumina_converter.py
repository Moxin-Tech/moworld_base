from checkpoint.sora_model.sora_model_converter import SoraModelConverter


class LuminaConverter(SoraModelConverter):
    """Converter for lumina"""

    _supported_methods = ["hf_to_mm", "mm_to_hf"]
