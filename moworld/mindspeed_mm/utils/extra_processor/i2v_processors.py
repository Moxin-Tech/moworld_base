from .cogvideox_i2v_processor import CogVideoXI2VProcessor
from .hunyuanvideo_i2v_processor import HunyuanVideoI2VProcessor
from .opensoraplan_i2v_processor import OpenSoraPlanI2VProcessor
from .wan_i2v_processor import WanVideoI2VProcessor
from .wan_ti2v_processor import WanVideoTI2VProcessor
from .stepvideo_i2v_processor import StepVideoI2VProcessor
from .hunyuanvideo_15_i2v_processor import HunyuanVideo15I2VProcessor

I2V_PROCESSOR_MAPPINGS = {
    "cogvideox_i2v_processor": CogVideoXI2VProcessor,
    "opensoraplan_i2v_processor": OpenSoraPlanI2VProcessor,
    "wan_i2v_processor": WanVideoI2VProcessor,
    "wan_ti2v_processor": WanVideoTI2VProcessor,
    "hunyuanvideo_i2v_processor": HunyuanVideoI2VProcessor,
    "stepvideo_i2v_processor": StepVideoI2VProcessor,
    "hunyuanvideo_15_i2v_processor": HunyuanVideo15I2VProcessor
}


class I2VProcessor:
    """
    The extra processor of the image to video task
    I2VProcessor is the factory class for all i2v_processor

    Args:
        config (dict): for Instantiating an atomic methods
    """

    def __init__(self, config):
        super().__init__()
        i2v_processor_cls = I2V_PROCESSOR_MAPPINGS[config["processor_id"]]
        self.processor = i2v_processor_cls(config)

    def get_processor(self):
        return self.processor
