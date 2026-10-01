from torch import nn
from megatron.training.utils import print_rank_0
from megatron.core import mpu

from mindspeed_mm.models.common.checkpoint import load_checkpoint
from .dits import (
    Latte,
    VideoDitSparse,
    SatDiT,
    VideoDitSparseI2V,
    PTDiT,
    HunyuanVideoDiT,
    HunyuanVideo15DiT,
    WanDiT,
    StepVideoDiT,
    SparseUMMDiT,
    MMDiT,
    VACEModel
)


PREDICTOR_MODEL_MAPPINGS = {
    "videoditsparse": VideoDitSparse,
    "videoditsparsei2v": VideoDitSparseI2V,
    "latte": Latte,
    "satdit": SatDiT,
    "ptdit": PTDiT,
    "hunyuanvideodit": HunyuanVideoDiT,
    "hunyuanvideo15dit": HunyuanVideo15DiT,
    "wandit": WanDiT,
    "stepvideodit": StepVideoDiT,
    "SparseUMMDiT": SparseUMMDiT,
    "mmdit": MMDiT,
    "vace": VACEModel
}


class PredictModel(nn.Module):
    """
    The backnone of the denoising model
    PredictModel is the factory class for all unets and dits

    Args:
        config[dict]: for Instantiating an atomic methods
    """

    def __init__(self, config):
        super().__init__()
        model_cls = PREDICTOR_MODEL_MAPPINGS[config.model_id]
        config = self._build_predictor_layers_config(config)
        self.predictor = model_cls(**config.to_dict())
        if hasattr(config, "from_pretrained") and config.from_pretrained is not None:
            assign = False
            if hasattr(self.predictor, "device") and self.predictor.device.type == "meta":
                assign = True
            load_checkpoint(self.predictor, config.from_pretrained, assign=assign)
            print_rank_0("load predictor's checkpoint sucessfully")
        if hasattr(self.predictor, "post_init"):
            self.predictor.post_init()

    def get_model(self):
        return self.predictor

    def _build_predictor_layers_config(self, config):
        if mpu.get_pipeline_model_parallel_world_size() <= 1:
            return config

        self.pp_size = mpu.get_pipeline_model_parallel_world_size()
        self.enable_vpp = mpu.get_virtual_pipeline_model_parallel_world_size() is not None
        if self.enable_vpp:
            self.vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
            self.vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
        self.pp_rank = mpu.get_pipeline_model_parallel_rank()
        print(f"current pp_size: {self.pp_size}, pp_rank: {self.pp_rank}",
              f"vpp_size:{self.vpp_size}, vpp_rank: {self.vpp_rank}" if self.enable_vpp else None)

        if not hasattr(config, "pipeline_num_layers"):
            raise ValueError(f"The `pipeline_num_layers` must be specified in the config for pipeline parallel")
        if mpu.is_pipeline_first_stage():
            if self.enable_vpp:
                if sum(sum(pipeline_num_layer) for pipeline_num_layer in config.pipeline_num_layers) != config.num_layers:
                    raise ValueError(f"The sum of `pipeline_num_layers` must be equal to the `num_layers`")
            else:
                if sum(config.pipeline_num_layers) != config.num_layers:
                    raise ValueError(f"The sum of `pipeline_num_layers` must be equal to the `num_layers`")

        if self.enable_vpp:
            if self.vpp_size != len(config.pipeline_num_layers):
                raise ValueError(f"The vp_size {self.vpp_size} must be equal to the number of layers of "
                                 f"pipeline_num_layers {len(config.pipeline_num_layers)}.")
            for vp_rank in range(self.vpp_size):
                if self.pp_size != len(config.pipeline_num_layers[vp_rank]):
                    raise ValueError(f"The pp_size {self.pp_size} must be equal to the number of stages of "
                                     f"pipeline_num_layers {len(config.pipeline_num_layers[vp_rank])}.")
            if self.vpp_size * self.pp_size != len(config.pipeline_num_layers) * len(config.pipeline_num_layers[0]):
                raise ValueError(f"The product of vpp_size and pp_size must be equal to the num of stages in "
                                 f"pipeline_num_layers of predictor config, "
                                 f"but got vpp_size: {self.vpp_size}, pp_size: {self.pp_size}, "
                                 f"and total num of stages is "
                                 f"{len(config.pipeline_num_layers) * len(config.pipeline_num_layers[0])}")
        else:
            if self.pp_size != len(config.pipeline_num_layers):
                raise ValueError(f"The pp_size should be qual to the num of predictor pipeline layers: "
                                 f"{len(config.pipeline_num_layers)}")

        if self.enable_vpp:
            pipeline_start_idx = (sum(sum(config.pipeline_num_layers[i]) for i in range(self.vpp_rank)) +
                                  sum(config.pipeline_num_layers[self.vpp_rank][:self.pp_rank]))
            pipeline_end_idx = pipeline_start_idx + config.pipeline_num_layers[self.vpp_rank][self.pp_rank]
            local_num_layers = config.pipeline_num_layers[self.vpp_rank][self.pp_rank]
        else:
            pipeline_start_idx = sum(config.pipeline_num_layers[:self.pp_rank])
            pipeline_end_idx = sum(config.pipeline_num_layers[:self.pp_rank + 1])
            local_num_layers = config.pipeline_num_layers[self.pp_rank]
        if local_num_layers <= 0:
            raise ValueError(f"for pp_rank {self.pp_rank}, the predictor layer is {local_num_layers}, "
                             f"which is invalid. ")

        config.num_layers = local_num_layers
        config.pre_process = mpu.is_pipeline_first_stage()
        config.post_process = mpu.is_pipeline_last_stage()
        config.global_layer_idx = tuple(range(pipeline_start_idx, pipeline_end_idx))

        return config
