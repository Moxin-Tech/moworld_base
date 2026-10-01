from enum import Enum
from mindspeed_mm.fsdp.utils.decorators import Singleton


class TrainingStage(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"


class TrainingContext(metaclass=Singleton):
    _training_stage: TrainingStage = TrainingStage.FORWARD
    _layer_index: int = 0
    _model_depth: int = 0

    def set_training_stage(self, training_stage) -> None:
        self._training_stage = training_stage

    def get_training_stage(self) -> TrainingStage:
        return self._training_stage

    def set_layer_index(self, layer_index: int) -> None:
        self._layer_index = layer_index

    def set_model_depth(self, model_depth: int) -> None:
        self._model_depth = model_depth

    def get_layer_index(self) -> int:
        return self._layer_index

    def get_model_depth(self) -> int:
        return self._model_depth
