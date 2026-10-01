import importlib
from mindspeed_mm.models.transformers.custom_model_registry import get_model_class_global


class ModelHub:
    @staticmethod
    def build(config, transformer_config):
        """
        Constructs and returns the corresponding model class based on the provided configuration.

        This method determines the model class in the following priority order:
        1. First checks the `model_id` field in the `config` object (typically from the "model_id"
        field in model.json). If present, it looks up the corresponding model class in
        ModelHub.MODEL_MAPPINGS.
        2. If `model_id` is not available, it attempts to use the `architectures` field from
        `transformer_config` (usually from the config.json file in the Hugging Face model
        directory) to dynamically load the model class from the transformers library.
        3. If `architectures` is not available, it falls back to the `model_type` field in
        `transformer_config`, and looks up the corresponding model class in
        ModelHub.MODEL_MAPPINGS.

        Args:
            config (object): Configuration object expected to contain a `model_id` attribute.
            transformer_config (transformers.PretrainedConfig): Model configuration object
                from the transformers library, containing fields such as `architectures`
                and `model_type`.

        Returns:
            type: The corresponding model class if found.

        Raises:
            ValueError: If no model class can be determined after all lookup attempts.

        Example:
            model_class = ModelBuilder.build(config, transformer_config)
            model = model_class()
        """
        architectures = getattr(transformer_config, "architectures", [])
        model_type = getattr(transformer_config, "model_type", None)

        model_cls = None

        model_id = getattr(config, "model_id", None)
        if model_id:
            model_cls = get_model_class_global(model_id)
            return model_cls

        if architectures:
            transformers_module = importlib.import_module("transformers")
            model_cls = getattr(transformers_module, architectures[0], None)
            if model_cls is not None:
                return model_cls

        if model_type:
            model_cls = get_model_class_global(model_type)
            if model_cls is not None:
                return model_cls

        if model_cls is None:
            raise ValueError("load model from config failed")
        return model_cls
