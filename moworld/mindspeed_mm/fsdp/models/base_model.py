from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from mindspeed_mm.fsdp.params.model_args import ModelArguments


class BaseModel(ABC):
    """
    Base Model Abstract Base Class
    All custom models should inherit from this class and implement the required methods.
    """

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        config: ModelArguments,
    ):
        """
        Load model from pretrained weights.

        Args:
            config: ModelArguments

        Returns:
            Loaded model instance
        """
        pass

    @classmethod
    @abstractmethod
    def _from_config(cls, config: ModelArguments) -> "BaseModel":
        """
        Create model instance from configuration without loading pretrained weights.
        Typically used for initialization with meta device or when starting from scratch.

        Args:
            config: ModelArguments

        Returns:
            Model instance initialized from configuration
        """
        pass


class WeightInitMixin:
    """
    Weight Initialization Mixin Class

    Provides general model weight initialization functionality, supporting multiple layer types
    and composite model structures. Can be used as a mixin class with other torch.nn.Module subclasses.
    """

    def _init_weights(self, module, std=0.02):
        """
        Initialize the weights. This is quite general on purpose, in the spirit of what we usually do. For more complex
        initialization scheme, it should be overridden by the derived `PreTrainedModel` class. In case a model adds an explicit
        `nn.Parameter`, this method should also be overridden in order to initialize it correctly.
        """
        if getattr(module, "_is_initialized", False):
            return

        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding) and module.padding_idx is None:
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.MultiheadAttention):
            # This uses torch's original init
            module._reset_parameters()
        # We cannot use `isinstance` on the RMSNorms or LayerNorms, as they usually are custom modules which change names
        # between modelings (because they are prefixed with the model name)
        elif (
                isinstance(module, (nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
                or "norm" in module.__class__.__name__.lower()
        ):
            # Norms can exist without weights (in which case they are None from torch primitives)
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.fill_(1.0)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
        # 3. Added: Generic parameter scanning and initialization for unhandled module types
        else:
            # Scan all Parameter attributes of the module
            for name, param in module.named_parameters(recurse=False):
                # Only process parameters that directly belong to this module (not recursive to submodules)
                if "weight" in name.lower():
                    param.data.normal_(mean=0.0, std=std)
                elif "bias" in name.lower():
                    param.data.zero_()
                else:
                    # Use default initialization for unknown parameter types
                    param.data.normal_(mean=0.0, std=std)

        module._is_initialized = True

    @torch.no_grad()
    def init_weights(self):
        """
        This is equivalent to calling `self.apply(self._initialize_weights)`, but correctly handles composite models.
        This function dynamically dispatches the correct `init_weights` function to the modules as we advance in the
        module graph along the recursion. It can handle an arbitrary number of sub-models. Without it, every composite
        model would have to recurse a second time on all sub-models explicitly in the outer-most `_init_weights`, which
        is extremely error prone and inefficient.

        Note that the `torch.no_grad()` decorator is very important as well, as most of our `_init_weights` do not use
        `torch.nn.init` functions (which are all no_grad by default), but simply do in-place ops such as
        `module.weight.data.zero_()`.
        """

        # This function is equivalent to `torch.nn.Module.apply`, except that it dynamically adjust the function
        # to apply as we go down the graph
        def smart_apply(self, fn):
            for module in self.children():
                module.smart_apply(fn)
            fn(self)
            return self

        torch.nn.Module.smart_apply = smart_apply

        # Let the magic happen with this simple call
        self.smart_apply(self._init_weights)
