# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
import abc


class HyperModelABC(abc.ABC):
    """
    The hyper model wraps multiple models required in reinforcement learning into a single model,
    maintaining the original distributed perspective unchanged.

    In different reinforcement learning scenarios, different numbers and states of models are required,
    and the following methods need to be implemented as needed.
    For example, in DPO, one trainable model and one frozen reference model are needed,
    while in PPO, three trainable models and one inference model for reward are required.
    """
    def __init__(self, *args, **kwargs):
        """
        You need to initialize each of the models and states.
        """

    def __call__(self, *args, **kwargs):
        """
        Implementing the forward process of multiple models requires concatenating and chunking the inputs of
        multiple models along the batch dimension, paying attention to the perspective handling under PP parallelism.
        """

    def set_input_tensor(self):
        """
        In pipeline parallelism, you need to pass the output from the previous stage as input to the next stage.
        This requires a custom implementation based on the specific scenario.

        For example, in DPO, we use the get_attr_from_wrapped_model method to retrieve the original training model's
        set_input_tensor input [s, 4 * b, h] (which has already been replaced with the concatenated inputs of the
        training model and reference model, i.e., chosen and rejected of the training model and chosen and rejected of
        the reference model), and then separate them into the required inputs for each model
        and perform the actual set_input_tensor operation. In scenarios where pipeline parallelism is not used,
        such as tensor parallelism, the result of get_attr_from_wrapped_model is None,
        and this logic does not need to be executed.
        """
