import mindspore
import msadapter


def process_in_cpu_wrapper(func):
    """
    By default, MindSpore do everything on npu. Here manually set cpu when processing dataset
    """
    def wrapper(*args, **kwargs):
        # set device to CPU
        from mindspore.common.api import _pynative_executor
        _pynative_executor.sync()
        msadapter.configs.set_pyboost(False)
        mindspore.set_context(device_target="CPU")

        # process dataset
        result = func(*args, **kwargs)

        # set device to Ascend
        msadapter.configs.set_pyboost(True)
        _pynative_executor.sync()
        mindspore.set_context(device_target="Ascend")

        return result
    return wrapper
