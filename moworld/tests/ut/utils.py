"""
We can't use assert in our code for codecheck, so create this auxiliary function to wrap
the assert case in ut for ci.
"""


def judge_expression(expression):
    if not expression:
        raise AssertionError


class TestConfig(object):
    def __init__(self, entries):
        for k, v in entries.items():
            if isinstance(v, dict):
                self.__dict__[k] = TestConfig(v)
            else:
                self.__dict__[k] = v

    def to_dict(self):
        ret = {}
        for k, v in self.__dict__.items():
            if isinstance(v, self.__class__):
                ret[k] = v.to_dict()
            else:
                ret[k] = v
        return ret


def initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    pipeline_model_parallel_split_rank=None,
    context_parallel_size=1,
):
    import megatron.core.parallel_state as ps
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
        pipeline_model_parallel_split_rank=pipeline_model_parallel_split_rank,
        context_parallel_size=context_parallel_size,
    )


def clear_module(module_name):
    import sys
    modules_to_delete = [key for key in sys.modules.keys() if key.startswith(module_name)]
    for key in modules_to_delete:
        del sys.modules[key]
