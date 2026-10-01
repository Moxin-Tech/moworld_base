from functools import wraps
import os
import torch
import torch_npu
from megatron.training.global_vars import set_args, get_args
from megatron.training.tokenizer.tokenizer import build_tokenizer
from megatron.training.utils import print_rank_0
from megatron.training.training import print_datetime

from mindspeed.auto_settings.auto_settings import AutoSettings
from mindspeed.auto_settings.module.parse.profiling_parse import get_settings, get_model_params
from mindspeed.auto_settings.module.parse.profiling_parse.profiling_node_parse import GatherNodeProfiling


POLICY = None
OPTIMIZED_MBS_LIST = None
PP_SCHEDULE_LIST = None
OPTIMAL_LAYERS = None
ORIGIN_MBS = None
DATA_PARALLEL_SIZE = 1
ENABLE_SCHEDULER = False
FLOPS_COUNTER = None
RECORDED_COUNT = 0
TRAVERSED_COUNT = 0
_PROFILER = None


def auto_settings_fun(argument):
    set_args(argument)
    print("pretrain_decorator set_args ========================================")
    argument = get_args()
    working_dir_root = os.path.realpath(argument.auto_settings_work_dir)
    if not os.path.exists(working_dir_root) and argument.rank % torch.cuda.device_count() == 0:
        os.makedirs(working_dir_root)

    if argument.rank % torch.cuda.device_count() == 0:
        print("only rank 0 run auto tuning ========================================")
        settings = AutoSettings()
        settings.auto_setting_fun(argument)
    return


def auto_settings_parse_args():
    args = get_args()
    if not args.vocab_size:
        tokenizer = build_tokenizer(args)
        args.vocab_size = tokenizer.vocab_size
    get_settings(args, args.profile_save_path)
    print_rank_0("================OOTB_OPTIMIZER_PARSE_ARGS END EXIT!====================")
    return


def auto_settings_parse_model(model, mpu, args):
    get_model_params(model, mpu.get_pipeline_model_parallel_rank(), args.profile_save_path, args.context_parallel_size * args.tensor_model_parallel_size * args.data_parallel_size)
    print_rank_0("================OOTB_OPTIMIZER_PARSE_MODEL END EXIT!====================")
    return


def auto_settings_profile(args):
    res_dir = args.profile_save_path
    cur_rank = torch.distributed.get_rank()
    if res_dir and cur_rank % torch.cuda.device_count() == 0:
        GatherNodeProfiling(res_dir).parse_node_pkl(args)
    print_datetime('after training is done')
    return


def train_decorator(step_fn):
    @wraps(step_fn)
    def wrapper(*args, **kwargs):
        args_ = get_args()
        if args_.profile:
            args_.profile_npu = True
            args_.profile = False
        else:
            args_.profile_npu = False

        if judge_if_profile(args_):
            active = args_.profile_step_end - args_.profile_step_start
            skip_first = args_.profile_step_start

            if args_.profile_with_cpu:
                activities = [torch_npu.profiler.ProfilerActivity.NPU, torch_npu.profiler.ProfilerActivity.CPU]
            else:
                activities = [torch_npu.profiler.ProfilerActivity.NPU]

            if args_.profile_level == 'level0':
                profiler_level = torch_npu.profiler.ProfilerLevel.Level0
            elif args_.profile_level == 'level1':
                profiler_level = torch_npu.profiler.ProfilerLevel.Level1
            elif args_.profile_level == 'level2':
                profiler_level = torch_npu.profiler.ProfilerLevel.Level2
            else:
                raise ValueError(f"profiler_level only support level0, level1, level2, but gets {args_.profile_level}")

            experimental_config = torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=profiler_level,
                l2_cache=False
            )

            global _PROFILER
            with torch_npu.profiler.profile(
                activities=activities,
                record_shapes=args_.profile_record_shapes,
                profile_memory=args_.profile_with_memory,
                with_stack=args_.profile_with_stack,
                experimental_config=experimental_config,
                schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=active, repeat=1, skip_first=skip_first),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(args_.profile_save_path)
            ) as prof:
                _PROFILER = prof
                return step_fn(*args, **kwargs)
        else:
            return step_fn(*args, **kwargs)

    return wrapper


def train_step_decorator(step_fn):
    @wraps(step_fn)
    def wrapper(*args, **kwargs):
        args_ = get_args()
        flop_count = None
        if args_.op_cal_tflops:
            flop_count = get_flops_counter()
            flop_count.start()
        ret = step_fn(*args, **kwargs)

        if args_.profile_npu and (torch.distributed.get_rank() in args_.profile_ranks):
            if _PROFILER is not None:
                _PROFILER.step()
        if args_.op_cal_tflops:
            flop_count = get_flops_counter()
            counts = flop_count.get_flops()
            set_count(counts)
            flop_count.stop()
        return ret
    return wrapper


def generated_flops_counter():
    from torch_npu.utils.flops_count import FlopsCounter
    global FLOPS_COUNTER
    FLOPS_COUNTER = FlopsCounter()


def get_flops_counter():
    global FLOPS_COUNTER
    if FLOPS_COUNTER is None:
        generated_flops_counter()
    return FLOPS_COUNTER


def set_count(count):
    global RECORDED_COUNT
    global TRAVERSED_COUNT
    RECORDED_COUNT = count[0]
    TRAVERSED_COUNT = count[1]


def get_count():
    global RECORDED_COUNT
    global TRAVERSED_COUNT
    if RECORDED_COUNT == 0 and TRAVERSED_COUNT == 0:
        flops_counter = get_flops_counter()
        count = flops_counter.get_flops()
        set_count(count)
    return RECORDED_COUNT, TRAVERSED_COUNT


def judge_if_profile(args):
    if not hasattr(args, 'profile_npu') or not args.profile_npu:
        return False
    if (torch.distributed.get_rank() in args.profile_ranks) or (-1 in args.profile_ranks):
        return True
    return False
