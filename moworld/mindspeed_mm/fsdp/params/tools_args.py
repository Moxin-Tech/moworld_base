# Copyright 2025 Bytedance Ltd. and/or its affiliates
from dataclasses import dataclass, field
from typing import List, Literal, Optional
import logging

from mindspeed_mm.config.arguments.base_args import BaseArguments


logger = logging.getLogger(__name__)


class StaticParam(BaseArguments):
    level: str = field(
        default="level1",
        metadata={"help": "The info level of profiler."},
    )
    with_stack: bool = field(
        default=False,
        metadata={"help": "Whether to collect operator call stack info."},
    )
    with_memory: bool = field(
        default=False,
        metadata={"help": "Whether to collect the memory usage of the operator."},
    )
    record_shapes: bool = field(
        default=False,
        metadata={"help": "Whether to collect the innput shapes and input types of operators."},
    )
    with_cpu: bool = field(
        default=False,
        metadata={"help": "Whether to collect CPU events."},
    )
    save_path: str = field(
        default="./profiling",
        metadata={"help": "Direction to export the profiling result."},
    )
    start_step: int = field(
        default=10,
        metadata={"help": "Start step for profiling."},
    )
    end_step: int = field(
        default=11,
        metadata={"help": "End step for profiling."},
    )
    data_simplification: bool = field(
        default=False,
        metadata={"help": "Whether to enable the data simplification mode."},
    )
    aic_metrics_type: str = field(
        default="PipeUtilization",
        metadata={"help": "AI Core performance metric collection items."},
    )
    analyse_flag: bool = field(
        default=True,
        metadata={"help": "Whether to analyse profiling online."},
    )


class Profiler(BaseArguments):
    enable: bool = field(
        default=False,
        metadata={"help": "Enable profiling."},
    )
    profile_type: str = field(
        default="static",
        metadata={"help": "the type of profiling"},
    )
    ranks: List[int] = field(
        default_factory=lambda: [0],
        metadata={
            "help": "List of ranks to profile (default is rank 0 only)"
        },
    )
    static_param: StaticParam = field(default_factory=StaticParam)


class MemoryProfiler(BaseArguments):
    enable: bool = field(
        default=False,
        metadata={"help": "Enable memory profiling."},
    )
    start_step: int = field(
        default=1,
        metadata={"help": "Start step for memory profiling."},
    )
    end_step: int = field(
        default=2,
        metadata={"help": "End step for memory profiling."},
    )
    save_path: str = field(
        default="./memory_snapshot",
        metadata={"help": "Direction to export the memory profiling result."},
    )
    dump_ranks: List[int] = field(
        default_factory=lambda: [0],
        metadata={"help": "List of ranks to memory profile (default is rank 0 only)"},
    )
    stacks: Literal["python", "all"] = field(
        default="all",
        metadata={
            "help": "python, include Python, TorchScript, and inductor frames in tracebacks, all, additionally include C++ frames."},
    )
    max_entries: Optional[int] = field(
        default=None,
        metadata={"help": "Keep a maximum of `max_entries` alloc/free events in the recorded history recorded."},
    )
    mem_info: bool = field(
        default=False,
        metadata={"help": "Whether to print memory infos."},
    )


class ToolsArguments(BaseArguments):
    profile: Profiler = field(default_factory=Profiler)
    memory_profile: MemoryProfiler = field(default_factory=MemoryProfiler)
