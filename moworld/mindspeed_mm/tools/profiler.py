# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import torch
import torch_npu

from torch_npu.profiler.profiler import analyse


class Profiler:
    """
    Instantiate a Profiler from config.

    Args:
        config (dict): the general config for Text Encoder Model
        {
            "enable": type-bool, enable profiling capability
            "profile_type": type-str, static or dynamic
            "ranks": type-list, global ranks to profile.The default value of -1 means to profile all ranks,
            "static_param":
                "level": type-str, profiling level0, level1, level2,
                "with_stack": type-bool, profiling with stack info,
                "with_memory": type-bool, profiling with memory info,
                "record_shapes": type-bool, profiling with shape info,
                "with_cpu": type-bool, profiling with cpu info,
                "save_path": type-str, path to save profiling files,
                "start_step": type-int, profiling start step,
                "end_step": type-int, profiling end step,
                "data_simplification": type-bool, profiling with Simplified data,
            "dynamic_param":
                "config_path": type-str, path of config and log,
        }

    example:
        prof = Profiler(prof_config)
        prof.start()
        while train:
            train_one_step
            prof.step()
        prof.stop()
    """
    def __init__(self, config):
        self.enable = config.enable
        self.profile_type = config.profile_type
        self.ranks = config.ranks

        self.sp_level = config.static_param.level
        self.sp_with_stack = config.static_param.with_stack
        self.sp_with_memory = config.static_param.with_memory
        self.sp_record_shapes = config.static_param.record_shapes
        self.sp_with_cpu = config.static_param.with_cpu
        self.sp_save_path = config.static_param.save_path
        self.sp_start_step = config.static_param.start_step
        self.sp_end_step = config.static_param.end_step
        self.sp_data_simplification = config.static_param.data_simplification
        self.sp_analyse_flag = config.static_param.analyse_flag

        self.dp_config_path = config.dynamic_param.config_path

        self.aic_metrics_type = config.static_param.aic_metrics_type

        if self.profile_type == "static":
            if self.sp_level == 'level0':
                profiler_level = torch_npu.profiler.ProfilerLevel.Level0
            elif self.sp_level == 'level1':
                profiler_level = torch_npu.profiler.ProfilerLevel.Level1
            elif self.sp_level == 'level2':
                profiler_level = torch_npu.profiler.ProfilerLevel.Level2
            else:
                raise ValueError(f"profiler_level only supports level0,"
                                f" 1, and 2, but gets {self.sp_level}")
            if self.aic_metrics_type == 'PipeUtilization':
                aic_metrics_type = torch_npu.profiler.AiCMetrics.PipeUtilization
            elif self.aic_metrics_type == 'ArithmeticUtilization':
                aic_metrics_type = torch_npu.profiler.AiCMetrics.ArithmeticUtilization
            else:
                raise ValueError(f"aic_metrics_type only supports PipeUtilization and ArithmeticUtilization")
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                aic_metrics=aic_metrics_type,
                profiler_level=profiler_level,
                data_simplification=self.sp_data_simplification,
            )
            skip_first = self.sp_start_step
            active = self.sp_end_step - self.sp_start_step

            activities = [torch_npu.profiler.ProfilerActivity.NPU]
            if self.sp_with_cpu:
                activities.append(torch_npu.profiler.ProfilerActivity.CPU)

            self.prof = torch_npu.profiler.profile(
                with_stack=self.sp_with_stack,
                record_shapes=self.sp_record_shapes,
                profile_memory=self.sp_with_memory,
                activities=activities,
                schedule=torch_npu.profiler.schedule(
                    wait=0, warmup=0, active=active, repeat=1, skip_first=skip_first),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(self.sp_save_path,
                                                                            analyse_flag=self.sp_analyse_flag),
                experimental_config=experimental_config)

        elif self.profile_type == "dynamic":
            from torch_npu.profiler import dynamic_profile as dp
            self.prof = dp

        else:
            raise ValueError(f"profile_type only supports static and dynamic,"
                            f" but gets {self.profile_type}")

    def _enable_profile(self):
        '''
        Determine whether to enable profile
        '''
        if not self.enable:
            return False
        if self.ranks == [-1]:
            return True
        if torch.distributed.get_rank() in self.ranks:
            return True
        return False

    def start(self):
        if self._enable_profile():
            if self.profile_type == "static":
                self.prof.start()
            else:
                self.prof.init(self.dp_config_path)

    def step(self):
        if self._enable_profile():
            self.prof.step()

    def stop(self):
        if self._enable_profile():
            if self.profile_type == "static":
                self.prof.stop()
            else:
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="profile offline analysing tool")
    parser.add_argument("--mm-tool", help="MM tool config path", default="./mindspeed_mm/tools/tools.json")
    parser.add_argument("--profiler-path", help="Path to the profiler data directory")
    parser.add_argument("--max-process-number", type=int, help="Maximum process number for analysis (default: CPU cores / 2)")
    parser.add_argument("--export-type", action="append", choices=["text", "db"],
                        help="Export type(s) for analysis results, supports: text, db, can be specified multiple times, default: text")
    args = parser.parse_args()

    # Determine profiler path
    if args.profiler_path:
        profiler_path = args.profiler_path
    else:
        from mindspeed_mm.configs.config import MMConfig
        config = MMConfig({"tool": args.mm_tool})
        profiler_path = os.path.join(config.tool.profile.static_param.save_path)

    analyse(
        profiler_path=profiler_path,
        max_process_number=args.max_process_number,
        export_type=args.export_type
    )
