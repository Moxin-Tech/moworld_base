import os
import shutil
import tempfile

import torch
import torch.nn as nn
import torch_npu

import mindspeed.megatron_adaptor
from utils import judge_expression
from mindspeed_mm.configs.config import ConfigReader
from mindspeed_mm.tools.profiler import Profiler


class TestProfiler:

    def setup_class(self):
        self.profiler_path = os.path.join(tempfile.mkdtemp())
        print(f">> create temporary directory: {self.profiler_path}")

        config = {
            "profile": {
                "enable": True,
                "profile_type": "static",
                "ranks": [-1],
                "static_param": {
                    "level": "level1",
                    "with_stack": False,
                    "with_memory": False,
                    "record_shapes": False,
                    "with_cpu": True,
                    "save_path": self.profiler_path,
                    "start_step": 10,
                    "end_step": 11,
                    "data_simplification": True,
                    "aic_metrics_type": "PipeUtilization",
                    "analyse_flag": True,
                },
                "dynamic_param": {
                    "config_path": self.profiler_path
                }
            }
        }
        tool_cfg = ConfigReader(config)
        self.prof = Profiler(tool_cfg.profile)

        device = "cpu"
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
        self.mock_model = nn.Linear(10, 10, device=device)
        self.mock_data = torch.rand(1, 10, device=device)

    def setup_method(self):
        self.prof.start()

    def teardown_class(self):
        if os.path.exists(self.profiler_path):
            shutil.rmtree(self.profiler_path)
            print(f">> delete temporary directory: {self.profiler_path}")

    def teardown_method(self):
        self.prof.stop()

    def test_static_profiler_in_L1_level(self):
        for _ in range(20):
            self.mock_model(self.mock_data)
            self.prof.step()
        judge_expression(len(os.listdir(self.profiler_path)) > 0)
