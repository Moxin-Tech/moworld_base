import os
import torch
import torch_npu
import mindspeed.megatron_adaptor

from diffusers.schedulers import (
    DDPMScheduler,
    EulerAncestralDiscreteScheduler,
    UniPCMultistepScheduler
)

from megatron.training.arguments import parse_args
from megatron.training.global_vars import set_args
from megatron.training.initialize import _initialize_distributed
from mindspeed_mm.models.diffusion.diffusers_scheduler import DiffusersScheduler
from tests.ut.utils import initialize_model_parallel, judge_expression


class TestDiffusersScheduler:
    @staticmethod
    def test_ddpm_init():
        config = {
            "model_id": "DDPM",
            "num_train_steps": 1000,
            "noise_offset": 0.02,
            "snr_gamma": 5.0,
            "prediction_type": "epsilon",
            "guidance_scale": 4.5
        }
        diffusion = DiffusersScheduler(config).diffusion
        judge_expression(isinstance(diffusion, DDPMScheduler))

    @staticmethod
    def test_euler_ancestral_discrete_init():
        config = {
            "model_id": "EulerAncestralDiscrete",
            "num_inference_steps": 100,
            "guidance_scale": 7.5,
            "prediction_type": "v_prediction",
            "rescale_betas_zero_snr": True,
            "device": "npu",
            "timestep_spacing": "trailing"
        }
        diffusion = DiffusersScheduler(config).diffusion
        judge_expression(isinstance(diffusion, EulerAncestralDiscreteScheduler))

    @staticmethod
    def test_unipcmultistep_init():
        config = {
            "model_id": "UniPCMultistepScheduler",
            "num_train_steps": 1000,
            "num_inference_steps": 50,
            "guidance_scale": 5.0,
            "beta_end": 0.02,
            "beta_schedule": "linear",
            "beta_start": 0.0001,
            "disable_corrector": [],
            "dynamic_thresholding_ratio": 0.995,
            "final_sigmas_type": "zero",
            "lower_order_final": True,
            "predict_x0": True,
            "prediction_type": "flow_prediction",
            "rescale_betas_zero_snr": False,
            "sample_max_value": 1.0,
            "solver_order": 2,
            "solver_p": None,
            "solver_type": "bh2",
            "steps_offset": 0,
            "thresholding": False,
            "timestep_spacing": "linspace",
            "trained_betas": None,
            "use_karras_sigmas": False
        }
        diffusion = DiffusersScheduler(config).diffusion
        judge_expression(isinstance(diffusion, UniPCMultistepScheduler))

    @staticmethod
    def test_training_loss():
        config = {
            "model_id": "DDPM",
            "num_train_steps": 1000,
            "noise_offset": 0.02,
            "snr_gamma": 5.0,
            "prediction_type": "epsilon",
            "guidance_scale": 4.5
        }
        diffusion = DiffusersScheduler(config)

        model_output = torch.rand([1, 16, 16, 30, 50])
        x_start = torch.rand([1, 16, 16, 30, 50])
        noise = torch.rand([1, 16, 16, 30, 50])
        t = torch.randint(0, 1000, (1,), dtype=torch.int64)

        loss = diffusion.training_losses(
            model_output=model_output,
            x_start=x_start,
            noise=noise,
            t=t
        )
        judge_expression(isinstance(loss, torch.FloatTensor))

    @staticmethod
    def test_q_sample():
        args = parse_args(None, True)
        args.context_parallel_size = 1
        args.tensor_model_parallel_size = 1
        args.pipeline_model_parallel_size = 1
        args.virtual_pipeline_model_parallel_size = None
        set_args(args)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6000"
        _initialize_distributed(None, None)
        initialize_model_parallel()
        config = {
            "model_id": "DDPM",
            "num_train_steps": 1000,
            "noise_offset": 0.02,
            "snr_gamma": 5.0,
            "prediction_type": "epsilon",
            "guidance_scale": 4.5
        }
        diffusion = DiffusersScheduler(config)

        x_start = torch.rand([1, 16, 16, 30, 50])

        x_t, noise, t = diffusion.q_sample(x_start)
        judge = (x_t.shape == x_start.shape) and (noise.shape == x_start.shape) and isinstance(t, torch.Tensor)
        judge_expression(judge)
