import torch
import torch_npu
import mindspeed.megatron_adaptor

from mindspeed_mm.models.diffusion.wan_flow_match_scheduler import WanFlowMatchScheduler
from tests.ut.utils import judge_expression


class TestWanFlowMatchScheduler:
    @staticmethod
    def test_training_init():
        train_num_timesteps = 500
        scheduler = WanFlowMatchScheduler(num_train_timesteps=train_num_timesteps)
        judge_expression(not scheduler.do_classifier_free_guidance)
        judge_expression(isinstance(scheduler.sigmas, torch.Tensor))
        judge_expression(
            scheduler.num_train_timesteps == train_num_timesteps and
            scheduler.timesteps.shape[0] == train_num_timesteps and
            scheduler.linear_timesteps_weights.shape == (train_num_timesteps,)
        )

    @staticmethod
    def test_inference_init():
        test_num_timesteps = 50
        scheduler = WanFlowMatchScheduler(num_inference_timesteps=test_num_timesteps)
        judge_expression(scheduler.timesteps.shape[0] == test_num_timesteps)
        judge_expression(not hasattr(scheduler, "linear_timesteps_weights"))

    @staticmethod
    def test_sigma_conditions():
        scheduler = WanFlowMatchScheduler(
            num_inference_timesteps=100,
            inverse_timesteps=True,
            extra_one_step=True,
            reverse_sigmas=True,
            shift=3.0
        )
        extra_sigmas = torch.linspace(1.0, 0.003 / 1.002, 101)[:-1]
        flip_sigmas = torch.flip(extra_sigmas, dims=[0])
        scaled_sigmas = 3.0 * flip_sigmas / (1 + 2.0 * flip_sigmas)
        judge_expression(torch.allclose(scheduler.sigmas, 1 - scaled_sigmas))

    @staticmethod
    def test_training_weight_loss():
        scheduler = WanFlowMatchScheduler(num_train_timesteps=200)
        timestep = torch.Tensor([50])
        timestep_idx = torch.argmin((scheduler.timesteps - timestep).abs())
        expect_weights = scheduler.linear_timesteps_weights[timestep_idx]
        weights = scheduler._training_weight(timestep)
        judge_expression(torch.isclose(weights, expect_weights))

        model_output = torch.ones([1, 16, 16, 30, 50])
        loss = scheduler.training_losses(
            model_output=model_output,
            x_start=model_output,
            noise=torch.zeros_like(model_output),
            t=timestep
        )
        expect_loss = torch.nn.functional.mse_loss(model_output, -model_output) * expect_weights
        judge_expression(torch.allclose(loss, expect_loss))

    @staticmethod
    def test_sample():
        class Dummy_Model:
            def __call__(self, x, t, emb, **kwargs):
                return torch.zeros_like(x)

        scheduler_classifier = WanFlowMatchScheduler(
            num_inference_timesteps=5,
            guidance_scale=0.0
        )
        scheduler_classifier_free = WanFlowMatchScheduler(
            num_train_timesteps=10,
            guidance_scale=5.0
        )
        latents = torch.randn([1, 16, 16, 30, 50])
        output_classifer = scheduler_classifier.sample(Dummy_Model(), latents, {})
        output_classifer_free = scheduler_classifier_free.sample(Dummy_Model(), latents, {})
        judge_expression(output_classifer.shape == latents.shape)
        judge_expression(output_classifer_free.shape == latents.shape)

    @staticmethod
    def test_boundary_sigma():
        scehduler = WanFlowMatchScheduler(num_inference_timesteps=3, reverse_sigmas=False)
        last_timestep = scehduler.timesteps[-1]
        sample_before = torch.randn([1, 16, 16, 30, 50])
        model_output = torch.randn([1, 16, 16, 30, 50])
        sample_after = scehduler._step(model_output, last_timestep, sample_before)

        current_sigma = scehduler.sigmas[-1]
        next_sigma = 0
        expect_sample = sample_before + model_output * (next_sigma - current_sigma)
        judge_expression(torch.allclose(sample_after, expect_sample))

    @staticmethod
    def test_q_sample_randomness():
        torch.manual_seed(42)
        scheduler = WanFlowMatchScheduler()
        latents = torch.ones([1, 16, 16, 30, 50])
        noised_1, _, __ = scheduler.q_sample(latents)
        torch.manual_seed(42)
        noised_2, _, __ = scheduler.q_sample(latents)
        judge_expression(torch.allclose(noised_1, noised_2))
