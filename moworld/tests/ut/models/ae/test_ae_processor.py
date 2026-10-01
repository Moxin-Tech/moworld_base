import os

import torch

import mindspeed.megatron_adaptor
from mindspeed_mm.models.ae.base import AEModel
from mindspeed_mm.utils.utils import is_npu_available
from tests.ut.utils import TestConfig, judge_expression

VAE_ENCODE_OUTPUT = [-22469.75, -28455.1484375]
VAE_DECODE_OUTPUT = [2160273.75, 935986.1875]
RELATIVE_DIFF_THRESHOLD = 0.0001
VIDEO_PATH = "/home/ci_resource/models/opensoraplan1_3/wfvae/test_vae_video.pt"
WFVAE_PATH = "/home/ci_resource/models/opensoraplan1_3/wfvae/wfvae_mm.pt"
HYV_VAE_PATH = "/home/ci_resource/models/hunyuanvideo_t2v/hunyuan_vae/pytorch_model.pt"


def init_distributed():
    if not torch.distributed.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        torch.distributed.init_process_group(
            backend='gloo',
            rank=0,
            world_size=1
        )


if is_npu_available():
    import torch_npu
    from torch_npu.contrib import transfer_to_npu

    torch.npu.config.allow_internal_format = False


class TestAE:
    # Please add tests at the end of the list.
    VAE_TEST_CASES = [
        {
            "model_id": "wfvae",
            "from_pretrained": WFVAE_PATH,
            "dtype": "bf16",
            "output_dtype": "bf16",
            "base_channels": 128,
            "decoder_energy_flow_hidden_size": 128,
            "decoder_num_resblocks": 2,
            "dropout": 0.0,
            "encoder_energy_flow_hidden_size": 64,
            "encoder_num_resblocks": 2,
            "latent_dim": 8,
            "use_attention": True,
            "norm_type": "aelayernorm",
            "t_interpolation": "trilinear",
            "use_tiling": False,
            "connect_res_layer_num": 2,
            "vae_cp_size": 1
        },
        {
            "model_id": "autoencoder_kl_hunyuanvideo",
            "from_pretrained": HYV_VAE_PATH,
            "dtype": "float32",
            "latent_channels": 16,
            "block_out_channels": [128, 256, 512, 512],
            "layers_per_block": 2,
            "in_channels": 3,
            "norm_num_groups": 32,
            "out_channels": 3,
            "sample_size": 256,
            "sample_tsize": 64,
            "down_block_types": [
                "DownEncoderBlockCausal3D",
                "DownEncoderBlockCausal3D",
                "DownEncoderBlockCausal3D",
                "DownEncoderBlockCausal3D"
            ],
            "up_block_types": [
                "UpDecoderBlockCausal3D",
                "UpDecoderBlockCausal3D",
                "UpDecoderBlockCausal3D",
                "UpDecoderBlockCausal3D"
            ],
            "scaling_factor": 0.476986,
            "time_compression_ratio": 4,
            "mid_block_add_attention": True,
            "act_fn": "silu",
            "enable_tiling": True
        }
    ]

    def run_vae_test(self, idx, vae_config):
        init_distributed()
        test_name = vae_config["model_id"]
        if vae_config["dtype"] == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        # init model
        ae_configs = TestConfig(vae_config)
        vae_model = AEModel(ae_configs).eval()
        vae_model.requires_grad_(False)
        vae_model = vae_model.npu().to(dtype)
        test_video = torch.load(VIDEO_PATH, map_location="cpu").to(dtype).npu()

        # test encode
        encode_output, _ = vae_model.encode(test_video)
        judge_expression(abs(encode_output.clone().float().cpu().sum().item() - VAE_ENCODE_OUTPUT[idx]) / abs(VAE_ENCODE_OUTPUT[idx]) < RELATIVE_DIFF_THRESHOLD)

        # test decode
        decode_output = vae_model.decode(encode_output)
        judge_expression(abs(decode_output.clone().float().cpu().sum().item() - VAE_DECODE_OUTPUT[idx]) / abs(VAE_DECODE_OUTPUT[idx]) < RELATIVE_DIFF_THRESHOLD)

    def test_all_vae(self):
        for idx, case_config in enumerate(self.VAE_TEST_CASES):
            self.run_vae_test(idx, case_config)
