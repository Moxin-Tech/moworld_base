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

from functools import wraps
from mindspeed.arguments import _add_auto_settings_args


def extra_args_provider_decorator(extra_args_provider):
    @wraps(extra_args_provider)
    def wrapper(parser):
        if extra_args_provider is not None:
            parser = extra_args_provider(parser)
        parser = process_args(parser)
        return parser

    return wrapper


def process_args(parser):
    parser.conflict_handler = "resolve"
    parser = _add_lora_args(parser)
    parser = _add_training_args(parser)
    parser = _add_network_size_args(parser)
    parser = _add_dummy_optimizer_args(parser)
    parser = _add_logging_args(parser)
    parser = _add_security_args(parser)
    parser = _add_auto_parallel_mm_args(parser)
    parser = _add_rlfh_args(parser)
    parser = _add_network_args(parser)
    parser = _add_data_balance_args(parser)
    parser = _add_auto_settings_args(parser)
    parser = _add_optim_arguments(parser)
    parser = _add_muon_optim_arguments(parser)
    parser = _add_text_dynamic_batching_args(parser)
    parser = _add_image_mbs_balance_args(parser)
    return parser


def _add_lora_args(parser):
    group = parser.add_argument_group(title='lora')

    group.add_argument('--lora-target-modules', nargs='+', type=str, default=[],
                       help='Use lora in target modules.')
    group.add_argument('--lora-target-parameters', nargs='+', type=str, default=[],
                       help='Use lora in target parameters.')
    group.add_argument('--lora-apply-modules', nargs='+', type=str, default=["all"],
                       help='Use lora exclude modules')
    group.add_argument('--lora-mixed-training', type=bool, default=False,
                       help='Mixed training for lora and non-lora args')
    group.add_argument('--load-base-model', type=str, default=None,
                       help='Directory containing a base model checkpoint for lora.')
    group.add_argument('--lora-dropout', type=float, default=0.0, help="lora dropout rate")
    group.add_argument('--lora-r', type=int, default=8,
                       help='Lora rank.')
    group.add_argument('--lora-alpha', type=int, default=16,
                       help='Lora alpha.')
    group.add_argument('--lora-register-forward-hook', nargs='+', type=str,
                       default=['word_embeddings', 'input_layernorm', 'final_layernorm'],
                       help='Lora register forward hook.')

    return parser


def _add_training_args(parser):
    group = parser.add_argument_group(title='training')

    group.add_argument('--use-deter-comp',
                       action='store_true',
                       default=False,
                       help='Enable deterministic computing for npu')
    group.add_argument('--jit-compile',
                       action='store_true',
                       default=False,
                       help='Setting jit compile mode to True')
    group.add_argument('--allow-tf32',
                       action='store_true',
                       default=False,
                       help='Use tf32 to train')
    group.add_argument('--downcast-to-bf16',
                       action='store_true',
                       default=False,
                       help='whether to downcast model weight from fp32 to bf16 while loading ckpt')
    group.add_argument('--allow-internal-format',
                       action='store_true',
                       default=False,
                       help='Use internal format to train')
    group.add_argument('--virtual-pipeline-model-parallel-size',
                       type=int,
                       default=None,
                       help='vpp size')
    group.add_argument('--encoder-dp-balance',
                       action='store_true',
                       default=False,
                       help='Balance for encoder')
    group.add_argument('--recompute-skip-core-attention',
                       action='store_true',
                       default=False,
                       help='Recomputing will skip the Flash attention if True')
    group.add_argument('--recompute-num-layers-skip-core-attention',
                       type=int,
                       default=0)
    group.add_argument('--hetero-parallel',
                       action='store_true',
                       default=False,
                       help='apply different parallelism to different models')
    group.add_argument('--hetero-encoder-mbs-scale',
                       type=int,
                       default=1,
                       help='Adjust ViT/audio encoder MBS to x-times LLM decoder MBS (x = this param)')
    group.add_argument('--calculate-per-sample-loss',
                       action='store_true',
                       default=False,
                       help=('Calculate the loss at the sample level: perform token-level mean '
                       'within each sample, and sequence-level mean across samples.'))
    group.add_argument('--calculate-square-loss',
                       action='store_true',
                       default=False,
                       help=('Calculate the loss.'))
    group.add_argument('--calculate-token-loss',
                       action='store_true',
                       default=False,
                       help=('Calculate the loss.'))
    group.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd', 'muon'],
                       help='Optimizer function')
    return parser


def _add_network_size_args(parser):
    group = parser.add_argument_group(title='network_size_args')

    group.add_argument('--padded-vocab-size',
                       type=int,
                       default=None,
                       help='set padded vocab size')

    return parser


def _add_dummy_optimizer_args(parser):
    group = parser.add_argument_group(title='dummy optimizer args')

    group.add_argument('--enable-dummy-optimizer',
                       action='store_true',
                       default=False,
                       help='enable dummy optimizer')

    return parser


def _add_logging_args(parser):
    group = parser.add_argument_group(title='logging')

    group.add_argument('--log-tps',
                       action='store_true',
                       default=False,
                       help='calculate and log average tokens per sample')

    return parser


def _add_security_args(parser):
    group = parser.add_argument_group(title='security configuration')

    group.add_argument('--trust-remote-code',
                       action='store_true',
                       default=False,
                       help='Whether or not to allow for custom models defined on the Hub in their own modeling files.')

    return parser


def _add_auto_parallel_mm_args(parser):
    group = parser.add_argument_group(title='auto_parallel_mm')
    group.add_argument('--profile-subgraph-seg', action='store_true', default=False, help='model segmentation')
    group.add_argument('--profile-stage', type=int, default=None, help='model profile stage')
    group.add_argument('--simulated-nnodes', type=int, default=None, help='the simulated number of node in the cluster')
    group.add_argument('--simulated-nproc-per-node', type=int, default=None, help='the simulated number of NPU on each node')

    return parser


def _add_rlfh_args(parser):
    group = parser.add_argument_group(title='dpo')

    group.add_argument(
        '--dpo-beta',
        type=float,
        default=0.1,
        help="The beta parameter for the DPO loss"
    )
    group.add_argument(
        '--dpo-loss-type',
        default="sigmoid",
        choices=["sigmoid"],
        help="The type of DPO loss to use"
    )
    group.add_argument(
        "--dpo-label-smoothing",
        type=float,
        default=0.0,
        help="The robust DPO label smoothing parameter in cDPO that should be between 0 and 0.5."
    )
    group.add_argument(
        '--ref-model',
        default=None,
        type=str,
        help='Path to the reference model used for the PPO or DPO training.'
    )
    group.add_argument(
        '--pref-ftx',
        default=0.0,
        type=float,
        help="The supervised fine-tuning loss coefficient in DPO training.",
    )

    return parser


def _add_network_args(parser):
    group = parser.add_argument_group(title='network')

    # MM_GRPO use，judging training methods
    group.add_argument(
        '--stage',
        default=None,
        choices=["ray_grpo"],
        help='Determine training mode'
    )

    return parser


def _add_data_balance_args(parser):
    group = parser.add_argument_group(title="GBS_data_balance")
    group.add_argument("--use-data-balance",
                       action='store_true',
                       default=False,
                       help="Enable data balance")
    group.add_argument("--data_balance_sorting_algo", type=str, default="post_global_balancing_greedy_without_pad",
                       help="data balance sorting algorithm:"
                            "post_global_balancing_greedy_without_pad: a greedy post global balancing algorithm without padding")

    return parser


def _add_image_mbs_balance_args(parser):
    group = parser.add_argument_group(title="MBS_data_balance")
    group.add_argument("--use-image-mbs-data-balance",
                       action='store_true',
                       default=False,
                       help="Enable data balance")
    group.add_argument("--mbs_data_balance_sorting_algo", type=str, default="post_mbs_balancing_greedy_without_pad",
                       help="data balance sorting algorithm:"
                            "post_mbs_balancing_greedy_without_pad: a greedy post local balancing algorithm without padding")

    return parser


def _add_text_dynamic_batching_args(parser):
    group = parser.add_argument_group(title="text_dynamic_batching")
    group.add_argument("--use-txt-dynamic-batching",
                       action='store_true',
                       default=False,
                       help="Enable dynamic batching for LLM")
    group.add_argument("--max-seq-len", type=int, default=2048,
                       help="max sequence length of concatenated text for each micro batch")
    group.add_argument("--dynamic-batch-buffer-size", type=int, default=200,
                       help="the size of dynamic batching buffer")

    return parser


def _add_optim_arguments(parser):
    group = parser.add_argument_group(title='optimization_filter')

    group.add_argument(
        '--weight-decay-exclude-modules',
        nargs='+',
        type=str,
        default=[],
        help='Keywords in parameter names to exclude from weight decay. Empty list disables this feature.'
    )
    group.add_argument(
        '--lr-scale-modules',
        nargs='+',
        type=str,
        default=[],
        help='Keywords in parameter names to apply learning rate scaling. Empty list disables this feature.'
    )
    group.add_argument(
        '--lr-mult',
        type=float,
        default=1.0,
        help='Learning rate multiplier for parameters matching scale-lr-keywords. '
    )

    return parser


def _add_muon_optim_arguments(parser):
    group = parser.add_argument_group(title='muon_optimizer')

    group.add_argument(
        '--matched-adamw-rms',
        type=float,
        default=0.2,
        help='Matched AdamW RMS value for Muon optimizer. '
                'Controls how closely Muon matches AdamW update magnitude. '
                'Typical range: 0.1-0.4. Default: 0.2'
    )

    group.add_argument(
        '--muon-momentum',
        type=float,
        default=0.95,
        help='Momentum coefficient for Muon internal SGD. '
                'Higher values give more weight to previous updates. '
                'Range: 0.0-1.0. Default: 0.95'
    )

    group.add_argument(
        '--ns-steps',
        type=int,
        default=5,
        help='Number of Newton-Schulz iterations for orthogonalization. '
                'More steps give better orthogonalization but slower training. '
                'Range: 1-10. Default: 5'
    )

    return parser
