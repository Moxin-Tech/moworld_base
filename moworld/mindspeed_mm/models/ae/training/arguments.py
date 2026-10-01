# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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


def parse_ae_args():
    parser = argparse.ArgumentParser(description="Distributed Training")
    parser.add_argument("--seed", type=int, default=1234, help="seed")

    # Standard arguments.
    parser = _add_learning_rate_args(parser)
    parser = _add_training_args(parser)
    parser = _add_checkpointing_args(parser)
    parser = _add_mix_precision_args(parser)

    # Custom arguments.
    parser.add_argument("--data-config", type=str, default=None)
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--tool-config", type=str, default=None)

    # Parse.
    args = parser.parse_args()

    # Args from environment
    args.rank = int(os.getenv("rank", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))

    return args


def _add_learning_rate_args(parser):
    group = parser.add_argument_group(title="learning rate")

    group.add_argument("--ae-lr", type=float, default=1e-5,
                       help="ae model learning rate")
    group.add_argument("--discriminator-lr", type=float, default=1e-5,
                       help="discriminator model learning rate")
    return parser


def _add_training_args(parser):
    group = parser.add_argument_group(title="training")

    group.add_argument("--epochs", type=int, default=10,
                        help="number of epochs to train")
    group.add_argument('--micro-batch-size', type=int, default=None,
                       help='Batch size per model instance (local batch size). '
                       'Global batch size is local batch size times data '
                       'parallel size times number of micro batches.')
    group.add_argument("--train-iters", type=int, default=None,
                       help="Total number of iterations to train over all "
                       "training runs.")
    group.add_argument('--num-workers', type=int, default=2,
                       help="Dataloader number of workers.")
    group.add_argument("--log-interval", type=int, default=5,
                       help="Report loss interval.")
    group.add_argument("--ae-wd", type=float, default=1e-4,
                       help="ae model weight decay.")
    group.add_argument("--discriminator-wd", type=float, default=1e-2,
                       help="discriminator model weight decay.")
    group.add_argument("--freeze_encoder", action="store_true")
    group.add_argument("--clip_grad_norm", type=float, default=1e5)
    group.add_argument("--find_unused_parameters", action="store_true")
    group.add_argument("--ema", action="store_true")
    group.add_argument("--ema_decay", type=float, default=0.999)

    return parser


def _add_checkpointing_args(parser):
    group = parser.add_argument_group(title="checkpointing")

    group.add_argument("--save", type=str, default="./results/",
                        help="Output directory to save checkpoints to.")
    group.add_argument("--save-interval", type=int, default=1000,
                       help="Number of iterations between checkpoint saves")
    group.add_argument("--load", type=str, default=None,
                       help="ckpt file containing a model checkpoint.")

    return parser


def _add_mix_precision_args(parser):
    group = parser.add_argument_group(title="mixed precision")

    group.add_argument("--mix-precision", type=str, default="bf16",
                       choices=["fp16", "bf16", "fp32"],
                       help="mixed precision for training.")

    return parser
