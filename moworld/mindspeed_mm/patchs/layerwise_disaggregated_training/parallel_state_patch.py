# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import os
from datetime import timedelta
from functools import partial, wraps
from typing import Union, Callable, List, Optional

import torch
import numpy as np

from megatron.training import get_args, print_rank_0
from megatron.core import mpu
from megatron.core.parallel_state import (
    RankGenerator,
    default_embedding_ranks,
    default_position_embedding_ranks,
    get_nccl_options,
)

from megatron.core.utils import is_torch_min_version

# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None
# Model parallel group (both intra-, pipeline, and expert) that the current rank belongs to.
# Embedding group.
_EMBEDDING_GROUP = None
# Position embedding group.
_POSITION_EMBEDDING_GROUP = None

_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_PIPELINE_MODEL_PARALLEL_SPLIT_RANK = None

_PIPELINE_MODEL_PARALLEL_DECODER_START = None

# A list of ranks that have a copy of the embedding.
_EMBEDDING_GLOBAL_RANKS = None

# A list of ranks that have a copy of the position embedding.
_POSITION_EMBEDDING_GLOBAL_RANKS = None

# A list of global ranks for each pipeline group to ease calculation of the source
# rank when broadcasting from the first or last pipeline stage.
_PIPELINE_GLOBAL_RANKS = None

#
_PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE = None
_PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST = None
_PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST = None

# VTP (Virtual Tensor Parallelism) state
_VTP_ENABLED = False
_VTP_SIZE_LIST = None
_VTP_STAGE_RANKS = None
_VTP_INTRA_STAGE_GROUP = None
_VTP_MY_STAGE_IDX = None
_EDGE_TP_SIZE = 1

# VDP
_PIPELINE_MODEL_PARALLEL_GROUP_FOR_VDP_CROSS_CLOUD_TP = None
_PIPELINE_MODEL_PARALLEL_GROUP_FOR_VDP_CROSS_EDGE_CLOUD = None
_LAYERWISE_DISAGGREGATED_TRAINING = False
_VDP_SIZE = 1
_VDP_ENABLED = False


# 替换mpu.initialize_model_parallel
def install_ldt_init_model_parallel_patch():
    """Replace megatron's initialize_model_parallel with the VTP-aware wrapper.

    Called from validate_args (LDT branch only), which runs inside
    initialize_megatron BEFORE _initialize_distributed calls the function.
    """
    ori_fn = mpu.initialize_model_parallel
    if getattr(ori_fn, '_ldt_wrapped', False):
        return
    wrapped = initialize_model_parallel_wrapper(ori_fn)
    wrapped._ldt_wrapped = True
    mpu.initialize_model_parallel = wrapped


def initialize_model_parallel_wrapper(initialize_model_parallel):
    @wraps(initialize_model_parallel)
    def initialize_model_parallel_impl(
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        virtual_pipeline_model_parallel_size: Optional[int] = None,
        pipeline_model_parallel_split_rank: Optional[int] = None,
        pipeline_model_parallel_comm_backend: Optional[str] = None,
        use_sharp: bool = False,
        context_parallel_size: int = 1,
        hierarchical_context_parallel_sizes: Optional[List[int]] = None,
        expert_model_parallel_size: int = 1,
        num_distributed_optimizer_instances: int = 1,
        expert_tensor_parallel_size: Optional[int] = None,
        nccl_communicator_config_path: Optional[str] = None,
        distributed_timeout_minutes: int = 30,
        order: str = "tp-cp-ep-dp-pp",
        encoder_tensor_model_parallel_size: int = 0,
        encoder_pipeline_model_parallel_size: Optional[int] = 0,
        get_embedding_ranks: Optional[
            Callable[[List[int], Optional[int]], List[int]]
        ] = None,
        get_position_embedding_ranks: Optional[
            Callable[[List[int], Optional[int]], List[int]]
        ] = None,
        create_gloo_process_groups: bool = True,
    ) -> None:

        cli_args = get_args()

        global _LAYERWISE_DISAGGREGATED_TRAINING, _VDP_SIZE, _VDP_ENABLED
        _LAYERWISE_DISAGGREGATED_TRAINING = True
        _VDP_SIZE = cli_args.data_parallel_size

        # Auto-detect VTP sizes from per-node GPU topology when LDT is enabled
        vtp_sizes = _auto_detect_vtp_sizes(cli_args)
        vdp_size = cli_args.data_parallel_size

        _init_vdp_state(tensor_model_parallel_size, context_parallel_size, vdp_size, vtp_sizes)

        if _VDP_ENABLED:
            print_rank_0(f'Virtual DP enabled with dp size: {vdp_size}')

        if vtp_sizes and len(set(vtp_sizes)) > 1:
            org_args = (
                tensor_model_parallel_size,
                pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank,
            )
            org_kwargs = {
                "pipeline_model_parallel_comm_backend": pipeline_model_parallel_comm_backend,
                "use_sharp": use_sharp,
                "context_parallel_size": context_parallel_size,
                "hierarchical_context_parallel_sizes": hierarchical_context_parallel_sizes,
                "expert_model_parallel_size": expert_model_parallel_size,
                "num_distributed_optimizer_instances": num_distributed_optimizer_instances,
                "expert_tensor_parallel_size": expert_tensor_parallel_size,
                "nccl_communicator_config_path": nccl_communicator_config_path,
                "distributed_timeout_minutes": distributed_timeout_minutes,
                "order": order,
                "encoder_tensor_model_parallel_size": encoder_tensor_model_parallel_size,
                "encoder_pipeline_model_parallel_size": encoder_pipeline_model_parallel_size,
                "get_embedding_ranks": get_embedding_ranks,
                "get_position_embedding_ranks": get_position_embedding_ranks,
                "create_gloo_process_groups": create_gloo_process_groups,
            }

            if is_vdp_enabled():
                _initialize_vtp_static_vtp_vdp(initialize_model_parallel, vtp_sizes, org_args, org_kwargs)
            else:
                _initialize_vtp_static_only_vtp(initialize_model_parallel, vtp_sizes, org_args, org_kwargs)

            return

        # override get_world_size to support layerwise_disaggregated_training
        ori_get_world_size = torch.distributed.get_world_size

        def ldt_get_world_size(*args, **kwargs):
            real_world_size = ori_get_world_size()
            return real_world_size + (_VDP_SIZE - 1) * context_parallel_size * tensor_model_parallel_size

        torch.distributed.get_world_size = ldt_get_world_size

        # override RankGenerator to support layerwise_disaggregated_training
        ori_rank_generator = mpu.RankGenerator
        mpu.RankGenerator = LDTRankGenerator
        ori_create_group = mpu.create_group
        mpu.create_group = create_group

        # call ori initialize_model_parallel
        initialize_model_parallel(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank,
            pipeline_model_parallel_comm_backend,
            use_sharp,
            context_parallel_size,
            hierarchical_context_parallel_sizes,
            expert_model_parallel_size,
            num_distributed_optimizer_instances,
            expert_tensor_parallel_size,
            nccl_communicator_config_path,
            distributed_timeout_minutes,
            order,
            encoder_tensor_model_parallel_size,
            encoder_pipeline_model_parallel_size,
            get_embedding_ranks,
            get_position_embedding_ranks,
            create_gloo_process_groups,
        )

        # restore get_world_size and RankGenerator
        torch.distributed.get_world_size = ori_get_world_size
        mpu.RankGenerator = ori_rank_generator
        mpu.create_group = ori_create_group

        # sync all global variables from Megatron to LDT module
        _sync_all_global_variables(mpu)

        if encoder_pipeline_model_parallel_size is None:
            encoder_pipeline_model_parallel_size = 0

        if (
            encoder_tensor_model_parallel_size == 0
            and encoder_pipeline_model_parallel_size > 0
        ):
            encoder_tensor_model_parallel_size = tensor_model_parallel_size

        if get_embedding_ranks is None:
            get_embedding_ranks = partial(
                default_embedding_ranks, split_rank=pipeline_model_parallel_split_rank
            )

        if get_position_embedding_ranks is None:
            get_position_embedding_ranks = partial(
                default_position_embedding_ranks,
                split_rank=pipeline_model_parallel_split_rank,
            )

        if encoder_pipeline_model_parallel_size > 0:
            global _PIPELINE_MODEL_PARALLEL_DECODER_START
            _PIPELINE_MODEL_PARALLEL_DECODER_START = encoder_pipeline_model_parallel_size

        # Get world size and rank. Ensure some consistencies.
        if not torch.distributed.is_initialized():
            raise RuntimeError("torch.distributed is not initialized")
        world_size = torch.distributed.get_world_size() + (_VDP_SIZE - 1) * context_parallel_size * tensor_model_parallel_size

        if encoder_tensor_model_parallel_size > 0:
            if not (
                encoder_tensor_model_parallel_size <= tensor_model_parallel_size
            ):
                raise RuntimeError("We do not support encoders with more TP than the decoder.")

        encoder_model_size = (
            encoder_tensor_model_parallel_size
            * encoder_pipeline_model_parallel_size
            * context_parallel_size
        )
        decoder_model_size = (
            tensor_model_parallel_size
            * pipeline_model_parallel_size
            * context_parallel_size
        )
        total_model_size = encoder_model_size + decoder_model_size

        if world_size % total_model_size != 0:
            raise RuntimeError(
                f"world_size ({world_size}) is not divisible by {total_model_size}"
            )

        data_parallel_size: int = world_size // total_model_size

        encoder_world_size = encoder_model_size * data_parallel_size
        decoder_world_size = decoder_model_size * data_parallel_size

        if not (
            encoder_world_size + decoder_world_size == world_size
        ):
            raise RuntimeError(f"{encoder_world_size=} + {decoder_world_size=} != {world_size=}")

        if virtual_pipeline_model_parallel_size is not None:
            if not pipeline_model_parallel_size > 1:
                raise RuntimeError(
                    "pipeline-model-parallel size should be greater than 1 with interleaved schedule"
                )
            global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
            global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
            _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
            _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = (
                virtual_pipeline_model_parallel_size
            )

        if pipeline_model_parallel_split_rank is not None:
            global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
            _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank

        rank = torch.distributed.get_rank()

        nccl_comm_cfgs = {}
        if nccl_communicator_config_path is not None:
            try:
                import yaml
            except ImportError as e:
                raise RuntimeError(
                    "Cannot import `yaml`. Setting custom nccl communicator configs "
                    "requires the yaml package."
                ) from e

            with open(nccl_communicator_config_path, "r") as stream:
                nccl_comm_cfgs = yaml.safe_load(stream)

        if encoder_world_size > 0:
            encoder_rank_generator = LDTRankGenerator(
                tp=encoder_tensor_model_parallel_size,
                ep=1,
                dp=data_parallel_size,
                pp=encoder_pipeline_model_parallel_size,
                cp=context_parallel_size,
                order=order,
                rank_offset=0,
            )
        else:
            encoder_rank_generator = None

        decoder_rank_generator = LDTRankGenerator(
            tp=tensor_model_parallel_size,
            ep=1,
            dp=data_parallel_size,
            pp=pipeline_model_parallel_size,
            cp=context_parallel_size,
            order=order,
            rank_offset=encoder_world_size,
        )

        # Build expert rank generator
        if expert_tensor_parallel_size is None:
            expert_tensor_parallel_size = tensor_model_parallel_size
        expert_tensor_model_pipeline_parallel_size = (
            expert_tensor_parallel_size
            * expert_model_parallel_size
            * pipeline_model_parallel_size
        )
        expert_data_parallel_size = (
            decoder_world_size // expert_tensor_model_pipeline_parallel_size
        )
        if decoder_world_size % expert_tensor_model_pipeline_parallel_size != 0:
            raise RuntimeError(
                f"decoder world_size ({decoder_world_size}) is not divisible by expert_tensor_model_pipeline_parallel size ({expert_tensor_model_pipeline_parallel_size})"
            )

        expert_decoder_rank_generator = LDTRankGenerator(
            tp=expert_tensor_parallel_size,
            ep=expert_model_parallel_size,
            dp=expert_data_parallel_size,
            pp=pipeline_model_parallel_size,
            cp=1,
            order=order,
            rank_offset=encoder_world_size,
        )

        if not (
            order.endswith("pp")
            or pipeline_model_parallel_size == 1
            or expert_data_parallel_size == data_parallel_size
        ):
            raise RuntimeError("When not using pp-last rank ordering, the data parallel size of the attention and moe layers must be the same")

        if not (decoder_rank_generator.get_ranks(
            "pp"
        ) == expert_decoder_rank_generator.get_ranks(
            "pp"
        )):
            raise RuntimeError(f"Pipeline parallel groups are expected to be the same for Non-Expert and Expert part, \
        but got {decoder_rank_generator.get_ranks('pp')} and {expert_decoder_rank_generator.get_ranks('pp')}")

        def generator_wrapper(group_type, is_expert=False, **kwargs):
            """The `RankGenerator` class produces a hyper-rectangle for a given set of
            tensor, pipeline, data, expert, and context parallelism. If we have an encoder,
            in addition to the default decoder, we essentially instantiate two `RankGenerator`
            classes to construct the parallelism for each module separately, and we then have
            to stitch them together for the right groups. For now, this means pp and tp-pp.

            Let's say we have a total of 6 GPUs denoted by g0 ... g5.
            For encoder_tp=1, encoder_pp=1, decoder_tp=2, decoder_pp=1, dp=2,
            g0, g1 belong to encoder and g2, ..., g5 belong to decoder.
            The present function will create with "tp-dp-pp":
            3 data-parallel groups: [g0, g1], [g2, g4], [g3, g5]
            4 tensor model-parallel groups: [g0], [g1], [g2, g3], [g4, g5]
            4 pipeline model-parallel groups: [g0, g2], [g0, g3], [g1, g4], [g1, g5]
            """
            if is_expert:
                d_ranks = expert_decoder_rank_generator.get_ranks(group_type, **kwargs)
            else:
                d_ranks = decoder_rank_generator.get_ranks(group_type, **kwargs)

            if encoder_rank_generator is None:
                for x in d_ranks:
                    yield x
                return
            e_ranks = encoder_rank_generator.get_ranks(group_type, **kwargs)
            if group_type == "pp":
                # Map one encoder tp rank to several decoder tp ranks, because
                # encoder tp and decoder tp won't be the same size.
                # Assign this way to avoid getting the DP ranks mixed up with the PP ranks.
                # For example, if e_ranks = [0,1,2] and d_ranks = [3,4,5,6]
                # Should yield [0,3], [0,4], [1,5], [2,6]
                rep = len(d_ranks) // len(e_ranks)
                remain = len(d_ranks) % len(e_ranks)
                e_ind = 0
                e_rep = rep + int(e_ind < remain)
                for i, y in enumerate(d_ranks):
                    x = e_ranks[e_ind]
                    e_rep -= 1
                    if e_rep == 0:
                        e_ind += 1
                        e_rep = rep + int(e_ind < remain)
                    yield x + y
            elif group_type == "tp-pp":
                # For this group, we can just return the concatenated
                # groups together, because their sizes are the same.
                if len(e_ranks) != len(d_ranks):
                    raise RuntimeError("Length of encoder ranks and decoder ranks must be the same for tp-pp group")
                for x, y in zip(e_ranks, d_ranks):
                    yield x + y
            else:
                for x in e_ranks:
                    yield x
                for x in d_ranks:
                    yield x

        timeout = timedelta(minutes=distributed_timeout_minutes)

        # layerwise_disaggregated_training
        # Create VDP MP AR group
        global _PIPELINE_MODEL_PARALLEL_GROUP_FOR_VDP_CROSS_CLOUD_TP
        global _PIPELINE_MODEL_PARALLEL_GROUP_FOR_VDP_CROSS_EDGE_CLOUD
        if _PIPELINE_MODEL_PARALLEL_GROUP_FOR_VDP_CROSS_CLOUD_TP is not None:
            raise ValueError("VDP cross cloud tp group is already initialized")
        if _PIPELINE_MODEL_PARALLEL_GROUP_FOR_VDP_CROSS_EDGE_CLOUD is not None:
            raise ValueError("VDP cross edge cloud group is already initialized")
        vdp_mp_ar_ranks = [r for r in range(0, torch.distributed.get_world_size(), tensor_model_parallel_size)]
        vdp_cross_cloud_tp_ranks = vdp_mp_ar_ranks[:2]
        vdp_cross_edge_cloud_ranks = vdp_mp_ar_ranks[1:]
        vdp_cross_cloud_tp_group = create_group(ranks=vdp_cross_cloud_tp_ranks, timeout=timeout,
        pg_options=get_nccl_options('ctpg', nccl_comm_cfgs), group_desc='PIPELINE_MODEL_PARALLEL_GROUP_FOR_VDP_CROSS_CLOUD_TP')
        if rank in vdp_cross_cloud_tp_ranks:
            _PIPELINE_MODEL_PARALLEL_GROUP_FOR_VDP_CROSS_CLOUD_TP = vdp_cross_cloud_tp_group
        vdp_cross_edge_cloud_group = create_group(ranks=vdp_cross_edge_cloud_ranks, timeout=timeout,
        pg_options=get_nccl_options('ecg', nccl_comm_cfgs), group_desc='PIPELINE_MODEL_PARALLEL_GROUP_FOR_VDP_CROSS_EDGE_CLOUD')
        if rank in vdp_cross_edge_cloud_ranks:
            _PIPELINE_MODEL_PARALLEL_GROUP_FOR_VDP_CROSS_EDGE_CLOUD = vdp_cross_edge_cloud_group

        # global variables for communication stream
        global _PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE
        global _PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST
        global _PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST

        if pipeline_model_parallel_comm_backend == "ucc":
            # The UCC backend provides two key benefits:
            # 1) Achieves better bandwidth utilization than NCCL when using InfiniBand links.
            # 2) Does not use GPU SM resources (Zero-SM), mitigating performance interference
            #    with overlapping compute kernels.

            # The UCC backend is recommended in the following cases:
            # 1) When the exposed pipeline-parallel (PP) communications are significant.
            #    - E.g., Pipeline parallelism with very less gradient accumulation steps.
            #    - It may provide better performance due to improved bandwidth utilization.
            # 2) When the critical-path pipeline stage has substantial PP-communication overlap.
            #    - E.g., Uneven pipeline parallelism.
            #    - It may provide better performance due to zero SM resource usage.
            if "CUDA_DEVICE_MAX_CONNECTIONS" in os.environ:
                # UCC backend requires CUDA_DEVICE_MAX_CONNECTIONS variable to be larger than 1,
                # to gurantee the overlapped UCC communications. If this environment variable is set to 1,
                # all the UCC communication will be serialized.
                if os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] == "1":
                    raise RuntimeError("UCC-backend requires CUDA_DEVICE_MAX_CONNECTIONS > 1")

            # Setting up required environment variables for ucc backend
            #
            # "TORCH_UCC_BLOCKING_WAIT=none" allows non-blocking waits of the communiction handle
            # "UCC_EC_CUDA_STREAM_TASK_MODE" controls how CUDA execution engines (EC)
            # schedule tasks on CUDA streams.
            # "UCX_TLS" controls transport layer selection
            # "NSYS_UCP_COMM_PARAMS=1" enables capturing ucx tracing in nsys profiling
            # "UCX_RNDV_THRESH" controls threshold threshold for switching between
            # eager and rendezvous (RNDV) communication protocols.
            # "UCX_NET_DEVICES" select which network interfaces UCX should use.
            # "UCC_CL_BASIC_TLS" controls which Transport Layers are used by
            # the Basic Collective libraray

            os.environ["TORCH_UCC_BLOCKING_WAIT"] = (
                os.environ["TORCH_UCC_BLOCKING_WAIT"]
                if "TORCH_UCC_BLOCKING_WAIT" in os.environ
                else "none"
            )
            os.environ["UCC_EC_CUDA_STREAM_TASK_MODE"] = (
                os.environ["UCC_EC_CUDA_STREAM_TASK_MODE"]
                if "UCC_EC_CUDA_STREAM_TASK_MODE" in os.environ
                else "driver"
            )
            os.environ["UCX_TLS"] = (
                os.environ["UCX_TLS"] if "UCX_TLS" in os.environ else "ib,cuda_copy"
            )  # cuda_ipc (i.e., NVLink-enablement) will be later supported
            os.environ["NSYS_UCP_COMM_PARAMS"] = "1"
            os.environ["UCX_RNDV_THRESH"] = "0"
            os.environ["UCX_NET_DEVICES"] = "all"
            os.environ["UCC_CL_BASIC_TLS"] = "^sharp,nccl"

        # layerwise_disaggregated_training
        for ranks in generator_wrapper("pp"):
            # create pg for different communication streams
            group_new = create_group(
                ranks,
                timeout=timeout,
                backend=pipeline_model_parallel_comm_backend,
                pg_options=(
                    None
                    if pipeline_model_parallel_comm_backend == "ucc"
                    else get_nccl_options("pp", nccl_comm_cfgs)
                ),
                group_desc="PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE",
            )

            if not (
                pipeline_model_parallel_comm_backend is None
                or pipeline_model_parallel_comm_backend == "nccl"
                or pipeline_model_parallel_comm_backend == "ucc"
            ):
                raise RuntimeError(f'"{pipeline_model_parallel_comm_backend}" backend for PP communication is currently not supported')

            if rank in ranks:
                if _PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE is None:
                    _PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE = group_new
                    _PIPELINE_GLOBAL_RANKS_NEW_STREAM = ranks
                elif isinstance(_PIPELINE_GLOBAL_RANKS_NEW_STREAM[0], list):
                    _PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE.append(group_new)
                    _PIPELINE_GLOBAL_RANKS_NEW_STREAM.append(ranks)
                else:
                    _PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE = [
                        _PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE,
                        group_new,
                    ]
                    _PIPELINE_GLOBAL_RANKS_NEW_STREAM = [
                        _PIPELINE_GLOBAL_RANKS_NEW_STREAM,
                        ranks,
                    ]

            group_last_to_first = create_group(
                ranks,
                timeout=timeout,
                backend=pipeline_model_parallel_comm_backend,
                pg_options=(
                    None
                    if pipeline_model_parallel_comm_backend == "ucc"
                    else get_nccl_options("pp", nccl_comm_cfgs)
                ),
                group_desc="PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST",
            )

            if not (
                pipeline_model_parallel_comm_backend is None
                or pipeline_model_parallel_comm_backend == "nccl"
                or pipeline_model_parallel_comm_backend == "ucc"
            ):
                raise RuntimeError(f'"{pipeline_model_parallel_comm_backend}" backend for PP communication is currently not supported')

            if rank in ranks:
                if _PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST is None:
                    _PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST = group_last_to_first
                    _PIPELINE_GLOBAL_RANKS_LAST_TO_FIRST = ranks
                elif isinstance(_PIPELINE_GLOBAL_RANKS_LAST_TO_FIRST[0], list):
                    _PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST.append(
                        group_last_to_first
                    )
                    _PIPELINE_GLOBAL_RANKS_LAST_TO_FIRST.append(ranks)
                else:
                    _PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST = [
                        _PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST,
                        group_last_to_first,
                    ]
                    _PIPELINE_GLOBAL_RANKS_LAST_TO_FIRST = [
                        _PIPELINE_GLOBAL_RANKS_LAST_TO_FIRST,
                        ranks,
                    ]

            group_first_to_last = create_group(
                ranks,
                timeout=timeout,
                backend=pipeline_model_parallel_comm_backend,
                pg_options=(
                    None
                    if pipeline_model_parallel_comm_backend == "ucc"
                    else get_nccl_options("pp", nccl_comm_cfgs)
                ),
                group_desc="PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST",
            )

            if not (
                pipeline_model_parallel_comm_backend is None
                or pipeline_model_parallel_comm_backend == "nccl"
                or pipeline_model_parallel_comm_backend == "ucc"
            ):
                raise RuntimeError(f'"{pipeline_model_parallel_comm_backend}" backend for PP communication is currently not supported')

            if rank in ranks:
                if _PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST is None:
                    _PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST = group_first_to_last
                    _PIPELINE_GLOBAL_RANKS_FIRST_TO_LAST = ranks
                elif isinstance(_PIPELINE_GLOBAL_RANKS_FIRST_TO_LAST[0], list):
                    _PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST.append(
                        group_first_to_last
                    )
                    _PIPELINE_GLOBAL_RANKS_FIRST_TO_LAST.append(ranks)
                else:
                    _PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST = [
                        _PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST,
                        group_first_to_last,
                    ]
                    _PIPELINE_GLOBAL_RANKS_FIRST_TO_LAST = [
                        _PIPELINE_GLOBAL_RANKS_FIRST_TO_LAST,
                        ranks,
                    ]

        # VTP: initialize default (disabled) state.
        # Non-uniform VTP is fully handled by _initialize_vtp_static (early return
        # in the wrapper). This path only runs for uniform TP or no VTP.
        _init_vtp_state(False, [], [])

    return initialize_model_parallel_impl


def pre_validate_args_for_vtp(args):
    """Inflate world_size so Megatron's divisibility validation passes.

    This only affects the _validate_args pass. The wrapper (which runs after)
    uses the real world_size and calls Megatron init directly without going
    through the validation path again.

    Caller is responsible for guarding with
    ``args.layerwise_disaggregated_training`` before invoking.
    """
    world_size = getattr(args, 'world_size', None)
    if world_size is None:
        return
    tp = args.tensor_model_parallel_size
    pp = args.pipeline_model_parallel_size
    cp = getattr(args, 'context_parallel_size', 1) or 1
    expected = tp * pp * cp
    if world_size % expected == 0:
        return
    args._vtp_orig_world_size = world_size
    args.world_size = expected


def post_validate_args_for_vtp(args):
    """Restore real world_size after Megatron validation."""
    orig = getattr(args, '_vtp_orig_world_size', None)
    if orig is not None:
        args.world_size = orig
        del args._vtp_orig_world_size

    if int(os.environ['GROUP_RANK']) == 0 or int(os.environ['RANK']) == 0:
        dp_size = (int(os.environ['WORLD_SIZE']) - int(os.environ['LOCAL_WORLD_SIZE'])) // (args.context_parallel_size * args.tensor_model_parallel_size * (args.pipeline_model_parallel_size - 1))

    else:
        # GROUP_WORLD_SIZE is NNODES
        if args.pipeline_model_parallel_size == int(os.environ['GROUP_WORLD_SIZE']):
            dp_size = int(os.environ['LOCAL_WORLD_SIZE']) // (args.context_parallel_size * args.tensor_model_parallel_size)
        else:
            dp_size = int(os.environ['LOCAL_WORLD_SIZE']) // (args.context_parallel_size * args.tensor_model_parallel_size * ((args.pipeline_model_parallel_size - 1) / (int(os.environ['GROUP_WORLD_SIZE']) - 1)))

    args.data_parallel_size = int(dp_size)


def _init_vtp_state(vtp_enabled, vtp_size_list, stage_ranks):
    """Initialize VTP global state variables."""
    global _VTP_ENABLED, _VTP_SIZE_LIST, _VTP_STAGE_RANKS
    global _VTP_MY_STAGE_IDX

    _VTP_ENABLED = vtp_enabled
    _VTP_SIZE_LIST = vtp_size_list
    _VTP_STAGE_RANKS = stage_ranks

    if _VTP_ENABLED:
        print_rank_0(f'Virtual TP enabled with tp size list: {vtp_size_list}')

    rank = torch.distributed.get_rank()
    for stage_idx, stage in enumerate(stage_ranks):
        if rank in stage:
            _VTP_MY_STAGE_IDX = stage_idx
            break


def _create_vtp_groups(stage_ranks, timeout, backend):
    """Create VTP intra-stage communication group.

    PP rank0-only groups are already created during _initialize_vtp_static
    as standard PP groups (main, alternate, last-to-first, first-to-last),
    so only the intra-stage broadcast group is created here.
    """
    global _VTP_INTRA_STAGE_GROUP

    rank = torch.distributed.get_rank()

    for stage in stage_ranks:
        if len(stage) > 1:
            group = torch.distributed.new_group(
                ranks=stage, timeout=timeout, pg_options=get_nccl_options('tp', {}), group_desc='TENSOR_MODEL_PARALLEL_GROUP'
            )

            if rank in stage:
                _VTP_INTRA_STAGE_GROUP = group


# VTP getter functions
def is_vtp_enabled():
    return _VTP_ENABLED


def get_vtp_size_list():
    return _VTP_SIZE_LIST


def get_vtp_stage_ranks():
    return _VTP_STAGE_RANKS


def get_vtp_intra_stage_group():
    return _VTP_INTRA_STAGE_GROUP


def get_edge_tp_size():
    return _EDGE_TP_SIZE


def vtp_allreduce(tensor, op=torch.distributed.ReduceOp.SUM):
    """VTP-aware hierarchical allreduce.

    Replaces a flat 17-rank cross-network allreduce on model_parallel_group
    with a 3-step hierarchical reduction:
      1. Intra-stage TP allreduce  (intra-node, fast)
      2. Cross-stage PP allreduce  (rank0-only, 3 ranks)
      3. Intra-stage broadcast     (from rank0, fast)

    Mathematically correct for SUM, MAX, MIN — all are decomposable.
    """
    # Step 1: reduce within stage (TP group, intra-node)
    if mpu.get_tensor_model_parallel_world_size() > 1:
        torch.distributed.all_reduce(
            tensor, op=op, group=mpu.get_tensor_model_parallel_group()
        )

    # Step 2: reduce across stages (PP group, rank0 only — cross-network)
    if is_vtp_stage_rank0():
        torch.distributed.all_reduce(
            tensor, op=op, group=mpu.get_pipeline_model_parallel_group()
        )

    # Step 3: broadcast result to all ranks in stage
    intra_group = mpu.get_tensor_model_parallel_group()
    if intra_group is not None:
        stage_ranks = get_vtp_stage_ranks()
        my_stage = get_vtp_my_stage_idx()
        torch.distributed.broadcast(
            tensor, src=stage_ranks[my_stage][0], group=intra_group
        )


def vtp_hierarchical_barrier():
    """VTP-aware hierarchical barrier (3-step sync)."""
    # Step 1: TP barrier (intra-node)
    if mpu.get_tensor_model_parallel_world_size() > 1:
        torch.distributed.barrier(group=mpu.get_tensor_model_parallel_group())

    # Step 2: PP barrier (cross-network, rank0 only)
    if is_vtp_stage_rank0():
        torch.distributed.barrier(group=mpu.get_pipeline_model_parallel_group())

    # Step 3: Intra-stage barrier (intra-node)
    intra_group = get_vtp_intra_stage_group()
    if intra_group is not None:
        torch.distributed.barrier(group=intra_group)


def get_vtp_my_stage_idx():
    return _VTP_MY_STAGE_IDX


def is_vtp_stage_rank0():
    if not _VTP_STAGE_RANKS or _VTP_MY_STAGE_IDX is None:
        return True
    return torch.distributed.get_rank() == _VTP_STAGE_RANKS[_VTP_MY_STAGE_IDX][0]


def _auto_detect_vtp_sizes(args):
    global _EDGE_TP_SIZE

    world_size = torch.distributed.get_world_size()
    tp = args.tensor_model_parallel_size
    cp = args.context_parallel_size
    ep = args.expert_model_parallel_size
    dp = args.data_parallel_size
    pp = args.pipeline_model_parallel_size

    cloud_size = tp * cp * ep * dp * (pp - 1)
    edge_size = world_size - cloud_size
    edge_tp = edge_size // cp // ep if edge_size != tp * cp * ep * dp else tp

    _EDGE_TP_SIZE = edge_tp
    vtp_sizes = [edge_tp] + [tp] * (pp - 1)

    return vtp_sizes


def _transform_3d_list(data):
    if not data or not data[0]:
        return []

    # 1. Extract the 0th column as baseline values (create independent copies to avoid reference linkage from subsequent modifications)
    fixed_col = list(data[0][0])
    R = len(data)
    C = len(data[0]) - 1  # Number of columns in the variable portion

    # 2. Record the original lengths of each variable sublist, constructing an R x C length matrix
    lengths = [[len(data[r][c + 1]) for c in range(C)] for r in range(R)]

    # 3. Determine the starting point for incrementing sequence (take the first element of row 0, column 1)
    start_val = data[0][1][0] if lengths[0][0] > 0 else 0
    total_len = sum(sum(row_lens) for row_lens in lengths)

    # 4. Generate a globally continuous incrementing sequence
    full_seq = list(range(start_val, start_val + total_len))

    # 5. Split the sequence in column-first order and fill into temporary grid
    filled_grid = [[None] * C for _ in range(R)]
    seq_idx = 0
    for c in range(C):          # Outer: iterate over columns
        for r in range(R):      # Inner: iterate over rows
            length_val = lengths[r][c]
            filled_grid[r][c] = full_seq[seq_idx: seq_idx + length_val]
            seq_idx += length_val

    # 6. Assemble final result
    result = []
    for r in range(R):
        new_row = [list(fixed_col)]   # Uniformly replace the 0th column
        new_row.extend(filled_grid[r])
        result.append(new_row)

    return result


def find_3d_indices(data, target):
    """
    Find a target element in a 3D list and return its 3D position index.

    :param data: 3D list (allows irregular nesting)
    :param target: Target element to find
    :return: Tuple (first-level index, second-level index, innermost index); returns None if not found
    """
    if not isinstance(data, list):
        return None, None, None

    for i, layer in enumerate(data):          # First-level index
        if not isinstance(layer, list):
            continue
        for j, sublist in enumerate(layer):   # Second-level index
            if isinstance(sublist, (list, tuple)):
                try:
                    k = sublist.index(target) # Innermost index
                    return i, j, k
                except ValueError:
                    continue
    return None, None, None


def _initialize_vtp_static_vtp_vdp(fn, vtp_sizes, orig_args, orig_kwargs):
    """Initialize parallel state for static VTP with non-uniform TP sizes.

    When per-node GPU counts differ (e.g., [1, 2] for edge+cloud),
    world_size = sum(tp_sizes) * DP, which != max_tp * PP * DP.
    Megatron's standard init fails the world_size % (TP*PP) == 0 check.

    Strategy:
    1. Call Megatron's init with TP=sum(sizes), PP=1 to pass validation
    2. Override TP/PP/DP/model-parallel groups to match actual VTP layout
    3. Create LDT alternate PP groups (ping/pang, last-to-first, first-to-last)
    4. Initialize VTP state and communication groups
    """
    rank = torch.distributed.get_rank()
    vtp_model_size = sum(vtp_sizes)
    pp_size = len(vtp_sizes)

    data_parallel_size = get_vdp_size()

    modified_args = (max(vtp_sizes), pp_size, None) + orig_args[3:]
    modified_kwargs = dict(orig_kwargs)
    if 'expert_tensor_parallel_size' in modified_kwargs:
        modified_kwargs['expert_tensor_parallel_size'] = None


    # override get_world_size to support layerwise_disaggregated_training
    ori_get_world_size = torch.distributed.get_world_size

    def ldt_get_world_size(*args, **kwargs):
        stats_world_size = get_vdp_size() * max(vtp_sizes) * kwargs.get('context_parallel_size', 1) * pp_size
        return stats_world_size

    torch.distributed.get_world_size = ldt_get_world_size

    # override RankGenerator to support layerwise_disaggregated_training
    ori_rank_generator = mpu.RankGenerator
    mpu.RankGenerator = LDTRankGenerator
    ori_create_group = mpu.create_group
    mpu.create_group = create_group

    fn(*modified_args, **modified_kwargs)

    # restore get_world_size and RankGenerator
    torch.distributed.get_world_size = ori_get_world_size
    mpu.RankGenerator = ori_rank_generator
    mpu.create_group = ori_create_group

    # sync all global variables from Megatron to LDT module
    _sync_all_global_variables(mpu)

    # Build stage_ranks for each DP domain
    all_domain_stages = []
    for dp in range(data_parallel_size):
        offset = dp * vtp_model_size
        stages = []
        for tp_size in vtp_sizes:
            stages.append(list(range(offset, offset + tp_size)))
            offset += tp_size
        all_domain_stages.append(stages)

    all_domain_stages = _transform_3d_list(all_domain_stages)

    my_dp, my_stage_idx, my_intra_rank = find_3d_indices(all_domain_stages, rank)
    if my_stage_idx is None:
        raise RuntimeError(
            f"VTP static init: rank {rank} not found in any stage of domain {my_dp}. "
            f"stages={all_domain_stages}"
        )
    my_stages = all_domain_stages[my_dp]

    actual_tp = vtp_sizes[my_stage_idx]

    # Parse config for group creation
    nccl_comm_cfgs = {}
    nccl_config_path = orig_kwargs.get('nccl_communicator_config_path', None)
    if nccl_config_path:
        import yaml
        with open(nccl_config_path, 'r') as f:
            nccl_comm_cfgs = yaml.safe_load(f)

    timeout = timedelta(
        minutes=orig_kwargs.get('distributed_timeout_minutes', 30)
    )
    backend = orig_kwargs.get('pipeline_model_parallel_comm_backend', None)

    # Override TP groups (new_group is collective: all ranks must participate)
    for domain_stages in all_domain_stages:
        for stage in domain_stages:
            group = torch.distributed.new_group(stage, timeout=timeout)
            if rank in stage:
                mpu._TENSOR_MODEL_PARALLEL_GROUP = group
                mpu._TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = stage
    mpu._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = actual_tp
    mpu._MPU_TENSOR_MODEL_PARALLEL_RANK = my_intra_rank

    # Override PP groups: per-TP-intra-rank PP chains.
    # For same-TP cloud stages, each TP intra-rank has its own PP group for
    # direct P2P (e.g., [1,9], [2,10], ...). For stages with fewer TP ranks
    # (edge), the chain falls back to rank0.
    # Also create per-intra alternate groups (ping/pang double buffering).
    # L2F/F2L groups stay rank0-only (used for VTP wraparound only).
    pg_options = (
        get_nccl_options('pp', nccl_comm_cfgs)
        if backend != 'ucc' else None
    )
    global _PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE
    global _PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST
    global _PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST

    mpu._PIPELINE_MODEL_PARALLEL_GROUP = None
    mpu._PIPELINE_GLOBAL_RANKS = None
    for domain_stages in all_domain_stages:
        rank0_list = [s[0] for s in domain_stages]
        all_domain_ranks = [r for stage in domain_stages for r in stage]
        max_intra = max(len(stage) for stage in domain_stages)

        for intra in range(max_intra):
            pp_chain = []
            for stage in domain_stages:
                if intra < len(stage):
                    pp_chain.append(stage[intra])
                else:
                    pp_chain.append(stage[0])

            group = torch.distributed.new_group(
                pp_chain, timeout=timeout,
                backend=backend, pg_options=pg_options,
            )
            group_alt = torch.distributed.new_group(
                pp_chain, timeout=timeout,
                backend=backend, pg_options=pg_options,
            )

            if rank in pp_chain:
                is_rank0 = rank in rank0_list
                # rank0 members keep the intra=0 (rank0-only) groups;
                # non-rank0 members get their TP-peer PP group.
                if intra == 0 or not is_rank0:
                    if int(os.environ['GROUP_RANK']) == 0 or int(os.environ['RANK']) == 0:  # Edge side
                        if mpu._PIPELINE_MODEL_PARALLEL_GROUP is None:
                            mpu._PIPELINE_MODEL_PARALLEL_GROUP = group
                            mpu._PIPELINE_GLOBAL_RANKS = pp_chain
                            _PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE = group_alt
                        elif isinstance(mpu._PIPELINE_MODEL_PARALLEL_GROUP, list):
                            mpu._PIPELINE_MODEL_PARALLEL_GROUP.append(group)
                            mpu._PIPELINE_GLOBAL_RANKS.append(pp_chain)
                            _PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE.append(group_alt)
                        else:
                            mpu._PIPELINE_MODEL_PARALLEL_GROUP = [mpu._PIPELINE_MODEL_PARALLEL_GROUP, group]
                            mpu._PIPELINE_GLOBAL_RANKS = [mpu._PIPELINE_GLOBAL_RANKS, pp_chain]
                            _PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE = [_PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE, group_alt]
                    else:
                        mpu._PIPELINE_MODEL_PARALLEL_GROUP = group
                        mpu._PIPELINE_GLOBAL_RANKS = pp_chain
                        _PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE = group_alt


        # L2F/F2L groups: rank0-only, for VTP wraparound (edge<->last cloud).
        # Set for all domain ranks so accessors don't crash; non-rank0 ranks
        # hold a reference but never communicate through them (VTP guards).
        group_l2f = torch.distributed.new_group(
            rank0_list, timeout=timeout, backend=backend, pg_options=pg_options,
        )
        group_f2l = torch.distributed.new_group(
            rank0_list, timeout=timeout, backend=backend, pg_options=pg_options,
        )
        if rank in all_domain_ranks:
            if int(os.environ['GROUP_RANK']) == 0 or int(os.environ['RANK']) == 0:  # Edge side
                if _PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST is None:
                    _PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST = group_l2f
                    _PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST = group_f2l
                elif isinstance(_PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST, list):
                    _PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST.append(group_l2f)
                    _PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST.append(group_f2l)
                else:
                    _PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST = [_PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST, group_l2f]
                    _PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST = [_PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST, group_f2l]
            else:
                _PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST = group_l2f
                _PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST = group_f2l

    mpu._MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = pp_size
    mpu._MPU_PIPELINE_MODEL_PARALLEL_RANK = my_stage_idx

    # Override model-parallel group (all ranks in one DP domain)
    for domain_stages in all_domain_stages:
        all_ranks = [r for stage in domain_stages for r in stage]
        group = torch.distributed.new_group(all_ranks, timeout=timeout)
        if rank in all_ranks:
            mpu._MODEL_PARALLEL_GROUP = group
            mpu._MODEL_PARALLEL_GLOBAL_RANKS = all_ranks

    # Update args to reflect actual parallelism for this rank
    args = get_args()
    args.tensor_model_parallel_size = actual_tp
    args.data_parallel_size = data_parallel_size

    # Restore VPP that was cleared for the Megatron init call (VPP=None to
    # avoid PP>1 assertion with PP=1). LDT u-shaped needs VPP so that
    # is_pipeline_first_stage() returns False when building non-first
    # virtual stage models on PP rank 0.
    orig_vpp = orig_args[2] if len(orig_args) > 2 else None
    if orig_vpp is not None:
        mpu._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = orig_vpp
        mpu._VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0

    # Initialize VTP state
    _init_vtp_state(True, vtp_sizes, my_stages)

    # Create VTP communication groups
    _create_vtp_groups(my_stages, timeout, backend)


def _initialize_vtp_static_only_vtp(fn, vtp_sizes, orig_args, orig_kwargs):
    """Initialize parallel state for static VTP with non-uniform TP sizes.

    When per-node GPU counts differ (e.g., [1, 2] for edge+cloud),
    world_size = sum(tp_sizes) * DP, which != max_tp * PP * DP.
    Megatron's standard init fails the world_size % (TP*PP) == 0 check.

    Strategy:
    1. Call Megatron's init with TP=sum(sizes), PP=1 to pass validation
    2. Override TP/PP/DP/model-parallel groups to match actual VTP layout
    3. Create LDT alternate PP groups (ping/pang, last-to-first, first-to-last)
    4. Initialize VTP state and communication groups
    """
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    vtp_model_size = sum(vtp_sizes)
    pp_size = len(vtp_sizes)

    if world_size % vtp_model_size != 0:
        raise RuntimeError(
            f"VTP static: world_size ({world_size}) is not divisible by "
            f"sum(vtp_sizes) ({vtp_model_size})"
        )
    data_parallel_size = world_size // vtp_model_size

    # Call Megatron's init with TP=sum, PP=1, VPP=None to pass validation.
    # This creates basic distributed state with "wrong" groups that we override below.
    # Also reset expert_tensor_parallel_size so Megatron defaults it to the
    # modified TP (=vtp_model_size), otherwise the original TP=max_tp value
    # causes decoder_world_size % expert_tp_pp check to fail.
    modified_args = (vtp_model_size, 1, None) + orig_args[3:]
    modified_kwargs = dict(orig_kwargs)
    if 'expert_tensor_parallel_size' in modified_kwargs:
        modified_kwargs['expert_tensor_parallel_size'] = None
    fn(*modified_args, **modified_kwargs)

    # Build stage_ranks for each DP domain
    all_domain_stages = []
    for dp in range(data_parallel_size):
        offset = dp * vtp_model_size
        stages = []
        for tp_size in vtp_sizes:
            stages.append(list(range(offset, offset + tp_size)))
            offset += tp_size
        all_domain_stages.append(stages)

    # Find current rank's position
    my_dp = rank // vtp_model_size
    my_stages = all_domain_stages[my_dp]
    my_stage_idx = None
    my_intra_rank = None
    for idx, stage in enumerate(my_stages):
        if rank in stage:
            my_stage_idx = idx
            my_intra_rank = stage.index(rank)
            break

    if my_stage_idx is None:
        raise RuntimeError(
            f"VTP static init: rank {rank} not found in any stage of domain {my_dp}. "
            f"stages={my_stages}"
        )
    actual_tp = vtp_sizes[my_stage_idx]

    # Parse config for group creation
    nccl_comm_cfgs = {}
    nccl_config_path = orig_kwargs.get('nccl_communicator_config_path', None)
    if nccl_config_path:
        import yaml
        with open(nccl_config_path, 'r') as f:
            nccl_comm_cfgs = yaml.safe_load(f)

    timeout = timedelta(
        minutes=orig_kwargs.get('distributed_timeout_minutes', 30)
    )
    backend = orig_kwargs.get('pipeline_model_parallel_comm_backend', None)

    # Override TP groups (new_group is collective: all ranks must participate)
    for domain_stages in all_domain_stages:
        for stage in domain_stages:
            group = torch.distributed.new_group(stage, timeout=timeout)
            if rank in stage:
                mpu._TENSOR_MODEL_PARALLEL_GROUP = group
                mpu._TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = stage
    mpu._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = actual_tp
    mpu._MPU_TENSOR_MODEL_PARALLEL_RANK = my_intra_rank

    # Override PP groups: per-TP-intra-rank PP chains.
    # For same-TP cloud stages, each TP intra-rank has its own PP group for
    # direct P2P (e.g., [1,9], [2,10], ...). For stages with fewer TP ranks
    # (edge), the chain falls back to rank0.
    # Also create per-intra alternate groups (ping/pang double buffering).
    # L2F/F2L groups stay rank0-only (used for VTP wraparound only).
    pg_options = (
        get_nccl_options('pp', nccl_comm_cfgs)
        if backend != 'ucc' else None
    )
    global _PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE
    global _PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST
    global _PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST
    for domain_stages in all_domain_stages:
        rank0_list = [s[0] for s in domain_stages]
        all_domain_ranks = [r for stage in domain_stages for r in stage]
        max_intra = max(len(stage) for stage in domain_stages)

        for intra in range(max_intra):
            pp_chain = []
            for stage in domain_stages:
                if intra < len(stage):
                    pp_chain.append(stage[intra])
                else:
                    pp_chain.append(stage[0])

            group = torch.distributed.new_group(
                pp_chain, timeout=timeout,
                backend=backend, pg_options=pg_options,
            )
            group_alt = torch.distributed.new_group(
                pp_chain, timeout=timeout,
                backend=backend, pg_options=pg_options,
            )

            if rank in pp_chain:
                is_rank0 = rank in rank0_list
                # rank0 members keep the intra=0 (rank0-only) groups;
                # non-rank0 members get their TP-peer PP group.
                if intra == 0 or not is_rank0:
                    mpu._PIPELINE_MODEL_PARALLEL_GROUP = group
                    mpu._PIPELINE_GLOBAL_RANKS = pp_chain
                    _PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE = group_alt

        # L2F/F2L groups: rank0-only, for VTP wraparound (edge<->last cloud).
        # Set for all domain ranks so accessors don't crash; non-rank0 ranks
        # hold a reference but never communicate through them (VTP guards).
        group_l2f = torch.distributed.new_group(
            rank0_list, timeout=timeout, backend=backend, pg_options=pg_options,
        )
        group_f2l = torch.distributed.new_group(
            rank0_list, timeout=timeout, backend=backend, pg_options=pg_options,
        )
        if rank in all_domain_ranks:
            _PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST = group_l2f
            _PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST = group_f2l

    mpu._MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = pp_size
    mpu._MPU_PIPELINE_MODEL_PARALLEL_RANK = my_stage_idx

    # Override model-parallel group (all ranks in one DP domain)
    for domain_stages in all_domain_stages:
        all_ranks = [r for stage in domain_stages for r in stage]
        group = torch.distributed.new_group(all_ranks, timeout=timeout)
        if rank in all_ranks:
            mpu._MODEL_PARALLEL_GROUP = group
            mpu._MODEL_PARALLEL_GLOBAL_RANKS = all_ranks

    # Override DP groups if DP > 1
    if data_parallel_size > 1:
        create_gloo = orig_kwargs.get('create_gloo_process_groups', True)
        for stage_idx in range(pp_size):
            for intra in range(vtp_sizes[stage_idx]):
                dp_ranks = [
                    all_domain_stages[dp][stage_idx][intra]
                    for dp in range(data_parallel_size)
                ]
                g_nccl = torch.distributed.new_group(
                    dp_ranks, timeout=timeout
                )
                g_gloo = (
                    torch.distributed.new_group(
                        dp_ranks, timeout=timeout, backend='gloo'
                    )
                    if create_gloo else None
                )
                if rank in dp_ranks:
                    mpu._DATA_PARALLEL_GROUP = g_nccl
                    mpu._DATA_PARALLEL_GROUP_GLOO = g_gloo
                    mpu._DATA_PARALLEL_GLOBAL_RANKS = dp_ranks
        mpu._MPU_DATA_PARALLEL_WORLD_SIZE = data_parallel_size
        mpu._MPU_DATA_PARALLEL_RANK = my_dp

    # Update args to reflect actual parallelism for this rank
    args = get_args()
    args.tensor_model_parallel_size = actual_tp
    args.data_parallel_size = data_parallel_size

    # Restore VPP that was cleared for the Megatron init call (VPP=None to
    # avoid PP>1 assertion with PP=1). LDT u-shaped needs VPP so that
    # is_pipeline_first_stage() returns False when building non-first
    # virtual stage models on PP rank 0.
    orig_vpp = orig_args[2] if len(orig_args) > 2 else None
    if orig_vpp is not None:
        mpu._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = orig_vpp
        mpu._VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0

    # Initialize VTP state
    _init_vtp_state(True, vtp_sizes, my_stages)

    # Create VTP communication groups
    _create_vtp_groups(my_stages, timeout, backend)


# VDP scenario
def create_group(
    ranks=None,
    timeout=None,
    backend=None,
    pg_options=None,
    use_local_synchronization=False,
    group_desc=None,
):
    """Creates a ProcessGroup."""
    # add layerwise disaggregated training support
    if _LAYERWISE_DISAGGREGATED_TRAINING:
        kwargs = {
            'ranks': list(set(ranks)),
            'timeout': timeout,
            'backend': backend,
            'pg_options': pg_options,
            'use_local_synchronization': use_local_synchronization,
            'group_desc': group_desc,
        }
    else:
        kwargs = {
            'ranks': ranks,
            'timeout': timeout,
            'backend': backend,
            'pg_options': pg_options,
            'use_local_synchronization': use_local_synchronization,
            'group_desc': group_desc,
        }
    if not is_torch_min_version('2.4.0'):
        kwargs.pop('group_desc')
        if timeout is None:
            # Old version (e.g. v2.1.2) sets default_pg_timeout as default value to timeout
            # in function signature, then check tiemout value type.
            # New version sets None as default value to timeout in function signature. If value
            # is None, torch will give value according to the backend, then check type.
            # So need to unset timeout here if caller doesn't set value. Otherwise there is
            # type error.
            kwargs.pop('timeout')
    return torch.distributed.new_group(**kwargs)


def transform_x_dimension(data: Union[List[List[int]], np.ndarray], x_dim: int) -> np.ndarray:
    """
    Stream chunking and flattening transformation of 2D array along the specified X-axis dimension, strictly preserving original shape.

    Business logic:
      1. Flatten the original data into a 1D continuous sequence
      2. Split into continuous chunks by x_dim
      3. Take the first `rows` chunks and assign them to corresponding rows
      4. Repeat each chunk horizontally to fill the original number of columns
    """
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError("Input must be a 2D array")

    rows, cols = arr.shape
    if not (0 < x_dim <= cols):
        raise ValueError(f"x_dim must be greater than 0 and not exceed the original column count {cols}")
    if cols % x_dim != 0:
        raise ValueError(f"To maintain periodic tiling, the original column count {cols} must be divisible by x_dim {x_dim}")

    # 1. Flatten + reshape into continuous block view by x_dim (zero-copy)
    blocks = arr.ravel().reshape(-1, x_dim)

    # 2. Take the first rows blocks (to ensure output row count remains unchanged)
    selected = blocks[:rows]

    # 3. Tile horizontally to original column width while maintaining shape
    result = np.tile(selected, reps=(1, cols // x_dim))

    return result


# VDP scenario
class LDTRankGenerator(RankGenerator):
    def __init__(self, tp: int, ep: int, dp: int, pp: int, cp: int, order: str, rank_offset: int = 0):
        super().__init__(tp, ep, dp, pp, cp, order, rank_offset)

    def get_ranks(self, token):
        """Get rank group by input token.

        Args:
            token (str):
                Specify the ranks type that want to get. If we want
                to obtain multiple parallel types, we can use a hyphen
                '-' to separate them. For example, if we want to obtain
                the TP_DP group, the token should be 'tp-dp'.
        """
        def add_value_to_array(arr, value):
            """Add value to each element in the numpy array."""
            arr = np.array(arr)
            return arr + value

        def mod_array(arr, mod_value):
            if mod_value == 0:
                raise ValueError("mod_value must be greater than 0.")
            arr = np.array(arr)
            return np.mod(arr, mod_value)

        def get_edge_card_size():
            """Get edge card size."""
            edge_card_size = 0
            # Get edge_card_size on rank 0 and broadcast to other nodes
            if int(os.environ.get("GROUP_RANK", -1)) == 0 or int(os.environ.get("RANK", -1)) == 0:
                edge_card_size = int(os.environ.get("LOCAL_WORLD_SIZE", 0))
                if is_vtp_enabled():
                    edge_card_size = edge_card_size * self.tp

                # Create a list for broadcasting
                broadcast_list = [edge_card_size]
                torch.distributed.broadcast_object_list(broadcast_list, src=0)
                # Update edge_card_size for the source process
                edge_card_size = broadcast_list[0]
            else:
                # Create a list for receiving broadcast
                broadcast_list = [0]
                torch.distributed.broadcast_object_list(broadcast_list, src=0)
                # Get edge_card_size from broadcast
                edge_card_size = broadcast_list[0]

            return edge_card_size

        if _LAYERWISE_DISAGGREGATED_TRAINING:
            edge_card_size = get_edge_card_size()
            edge_ranks = RankGenerator(self.tp, self.ep, self.dp, 1, self.cp, self.order).get_ranks(token)
            edge_ranks = mod_array(edge_ranks, edge_card_size)

            if token.find("tp") > -1:
                edge_ranks_vtp = transform_x_dimension(edge_ranks, get_edge_tp_size())
            else:
                edge_ranks_vtp = edge_ranks

            cloud_ranks = RankGenerator(self.tp, self.ep, self.dp, self.pp - 1, self.cp, self.order).get_ranks(token)
            cloud_ranks = add_value_to_array(cloud_ranks, edge_card_size)

            if token.find("pp") > -1:
                ranks = np.concatenate((edge_ranks, cloud_ranks), axis=1).tolist()
            else:
                ranks = np.concatenate((edge_ranks, cloud_ranks), axis=0).tolist()
        else:
            ranks = super().get_ranks(token)

        return ranks


def _init_vdp_state(tensor_model_parallel_size, context_parallel_size, vdp_size, vtp_sizes):
    global _VDP_ENABLED

    if vtp_sizes:
        edge_tp_size = vtp_sizes[0]
    else:
        edge_tp_size = tensor_model_parallel_size

    if int(os.environ['GROUP_RANK']) == 0 or int(os.environ['RANK']) == 0:
        if int(os.environ['LOCAL_WORLD_SIZE']) % (context_parallel_size * edge_tp_size) == 0:
            edge_dp_size = int(os.environ['LOCAL_WORLD_SIZE']) // (context_parallel_size * edge_tp_size)
            if edge_dp_size != vdp_size:
                _VDP_ENABLED = True
            else:
                _VDP_ENABLED = False
        else:
            _VDP_ENABLED = True
    else:
        # GROUP_WORLD_SIZE is NNODES
        edge_world_size = int(os.environ['WORLD_SIZE']) - int(os.environ['LOCAL_WORLD_SIZE']) * (int(os.environ['GROUP_WORLD_SIZE']) - 1)
        if edge_world_size % (context_parallel_size * edge_tp_size) == 0:
            edge_dp_size = edge_world_size // (context_parallel_size * edge_tp_size)
            if edge_dp_size != vdp_size:
                _VDP_ENABLED = True
            else:
                _VDP_ENABLED = False
        else:
            _VDP_ENABLED = True


def get_pipeline_model_parallel_group_alternate():
    """Get the alternate pipeline model parallel communication group.

    This function returns the alternate pipeline model parallel group used for
    double-buffering communication in pipeline parallel training. It works in
    conjunction with the default pipeline model parallel group to enable
    efficient alternating communication streams.

    Returns:
        torch.distributed.ProcessGroup or list[torch.distributed.ProcessGroup]:
            The alternate pipeline model parallel communication group(s).
            Returns a list if the current rank belongs to multiple pipeline groups.

    Raises:
        RuntimeError: If the pipeline model parallel group is not initialized.

    Note:
        - This group is used in double-buffering communication to improve performance
        - It is typically used alongside the default pipeline model parallel group
        - The two groups are alternated based on the pipeline parallel rank parity
    """
    if not (
        _PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE is not None
    ):
        raise RuntimeError("pipeline_model parallel group is not initialized")

    return _PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE


def get_pipeline_model_parallel_group_last_to_first():
    """Get the pipeline model parallel communication group for last-to-first direction.

    This function returns the pipeline model parallel group used for communication
    in the last-to-first direction. It is typically used when the pipeline parallel
    world size is odd, requiring additional communication streams for the first
    and last stages.

    Returns:
        torch.distributed.ProcessGroup or list[torch.distributed.ProcessGroup]:
            The pipeline model parallel communication group(s) for last-to-first direction.
            Returns a list if the current rank belongs to multiple pipeline groups.

    Raises:
        RuntimeError: If the pipeline model parallel group is not initialized.

    Note:
        - This group is used for communication from last stage to first stage
        - It is primarily used when pipeline parallel world size is odd
        - Used to handle edge cases in U-shaped pipeline parallelism
    """
    if not (
        _PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST is not None
    ):
        raise RuntimeError("pipeline_model parallel group is not initialized")

    return _PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST


def get_pipeline_model_parallel_group_first_to_last():
    """Get the pipeline model parallel communication group for first-to-last direction.

    This function returns the pipeline model parallel group used for communication
    in the first-to-last direction. It is typically used when the pipeline parallel
    world size is odd, requiring additional communication streams for the first
    and last stages.

    Returns:
        torch.distributed.ProcessGroup or list[torch.distributed.ProcessGroup]:
            The pipeline model parallel communication group(s) for first-to-last direction.
            Returns a list if the current rank belongs to multiple pipeline groups.

    Raises:
        RuntimeError: If the pipeline model parallel group is not initialized.

    Note:
        - This group is used for communication from first stage to last stage
        - It is primarily used when pipeline parallel world size is odd
        - Used to handle edge cases in U-shaped pipeline parallelism
    """
    if not (
        _PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST is not None
    ):
        raise RuntimeError("pipeline_model parallel group is not initialized")

    return _PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST


def _sync_all_global_variables(megatron_mpu):
    """Sync all global variables in megatron_mpu."""
    from mindspeed_mm.patchs.layerwise_disaggregated_training import parallel_state_patch as ldt_mpu

    global_variable_names = [
        # core parallel groups
        '_TENSOR_MODEL_PARALLEL_GROUP',
        '_PIPELINE_MODEL_PARALLEL_GROUP',
        '_MODEL_PARALLEL_GROUP',
        '_EMBEDDING_GROUP',
        '_POSITION_EMBEDDING_GROUP',
        '_DATA_PARALLEL_GROUP',
        '_DATA_PARALLEL_GROUP_GLOO',
        '_TENSOR_AND_DATA_PARALLEL_GROUP',

        # Expert-related parallel states
        '_EXPERT_MODEL_PARALLEL_GROUP',
        '_EXPERT_TENSOR_PARALLEL_GROUP',
        '_EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP',
        '_EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP',
        '_EXPERT_DATA_PARALLEL_GROUP',
        '_EXPERT_DATA_PARALLEL_GROUP_GLOO',
        '_MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE',
        '_MPU_EXPERT_MODEL_PARALLEL_RANK',
        '_MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE',
        '_MPU_EXPERT_TENSOR_PARALLEL_RANK',

        # Virtual pipeline parallel
        '_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK',
        '_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE',
        '_PIPELINE_MODEL_PARALLEL_SPLIT_RANK',
        '_PIPELINE_MODEL_PARALLEL_DECODER_START',

        # MPU dynamic values
        '_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE',
        '_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE',
        '_MPU_DATA_PARALLEL_WORLD_SIZE',
        '_MPU_DATA_PARALLEL_RANK',
        '_MPU_TENSOR_MODEL_PARALLEL_RANK',
        '_MPU_PIPELINE_MODEL_PARALLEL_RANK',

        # Global ranks lists
        '_EMBEDDING_GLOBAL_RANKS',
        '_POSITION_EMBEDDING_GLOBAL_RANKS',
        '_PIPELINE_GLOBAL_RANKS',
        '_DATA_PARALLEL_GLOBAL_RANKS',
        '_TENSOR_MODEL_PARALLEL_GLOBAL_RANKS',
        '_MODEL_PARALLEL_GLOBAL_RANKS',

        # Context parallel
        '_CONTEXT_PARALLEL_GROUP',
        '_CONTEXT_PARALLEL_GLOBAL_RANKS',
        '_HIERARCHICAL_CONTEXT_PARALLEL_GROUP',

        # Combined parallel groups
        '_DATA_PARALLEL_GROUP_WITH_CP',
        '_DATA_PARALLEL_GROUP_WITH_CP_GLOO',
        '_DATA_PARALLEL_GLOBAL_RANKS_WITH_CP',
        '_INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP',
        '_INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO',
        '_INTER_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP',
        '_TENSOR_AND_CONTEXT_PARALLEL_GROUP',
        '_TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP',

        # Additional LDT_specific groups
        '_PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE',
        '_PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST',
        '_PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST',

        # Memory buffers
        '_GLOBAL_MEMORY_BUFFER',

        # MOE logging
        '_MOE_LAYER_WISE_LOGGING_TRACKER',
    ]

    # Sync each global variable
    for var_name in global_variable_names:
        if hasattr(megatron_mpu, var_name):
            megatron_value = getattr(megatron_mpu, var_name)
            setattr(ldt_mpu, var_name, megatron_value)


def get_layerwise_disaggregated_training():
    return _LAYERWISE_DISAGGREGATED_TRAINING


def is_vdp_enabled():
    return _VDP_ENABLED


def get_vdp_size():
    """get the size of the virtual data parallel group"""
    return _VDP_SIZE


def get_pipeline_model_parallel_group_for_vdp_cross_cloud_tp():
    return _PIPELINE_MODEL_PARALLEL_GROUP_FOR_VDP_CROSS_CLOUD_TP


def get_pipeline_model_parallel_group_for_vdp_cross_edge_cloud():
    return _PIPELINE_MODEL_PARALLEL_GROUP_FOR_VDP_CROSS_EDGE_CLOUD
