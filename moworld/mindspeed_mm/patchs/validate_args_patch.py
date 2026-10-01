# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
import os
import warnings
import torch

from mindspeed.arguments import validate_args_wrapper
from megatron.core.rerun_state_machine import RerunStateMachine
from megatron.core.transformer.enums import AttnBackend
from megatron.core.utils import get_torch_version, is_torch_min_version
from mindspeed.patch_utils import MindSpeedPatchesManager as pm
from megatron.training.arguments import load_retro_args, _check_arg_is_not_none, _print_args
from megatron.training import print_rank_0
from megatron.training.utils import get_device_arch_version, update_use_dist_ckpt

from mindspeed_mm.configs.config import merge_mm_args
from mindspeed_mm.utils.utils import ensure_valid
from mindspeed_mm.patchs.layerwise_disaggregated_training.parallel_state_patch import (
    pre_validate_args_for_vtp,
    post_validate_args_for_vtp,
    install_ldt_init_model_parallel_patch,
)


def safe_getattr(mm_object, mm_name, mm_default_value):
    # If mm_object.mm_name != mm_default_value, log and prompt user that the actual used value is overwritten
    mm_value = getattr(mm_object, mm_name, mm_default_value)

    if mm_value != mm_default_value:
        print_rank_0(f'[INFO] the original value of {mm_name} is {mm_default_value}, now changed as {mm_value} which comes from model.json')
    return mm_value


def validate_args(args, defaults=None):

    if defaults is None:
        defaults = {}

    # merge mm config to args
    merge_mm_args(args)

    # Propagate JSON patch flags to args so downstream code (e.g. VTP
    # pre/post validation, init_model_parallel_wrapper) can read them without
    # requiring a duplicate CLI flag.
    if hasattr(args.mm.model, 'patch'):
        args.layerwise_disaggregated_training = bool(
            getattr(args.mm.model.patch, 'layerwise_disaggregated_training', False)
        )
    # use model.json to fill args
    if hasattr(args.mm.model, 'text_decoder'):
        args.num_layers = safe_getattr(args.mm.model.text_decoder, 'num_layers', args.num_layers)
        args.hidden_size = safe_getattr(args.mm.model.text_decoder, 'hidden_size', args.hidden_size)
        args.num_attention_heads = safe_getattr(args.mm.model.text_decoder, 'num_attention_heads', args.num_attention_heads)
        args.num_query_groups = safe_getattr(args.mm.model.text_decoder, 'num_query_groups', args.num_query_groups)
        args.max_position_embeddings = safe_getattr(args.mm.model.text_decoder, 'max_position_embeddings', args.max_position_embeddings)
        args.ffn_hidden_size = safe_getattr(args.mm.model.text_decoder, 'ffn_hidden_size', args.ffn_hidden_size)

        # MOE
        if hasattr(args.mm.model.text_decoder, 'num_moe_experts'):
            args.num_experts = safe_getattr(args.mm.model.text_decoder, 'num_moe_experts', args.num_experts)
            args.n_shared_experts = safe_getattr(args.mm.model.text_decoder, 'n_shared_experts', args.n_shared_experts)
            args.mm.model.text_decoder.moe_token_dispatcher_type = safe_getattr(args.mm.model.text_decoder, 'moe_token_dispatcher_type', args.moe_token_dispatcher_type)

            args.add_bias_linear = safe_getattr(args.mm.model.text_decoder, 'add_bias_linear', True)
            args.mm.model.text_decoder.v_head_dim = safe_getattr(args, 'v_head_dim', 0)
            args.rope_scaling_type = safe_getattr(args.mm.model.text_decoder, 'rope_scaling_type', None)

    # use model.json to fill predictor arg
    if hasattr(args.mm.model, 'predictor'):
        if hasattr(args.mm.model.predictor, 'mm_single_blocks_depth') and hasattr(args.mm.model.predictor, 'mm_double_blocks_depth'):
            mm_double_blocks_depth = getattr(args.mm.model.predictor, 'mm_double_blocks_depth', 0)
            if mm_double_blocks_depth <= 0:
                raise AssertionError(f"MindSpeed-MM Error: mm_double_blocks_depth must > 0, actually:{mm_double_blocks_depth}")
            args.num_layers = max(safe_getattr(args.mm.model.predictor, 'mm_single_blocks_depth', args.num_layers), mm_double_blocks_depth)
        args.num_layers = safe_getattr(args.mm.model.predictor, 'num_layers', args.num_layers)
        # Some models have num_layers as a list in the predictor, so sum it up to get total layers
        if isinstance(args.num_layers, (tuple, list)):
            args.num_layers = sum(args.num_layers)

        args.num_attention_heads = safe_getattr(args.mm.model.predictor, 'num_heads', args.num_attention_heads)
        head_dim = getattr(args.mm.model.predictor, 'head_dim', 1)
        if isinstance(head_dim, int):
            hidden_size = args.num_attention_heads * head_dim
            print_rank_0(f"[INFO] the original value of normalization is {args.hidden_size}, now changed as {hidden_size} which comes from model.json")
            args.hidden_size = hidden_size
        args.hidden_size = safe_getattr(args.mm.model.predictor, 'hidden_size', args.hidden_size)
        args.attention_dropout = safe_getattr(args.mm.model.predictor, 'dropout', args.attention_dropout)
        args.hidden_dropout = safe_getattr(args.mm.model.predictor, 'hidden_dropout', args.hidden_dropout)
        args.swiglu = safe_getattr(args.mm.model.predictor, 'swiglu', args.swiglu)
        args.masked_softmax_fusion = safe_getattr(args.mm.model.predictor, 'masked_softmax_fusion', args.masked_softmax_fusion)
        norm_type = getattr(args.mm.model.predictor, 'qk_norm_type', "")
        if isinstance(norm_type, str) and norm_type == "rmsnorm":
            print_rank_0(f"[INFO] the original value of normalization is {args.normalization}, now changed as RMSNorm which comes from model.json")
            args.normalization = "RMSNorm"
            args.use_fused_rmsnorm = safe_getattr(args.mm.model.predictor, 'use_fused_rmsnorm', args.use_fused_rmsnorm)
        args.seq_length = safe_getattr(args.mm.model.predictor, 'seq_length', 3072)
        args.max_position_embeddings = safe_getattr(args.mm.model.predictor, 'max_position_embeddings', args.seq_length)
        args.position_embedding_type = safe_getattr(args.mm.model.predictor, 'position_embedding_type', 'rope')
        args.rotary_base = safe_getattr(args.mm.model.predictor, 'rotary_base', 500000)
        args.tokenizer_type = safe_getattr(args.mm.model.predictor, 'tokenizer_type', "NullTokenizer")
        args.vocab_size = safe_getattr(args.mm.model.predictor, "vocab_size", 0)

    # use default value to fill feature_extration arg
    elif hasattr(args.mm.model, 'ae'):
        args.num_layers = 1
        args.hidden_size = 3072
        args.num_attention_heads = 48
        args.seq_length = 24
        args.attention_dropout = 0.0
        args.hidden_dropout = 0.0
        args.swiglu = True
        args.masked_softmax_fusion = False
        args.max_position_embeddings = 24
        args.position_embedding_type = "rope"
        args.rotary_base = 500000
        args.tokenizer_type = 'NullTokenizer'
        args.vocab_size = 0

    # Load saved args from Retro (if applicable).
    load_retro_args(args)

    # Set args.use_dist_ckpt from args.ckpt_format.
    if args.use_legacy_models:
        ensure_valid(args.ckpt_format == "torch", \
            "legacy model format only supports the 'torch' checkpoint format.")
    update_use_dist_ckpt(args)

    if args.encoder_pipeline_model_parallel_size == 0 and args.num_experts == 0:
        ensure_valid(args.encoder_tensor_model_parallel_size == args.tensor_model_parallel_size, "If non-MOE encoder shares first decoder pipeline rank it must have the same TP as the decoder.")

    if args.encoder_tensor_model_parallel_size > 0:
        ensure_valid(args.num_attention_heads % args.encoder_tensor_model_parallel_size == 0)
        ensure_valid(args.encoder_tensor_model_parallel_size <= args.tensor_model_parallel_size, "We do not support encoders with more TP than the decoder.")

    if args.encoder_pipeline_model_parallel_size > 0 and args.encoder_tensor_model_parallel_size == 0:
        args.encoder_tensor_model_parallel_size = args.tensor_model_parallel_size

    # VTP: temporarily inflate world_size so the divisibility check passes
    if getattr(args, 'layerwise_disaggregated_training', False):
        pre_validate_args_for_vtp(args)
    encoder_model_size = args.encoder_tensor_model_parallel_size * args.encoder_pipeline_model_parallel_size * args.context_parallel_size
    decoder_model_size = args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size
    total_model_size = encoder_model_size + decoder_model_size

    # Total model size.
    ensure_valid(args.world_size % total_model_size == 0,
        f"world size ({args.world_size}) is not divisible by total_model_size ({encoder_model_size=} + {decoder_model_size=})"
    )

    if args.attention_backend == AttnBackend.local:
        ensure_valid(args.spec[0] == 'local', '--attention-backend local is only supported with --spec local')

    # Pipeline model parallel size.
    args.transformer_pipeline_model_parallel_size = args.pipeline_model_parallel_size

    args.data_parallel_size = args.world_size // total_model_size

    # VTP: restore real world_size after validation
    if getattr(args, 'layerwise_disaggregated_training', False):
        post_validate_args_for_vtp(args)
        install_ldt_init_model_parallel_patch()

    if args.rank == 0:
        print('using world size: {}, data-parallel size: {}, '
              'context-parallel size: {}, '
              'hierarchical context-parallel sizes: {}'
              'tensor-model-parallel size: {}, '
              'encoder-tensor-model-parallel size: {}, '
              'pipeline-model-parallel size: {}, '
              'encoder-pipeline-model-parallel size: {}'.format(
                  args.world_size, args.data_parallel_size,
                  args.context_parallel_size,
                  args.hierarchical_context_parallel_sizes,
                  args.tensor_model_parallel_size,
                  args.encoder_tensor_model_parallel_size,
                  args.pipeline_model_parallel_size,
                  args.encoder_pipeline_model_parallel_size), flush=True)

    # Checks.

    # Backwards compatibility.
    if args.pipeline_model_parallel_split_rank is not None:
        args.encoder_pipeline_model_parallel_size = args.pipeline_model_parallel_split_rank
        args.pipeline_model_parallel_size -= args.encoder_pipeline_model_parallel_size
        ensure_valid(args.pipeline_model_parallel_size > 0)

    if args.hierarchical_context_parallel_sizes:
        from numpy import prod
        ensure_valid(args.context_parallel_size == prod(args.hierarchical_context_parallel_sizes))
    if "a2a+p2p" in args.cp_comm_type:
        ensure_valid(args.hierarchical_context_parallel_sizes is not None, \
        "--hierarchical-context-parallel-sizes must be set when a2a+p2p is used in cp comm")

    if args.expert_tensor_parallel_size is None:
        args.expert_tensor_parallel_size = args.tensor_model_parallel_size

    # Deprecated arguments.
    ensure_valid(args.batch_size is None, '--batch-size argument is no longer ' \
        'valid, use --micro-batch-size instead')
    del args.batch_size
    ensure_valid(args.warmup is None, '--warmup argument is no longer valid, use ' \
        '--lr-warmup-fraction instead')
    del args.warmup
    ensure_valid(args.model_parallel_size is None, '--model-parallel-size is no ' \
        'longer valid, use --tensor-model-parallel-size instead')
    del args.model_parallel_size

    if args.checkpoint_activations:
        if args.rank == 0:
            print('--checkpoint-activations is no longer valid, use --recompute-activations, '
                  'or, for more control, --recompute-granularity and --recompute-method.')
        exit()
    del args.checkpoint_activations

    if args.recompute_activations:
        args.recompute_granularity = 'selective'
    del args.recompute_activations

    # Set input defaults.
    for key in defaults:
        # For default to be valid, it should not be provided in the
        # arguments that are passed to the program. We check this by
        # ensuring the arg is set to None.
        if getattr(args, key, None) is not None:
            if args.rank == 0:
                print('WARNING: overriding default arguments for {key}:{v} \
                       with {key}:{v2}'.format(key=key, v=defaults[key],
                                               v2=getattr(args, key)),
                                               flush=True)
        else:
            setattr(args, key, defaults[key])

    if args.data_path is not None and args.split is None:
        legacy_default_split_value = '969, 30, 1'
        if args.rank == 0:
            print('WARNING: Please specify --split when using --data-path. Using legacy default value '
                  f'of "{legacy_default_split_value}"')
        args.split = legacy_default_split_value

    use_data_path = (args.data_path is not None) or (args.data_args_path is not None)
    if use_data_path:
        # Exactly one of the two has to be None if we use it.
        ensure_valid((args.data_path is None) or (args.data_args_path is None))
    use_per_split_data_path = any(
        elt is not None
        for elt in [args.train_data_path, args.valid_data_path, args.test_data_path]) or \
            args.per_split_data_args_path is not None
    if use_per_split_data_path:
        # Exactly one of the two has to be None if we use it.
        ensure_valid(any(elt is not None
                   for elt in [args.train_data_path, args.valid_data_path, args.test_data_path]) is False or \
            args.per_split_data_args_path is None
        )
    # Batch size.
    ensure_valid(args.micro_batch_size is not None)
    ensure_valid(args.micro_batch_size > 0)
    if args.global_batch_size is None:
        args.global_batch_size = args.micro_batch_size * args.data_parallel_size
        if args.rank == 0:
            print('setting global batch size to {}'.format(
                args.global_batch_size), flush=True)
    ensure_valid(args.global_batch_size > 0)

    # Uneven virtual pipeline parallelism
    ensure_valid(args.num_layers_per_virtual_pipeline_stage is None or args.num_virtual_stages_per_pipeline_rank is None, \
        '--num-layers-per-virtual-pipeline-stage and --num-virtual-stages-per-pipeline-rank cannot be set at the same time')

    if args.num_layers_per_virtual_pipeline_stage is not None or args.num_virtual_stages_per_pipeline_rank is not None:
        raise AssertionError('MindSpeed-MM Error: --num-layers-per-virtual-pipeline-stage is deprecated, please use --virtual-pipeline-model-parallel-size instead')
    if hasattr(args, 'virtual_pipeline_model_parallel_size') and args.virtual_pipeline_model_parallel_size is not None and args.virtual_pipeline_model_parallel_size > 1:
        if args.overlap_p2p_comm:
            ensure_valid(args.pipeline_model_parallel_size > 1, \
                'when interleaved schedule is used, pipeline-model-parallel size '\
                'should be greater than 1')
        else:
            ensure_valid(args.pipeline_model_parallel_size > 2, \
                'when interleaved schedule is used and p2p communication overlap is disabled, '\
                'pipeline-model-parallel size should be greater than 2 to avoid having multiple '\
                'p2p sends and recvs between same 2 ranks per communication batch')
        if hasattr(args.mm.model, 'text_decoder'):
            _pipeline_num_layers = getattr(args.mm.model.text_decoder, 'pipeline_num_layers', None)
            if _pipeline_num_layers is None or len(_pipeline_num_layers) != args.virtual_pipeline_model_parallel_size:
                raise AssertionError('MindSpeed-MM Error: vpp must enabled by --virtual-pipeline-model-parallel-size in shell and pipeline_num_layers in model.json, \
                    and virtual-pipeline-model-parallel-size must equal the length of pipeline_num_layers')
        elif hasattr(args.mm.model, 'predictor'):
            _pipeline_num_layers = getattr(args.mm.model.predictor, 'pipeline_num_layers', None)
            if _pipeline_num_layers is None or len(_pipeline_num_layers) != args.virtual_pipeline_model_parallel_size:
                raise AssertionError('MindSpeed-MM Error: vpp must enabled by --virtual-pipeline-model-parallel-size in shell and pipeline_num_layers in model.json, \
                    and virtual-pipeline-model-parallel-size must equal the length of pipeline_num_layers')
        else:
            raise AssertionError('MindSpeed-MM Error: vpp must enabled by --virtual-pipeline-model-parallel-size in shell and pipeline_num_layers in model.json')
    else:
        args.virtual_pipeline_model_parallel_size = None
        # Overlap P2P communication is disabled if not using the interleaved schedule.
        args.overlap_p2p_comm = False
        args.align_param_gather = False
        # Only print warning if PP size > 1.
        if args.rank == 0 and args.pipeline_model_parallel_size > 1:
            print('WARNING: Setting args.overlap_p2p_comm and args.align_param_gather to False '
                  'since non-interleaved schedule does not support overlapping p2p communication '
                  'and aligned param AG')

    if args.rank == 0:
        print(f"Number of virtual stages per pipeline stage: {args.virtual_pipeline_model_parallel_size}")

    if args.data_parallel_sharding_strategy == "optim_grads_params":
        args.overlap_param_gather = True
        args.overlap_grad_reduce = True

    if args.data_parallel_sharding_strategy == "optim_grads":
        args.overlap_grad_reduce = True

    if args.overlap_param_gather:
        ensure_valid(args.use_distributed_optimizer, \
            '--overlap-param-gather only supported with distributed optimizer')
        ensure_valid(args.overlap_grad_reduce, \
            'Must use --overlap-param-gather with --overlap-grad-reduce')
        ensure_valid(not args.use_legacy_models, \
            '--overlap-param-gather only supported with MCore models')

    if args.use_torch_fsdp2:
        ensure_valid(is_torch_min_version("2.4.0"), \
            'FSDP2 requires PyTorch >= 2.4.0 with FSDP 2 support.')
        ensure_valid(args.pipeline_model_parallel_size == 1, \
            '--use-torch-fsdp2 is not supported with pipeline parallelism')
        ensure_valid(args.expert_model_parallel_size == 1, \
            '--use-torch-fsdp2 is not supported with expert parallelism')
        ensure_valid(not args.use_distributed_optimizer, \
            "--use-torch-fsdp2 is not supported with MCore's distributed optimizer")
        ensure_valid(not args.gradient_accumulation_fusion, \
            '--use-torch-fsdp2 is not supported with gradient accumulation fusion')
        ensure_valid(args.ckpt_format in ('torch_dist', 'torch_dcp'), \
            '--use-torch-fsdp2 requires --ckpt-format torch_dist or torch_dcp')
        ensure_valid(args.untie_embeddings_and_output_weights, \
            '--use-torch-fsdp2 requires --untie-embeddings-and-output-weights')
        ensure_valid(not args.fp16, \
            '--use-torch-fsdp2 not supported with fp16 yet')
        ensure_valid(os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1", \
            'FSDP always requires CUDA_DEVICE_MAX_CONNECTIONS value large than one')

    if args.overlap_param_gather_with_optimizer_step:
        ensure_valid(args.use_distributed_optimizer, \
            '--overlap-param-gather-with-optimizer-step only supported with distributed optimizer')
        ensure_valid(args.overlap_param_gather, \
            'Must use --overlap-param-gather-with-optimizer-step with --overlap-param-gather')
        ensure_valid(args.virtual_pipeline_model_parallel_size is not None, \
            '--overlap-param-gather-with-optimizer-step only supported with interleaved pipeline parallelism')
        ensure_valid(not args.use_dist_ckpt, \
            '--overlap-param-gather-with-optimizer-step not supported with distributed checkpointing yet')
    if args.optimizer == "muon":
        ensure_valid(not args.use_distributed_optimizer, \
            "--optimizer muon is not supported with MCore's distributed optimizer")

    dtype_map = {
        'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp8': torch.uint8,
    }

    if hasattr(args, 'hetero_parallel') and args.hetero_parallel:
        ensure_valid(args.tensor_model_parallel_size == 1 and \
                     args.pipeline_model_parallel_size == 1 and \
                     args.context_parallel_size == 1, \
            """when enabling hetero-parallel, text decoder initialized by shell is not supported anymore,
            please initial it by model.json and set parallelisms to 1 in shell""")
        ensure_valid(not args.tp_2d, "tp_2d is not supported by hetero-parallel")
        ensure_valid(hasattr(args, 'mm') and hasattr(args.mm, 'model') and hasattr(args.mm.model, 'text_decoder'), \
                     """hetero-parallel only supports mm models that have text_decoder attr""")
        args.tensor_model_parallel_size = args.mm.model.text_decoder.tp
        args.pipeline_model_parallel_size = args.mm.model.text_decoder.pp
        args.context_parallel_size = args.mm.model.text_decoder.cp
        decoder_model_size = args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size
        args.global_batch_size = int(args.global_batch_size * args.world_size / decoder_model_size / args.data_parallel_size)
        args.data_parallel_size = args.world_size // decoder_model_size

    def map_dtype(d):
        if isinstance(d, torch.dtype):
            return d
        else:
            return dtype_map.get(d)

    args.main_grads_dtype = map_dtype(args.main_grads_dtype)
    args.main_params_dtype = map_dtype(args.main_params_dtype)
    args.exp_avg_dtype = map_dtype(args.exp_avg_dtype)
    args.exp_avg_sq_dtype = map_dtype(args.exp_avg_sq_dtype)

    if args.fp8_param_gather:
        ensure_valid(args.use_distributed_optimizer or args.use_torch_fsdp2, \
            '--fp8-param-gather only supported with distributed optimizer or torch fsdp2')

    if args.use_custom_fsdp:
        ensure_valid(args.use_distributed_optimizer, \
            '--use-custom-fsdp only supported with distributed optimizer')

        if args.data_parallel_sharding_strategy in ["optim_grads_params", "optim_grads"]:
            warnings.warn('Please make sure your TransformerEngine support FSDP + gradient accumulation fusion')
            ensure_valid(args.gradient_accumulation_fusion is False, \
                "optim_grads_params optim_grads are not supported with gradient accumulation fusion")

        if args.data_parallel_sharding_strategy == "optim_grads_params":
            ensure_valid(args.check_weight_hash_across_dp_replicas_interval is None, \
                'check_weight_hash_across_dp_replicas_interval is not supported with optim_grads_params')

        ensure_valid(os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1", \
            'FSDP always requires CUDA_DEVICE_MAX_CONNECTIONS value large than one')

    # Parameters dtype.
    args.params_dtype = torch.float
    if args.fp16:
        ensure_valid(not args.bf16)
        args.params_dtype = torch.half
        # Turn off checking for NaNs in loss and grads if using dynamic loss scaling,
        # where NaNs in grads / loss are signal to the loss scaler.
        if not args.loss_scale:
            args.check_for_nan_in_loss_and_grad = False
            if args.rank == 0:
                print('WARNING: Setting args.check_for_nan_in_loss_and_grad to False since '
                      'dynamic loss scaling is being used')
    if args.bf16:
        ensure_valid(not args.fp16)
        args.params_dtype = torch.bfloat16
        # bfloat16 requires gradient accumulation and all-reduce to
        # be done in fp32.
        if args.accumulate_allreduce_grads_in_fp32:
            ensure_valid(args.main_grads_dtype == torch.float32, \
                "--main-grads-dtype can only be fp32 when --accumulate-allreduce-grads-in-fp32 is set")

        if args.grad_reduce_in_bf16:
            args.accumulate_allreduce_grads_in_fp32 = False
        elif not args.accumulate_allreduce_grads_in_fp32 and args.main_grads_dtype == torch.float32:
            args.accumulate_allreduce_grads_in_fp32 = True
            if args.rank == 0:
                print('accumulate and all-reduce gradients in fp32 for '
                      'bfloat16 data type.', flush=True)

    if args.rank == 0:
        print('using {} for parameters ...'.format(args.params_dtype),
              flush=True)

    if args.dataloader_type is None:
        args.dataloader_type = 'single'

    # data
    ensure_valid(args.num_dataset_builder_threads > 0)

    # Consumed tokens.
    args.consumed_train_samples = 0
    args.skipped_train_samples = 0
    args.consumed_valid_samples = 0

    # Support for variable sequence lengths across batches/microbatches.
    # set it if the dataloader supports generation of variable sequence lengths
    # across batches/microbatches. Due to additional communication overhead
    # during pipeline parallelism, it should not be set if sequence length
    # is constant during training.
    args.variable_seq_lengths = False

    # Iteration-based training.
    if args.train_iters:
        # If we use iteration-based training, make sure the
        # sample-based options are off.
        ensure_valid(args.train_samples is None, \
            'expected iteration-based training')
        ensure_valid(args.lr_decay_samples is None, \
            'expected iteration-based learning rate decay')
        ensure_valid(args.lr_warmup_samples == 0, \
            'expected iteration-based learning rate warmup')
        ensure_valid(args.rampup_batch_size is None, \
            'expected no batch-size rampup for iteration-based training')
        if args.lr_warmup_fraction is not None:
            ensure_valid(args.lr_warmup_iters == 0, \
                'can only specify one of lr-warmup-fraction and lr-warmup-iters')

    # Sample-based training.
    if args.train_samples:
        # If we use sample-based training, make sure the
        # iteration-based options are off.
        ensure_valid(args.train_iters is None, \
            'expected sample-based training')
        ensure_valid(args.lr_decay_iters is None, \
            'expected sample-based learning rate decay')
        ensure_valid(args.lr_warmup_iters == 0, \
            'expected sample-based learnig rate warmup')
        if args.lr_warmup_fraction is not None:
            ensure_valid(args.lr_warmup_samples == 0, \
                'can only specify one of lr-warmup-fraction ' \
                'and lr-warmup-samples')

    if args.num_layers is not None:
        ensure_valid(args.encoder_num_layers is None, \
            'cannot have both num-layers and encoder-num-layers specified')
        args.encoder_num_layers = args.num_layers
    else:
        ensure_valid(args.encoder_num_layers is not None, \
            'either num-layers or encoder-num-layers should be specified')
        args.num_layers = args.encoder_num_layers

    # Check required arguments.
    required_args = ['num_layers', 'hidden_size', 'num_attention_heads',
                     'max_position_embeddings']
    for req_arg in required_args:
        _check_arg_is_not_none(args, req_arg)

    # Checks.
    if args.ffn_hidden_size is None:
        if args.swiglu:
            # reduce the dimnesion for MLP since projections happens on
            # two linear layers. this keeps the number of parameters in
            # the same ballpark as the counterpart with 4*h size
            # we keep it a multiple of 64, which means the actual tensor size
            # will be a multiple of 64 / tp_size
            args.ffn_hidden_size = int((4 * args.hidden_size * 2 / 3) / 64) * 64
        else:
            args.ffn_hidden_size = 4 * args.hidden_size

    if args.kv_channels is None:
        ensure_valid(args.hidden_size % args.num_attention_heads == 0)
        args.kv_channels = args.hidden_size // args.num_attention_heads

    if args.seq_length is not None and args.context_parallel_size > 1:
        ensure_valid(args.seq_length % (args.context_parallel_size * 2) == 0, \
            'seq-length should be a multiple of 2 * context-parallel-size ' \
            'if context-parallel-size > 1.')

    if args.seq_length is not None:
        ensure_valid(args.encoder_seq_length is None)
        args.encoder_seq_length = args.seq_length
    else:
        ensure_valid(args.encoder_seq_length is not None)
        args.seq_length = args.encoder_seq_length

    if args.seq_length is not None:
        ensure_valid(args.max_position_embeddings >= args.seq_length, \
            f"max_position_embeddings ({args.max_position_embeddings}) must be greater than " \
            f"or equal to seq_length ({args.seq_length}).")
    if args.decoder_seq_length is not None:
        ensure_valid(args.max_position_embeddings >= args.decoder_seq_length)
    if args.lr is not None:
        ensure_valid(args.min_lr <= args.lr)
    if args.save is not None:
        ensure_valid(args.save_interval is not None)
    if hasattr(args, 'encoder_offload_interval') and args.encoder_offload_interval is not None and args.save is not None:
        ensure_valid(args.save_interval % args.encoder_offload_interval == 0, \
            f"to use checkpointing under the encoder_offload_interval feature, " \
            f"save_interval ({args.save_interval}) must be an integer multiple of " \
            f"encoder_offload_interval ({args.encoder_offload_interval}).")
    # Mixed precision checks.
    if args.fp16_lm_cross_entropy:
        ensure_valid(args.fp16, 'lm cross entropy in fp16 only support in fp16 mode.')
    if args.fp32_residual_connection:
        ensure_valid(args.fp16 or args.bf16, \
            'residual connection in fp32 only supported when using fp16 or bf16.')

    if args.moe_grouped_gemm:
        ensure_valid(args.bf16, 'Currently GroupedGEMM for MoE only supports bf16 dtype.')
        dc = torch.cuda.get_device_capability()
        ensure_valid(dc[0] >= 8, "Unsupported compute capability for GroupedGEMM kernels.")

    if args.weight_decay_incr_style == 'constant':
        ensure_valid(args.start_weight_decay is None)
        ensure_valid(args.end_weight_decay is None)
        args.start_weight_decay = args.weight_decay
        args.end_weight_decay = args.weight_decay
    else:
        ensure_valid(args.start_weight_decay is not None)
        ensure_valid(args.end_weight_decay is not None)

    # Persistent fused layer norm.
    if not is_torch_min_version("1.11.0a0"):
        args.no_persist_layer_norm = True
        if args.rank == 0:
            print('Persistent fused layer norm kernel is supported from '
                  'pytorch v1.11 (nvidia pytorch container paired with v1.11). '
                  'Defaulting to no_persist_layer_norm=True')

    # Activation recomputing.
    if args.distribute_saved_activations:
        ensure_valid(args.tensor_model_parallel_size > 1, 'can distribute ' \
            'recomputed activations only across tensor model ' \
            'parallel groups')
        ensure_valid(args.recompute_granularity == 'full', \
            'distributed recompute activations is only '\
            'application to full recompute granularity')
        ensure_valid(args.recompute_method is not None, \
            'for distributed recompute activations to work you '\
            'need to use a recompute method ')
        ensure_valid(is_torch_min_version("1.10.0a0"), \
            'distributed recompute activations are supported for pytorch ' \
            'v1.10 and above (Nvidia Pytorch container >= 21.07). Current ' \
            f'pytorch version is v{get_torch_version()}.')

    if args.recompute_granularity == 'selective':
        ensure_valid(args.recompute_method is None, \
            'recompute method is not yet supported for ' \
            'selective recomputing granularity')

    # disable sequence parallelism when tp=1
    # to avoid change in numerics when
    # sequence_parallelism is enabled.
    if args.tensor_model_parallel_size == 1:
        if args.sequence_parallel:
            warnings.warn("Disabling sequence parallelism because tensor model parallelism is disabled")
        args.sequence_parallel = False

    if args.tp_comm_overlap:
        ensure_valid(args.sequence_parallel, 'Tensor parallel communication/GEMM overlap can happen only when sequence parallelism is enabled')

    # disable async_tensor_model_parallel_allreduce when
    # model parallel memory optimization is enabled
    if args.tensor_model_parallel_size > 1 or args.context_parallel_size > 1 and get_device_arch_version() < 10:
        # CUDA_DEVICE_MAX_CONNECTIONS requirement no longer exists since the Blackwell architecture
        if args.use_torch_fsdp2 or args.use_custom_fsdp:
            fsdp_impl = "Torch-FSDP2" if args.use_torch_fsdp2 else "Custom-FSDP"
            warnings.warn(
                f"Using tensor model parallelism or context parallelism with {fsdp_impl} together. "
                "Try not to using them together since they require different CUDA_MAX_CONNECTIONS "
                "settings for best performance. sequence parallelism requires setting the "
                f"environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 while {fsdp_impl} "
                "requires not setting CUDA_DEVICE_MAX_CONNECTIONS=1 for better parallelization.")
        else:
            ensure_valid(os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') == "1", \
                "Using tensor model parallelism or context parallelism require setting the environment variable " \
                "CUDA_DEVICE_MAX_CONNECTIONS to 1")

    # Disable bias gelu fusion if we are disabling bias altogether
    if not args.add_bias_linear:
        args.bias_gelu_fusion = False

    # Keep the 'add bias' args in sync; add_qkv_bias is more targeted.
    if args.add_bias_linear:
        args.add_qkv_bias = True

    # Retro checks.
    if args.retro_add_retriever:

        # Train samples should be auto-loaded.
        ensure_valid(args.train_samples is not None, \
            "args.train_samples should be auto-loaded from the retro config.")

        # Sequence parallelism unsupported.
        ensure_valid(not args.sequence_parallel, \
            "retro currently does not support sequence parallelism.")

        # Pipeline parallelism unsupported.
        ensure_valid(args.pipeline_model_parallel_size == 1, \
            "retro currently does not support pipeline parallelism.")

    if args.decoupled_lr is not None or args.decoupled_min_lr is not None:
        ensure_valid(not args.use_legacy_models, \
            '--decoupled-lr and --decoupled-min-lr is not supported in legacy models.')

    # Legacy RoPE arguments
    if args.use_rotary_position_embeddings:
        args.position_embedding_type = 'rope'
    if args.rotary_interleaved and args.apply_rope_fusion:
        raise RuntimeError('--rotary-interleaved does not work with rope_fusion.')
    if args.rotary_interleaved and args.use_legacy_models:
        raise RuntimeError('--rotary-interleaved is not supported in legacy models.')
    if args.position_embedding_type != 'rope':
        args.apply_rope_fusion = False

    # Would just need to add 'NoPE' as a position_embedding_type to support this, but for now
    # don't allow it to keep things simple
    if not args.add_position_embedding and args.position_embedding_type != 'rope':
        raise RuntimeError('--no-position-embedding is deprecated, use --position-embedding-type')

    # Relative position embeddings arguments
    if args.position_embedding_type == 'relative':
        ensure_valid((
            args.transformer_impl == "transformer_engine"
        ), 'Local transformer implementation currently does not support attention bias-based position embeddings.')

    # MultiModal rotary embeddings arguments
    if args.position_embedding_type == "mrope":
        ensure_valid(args.mrope_section is not None, \
            '--mrope-section should be set when using --position-embedding-type mrope.')

    # MoE Spec check
    if args.num_experts == 0:
        args.num_experts = None
    if args.num_experts is not None:
        ensure_valid(args.spec is None, "Model Spec must be None when using MoEs")

    if args.moe_ffn_hidden_size is None:
        args.moe_ffn_hidden_size = args.ffn_hidden_size

    # Context parallel
    if args.context_parallel_size > 1:
        ensure_valid(not args.use_legacy_models, "Context parallelism is not supported in legacy models.")

    # Expert parallelism check
    if args.expert_model_parallel_size > 1:
        ensure_valid(args.num_experts is not None, "num_experts must be non None to use expert model parallelism")
        ensure_valid(args.num_experts % args.expert_model_parallel_size == 0, \
            "Number of experts should be a multiple of expert model parallel_size.")
        ensure_valid(not args.fp16, \
            "Expert parallelism is not supported with fp16 training.")

    # Distributed checkpointing checks
    if args.use_dist_ckpt and args.use_legacy_models:
        raise RuntimeError('--use-dist-ckpt is not supported in legacy models.')

    # torch_dcp (torch.distributed.checkpoint) checkpointing format checks.
    if args.ckpt_format == "torch_dcp":
        ensure_valid(args.use_torch_fsdp2, "--ckpt-format torch_dcp is only tested with FSDP.")
        ensure_valid(args.tensor_model_parallel_size <= 1 and args.encoder_tensor_model_parallel_size <= 1, \
            "--ckpt-format torch_dcp is not tested with megatron tensor parallelism.")
        ensure_valid(args.pipeline_model_parallel_size <= 1 and args.encoder_pipeline_model_parallel_size <= 1, \
            "--ckpt-format torch_dcp is not tested with megatron pipeline parallelism.")

    # Data blend checks
    ensure_valid(args.mock_data + \
           bool(args.data_path) + \
           any([args.train_data_path, args.valid_data_path, args.test_data_path]) \
           <= 1, "A single data source must be provided in training mode, else None")

    # Deterministic mode
    if args.deterministic_mode:
        ensure_valid(not args.use_flash_attn, "Flash attention can not be used in deterministic mode.")
        ensure_valid(not args.cross_entropy_loss_fusion, "Cross Entropy Fusion is currently not deterministic.")

        all_reduce_choices = ["Tree", "Ring", "CollnetDirect", "CollnetChain", "^NVLS"]
        ensure_valid(os.getenv("NCCL_ALGO", -1) != -1 and os.getenv("NCCL_ALGO") in all_reduce_choices, \
            f"NCCL_ALGO must be one of {all_reduce_choices}.")

        torch.use_deterministic_algorithms(True)

    # Update the printed args to reflect that `apply_query_key_layer_scaling` also controls `attention_softmax_in_fp32`
    if args.apply_query_key_layer_scaling:
        args.attention_softmax_in_fp32 = True

    if args.result_rejected_tracker_filename is not None:
        # Append to passed-in args.iterations_to_skip.
        iterations_to_skip_from_file = RerunStateMachine.get_skipped_iterations_from_tracker_file(
            args.result_rejected_tracker_filename
        )
        args.iterations_to_skip.extend(iterations_to_skip_from_file)

    # Make sure all functionality that requires Gloo process groups is disabled.
    if not args.enable_gloo_process_groups:
        if args.use_distributed_optimizer:
            # If using distributed optimizer, must use distributed checkpointing.
            # Legacy checkpointing uses Gloo process groups to collect full distributed
            # optimizer state in the CPU memory of DP rank 0.
            ensure_valid(args.use_dist_ckpt)

    # Checkpointing
    if args.ckpt_fully_parallel_save_deprecated and args.rank == 0:
        print('--ckpt-fully-parallel-save flag is deprecated and has no effect.'
              ' Use --no-ckpt-fully-parallel-save to disable parallel save.')
    use_dist_ckpt_and_not_ckpt_fully_parallel_save = args.use_dist_ckpt and not args.ckpt_fully_parallel_save
    use_distributed_optimizer_and_rank = args.use_distributed_optimizer and args.rank == 0

    if use_dist_ckpt_and_not_ckpt_fully_parallel_save and use_distributed_optimizer_and_rank:
        print('Warning: With non-parallel ckpt save and DistributedOptimizer,'
              ' it will be impossible to resume training with different parallelism.'
              ' Consider removing flag --no-ckpt-fully-parallel-save.')
    if args.use_dist_ckpt_deprecated and args.rank == 0:
        print('--use-dist-ckpt is deprecated and has no effect.'
              ' Use --ckpt-format to select the checkpoint format.')
    if args.dist_ckpt_format_deprecated and args.rank == 0:
        print('--dist-ckpt-format is deprecated and has no effect.'
              ' Use --ckpt-format to select the checkpoint format.')

    # Inference args
    if args.inference_batch_times_seqlen_threshold > -1:
        ensure_valid(args.pipeline_model_parallel_size > 1, \
            "--inference-batch-times-seqlen-threshold requires setting --pipeline-model-parallel-size > 1.")

    if args.inference_dynamic_batching:
        ensure_valid(args.inference_dynamic_batching_buffer_size_gb is not None)
        ensure_valid(args.inference_dynamic_batching_buffer_guaranteed_fraction is not None)

    # MoE upcycling check
    if args.moe_use_upcycling:
        ensure_valid(args.save is not None, "When using upcycling, the --save option must be specified.")
        if not args.no_load_optim:
            args.no_load_optim = True
            print('Warning: disabling --no-load-optim for upcycling.')
        if not args.no_load_rng:
            args.no_load_rng = True
            print('Warning: disabling --no-load-rng for upcycling.')

    # Optimizer CPU offload check
    if args.optimizer_cpu_offload:
        ensure_valid(args.use_precision_aware_optimizer, (
            "The optimizer cpu offload must be used in conjunction with `--use-precision-aware-optimizer`, "
            "as the hybrid device optimizer reuses the code path of this flag."
        ))


    if args.non_persistent_ckpt_type == "local":
        ensure_valid(args.non_persistent_local_ckpt_dir is not None, "Tried to use local checkpointing without specifying --local-ckpt-dir!")
    if args.replication:
        ensure_valid(args.replication_jump is not None, "--replication requires the value of --replication-jump!")
        ensure_valid(args.non_persistent_ckpt_type == "local", f"--replication requires args.non_persistent_ckpt_type == 'local', but got: {args.non_persistent_ckpt_type}")
    elif args.replication_jump:
        print("Warning: --replication-jump was specified despite not using replication. Ignoring.")
        args.replication_jump = None

    if args.mtp_num_layers:
        ensure_valid(not args.use_legacy_models, "The legacy Megatron models does not support Multi-Token Prediction (MTP).")
        ensure_valid(args.context_parallel_size == 1, "Multi-Token Prediction (MTP) is not supported with Context Parallelism.")
        ensure_valid(args.position_embedding_type == "rope" or args.position_embedding_type == "none", (
            f"Multi-Token Prediction (MTP) is not supported with {args.position_embedding_type} position embedding type."
            + f"The supported position embedding types are rope and none."
        ))

    if args.fp8 and args.transformer_impl == 'local':
        raise AssertionError("FP just support TE implement.")
    if args.use_ascend_coc and args.transformer_impl == "transformer_engine":
        raise AssertionError("transformer engine does not support ascend coc.")
    if args.fp8 and args.use_ascend_mc2:
        raise AssertionError("FP8 currently does not support mc2.")
    if (getattr(args, "transformer_impl", "transformer_engine") == "transformer_engine"
        and getattr(args, "use_legacy_models", False)):
        raise AssertionError("transformer engine only support for mcore models")
    if getattr(args, "use_gmm_fp8", False):
        if args.fp8_recipe not in ("mxfp8", "tensorwise", "delayed"):
            warnings.warn(
                f"gmm fp8 only supports tensorwise, mxfp8, and delayed recipe, but {args.fp8_recipe} provided"
                f"using bf16 gmm instead."
            )

    # Print arguments.
    _print_args("arguments", args)

    ensure_valid(not (getattr(args, 'calculate_per_sample_loss', False) and getattr(args, 'calculate_per_token_loss', False)),
                 ("calculate_per_sample_loss and calculate_per_token_loss cannot be used simultaneously; "
                  "only one of them can be enabled. If neither is specified, loss computation will fall back to the default behavior. "
                  "For more details, please refer to the relevant documentation: vlm_model_loss_calculate_type.md"
                  ))

    return args

pm.register_patch("megatron.training.arguments.validate_args", validate_args, force_patch=True)
pm.register_patch("mindspeed_mm.patchs.validate_args_patch.validate_args", validate_args_wrapper)
pm.apply_patches()
