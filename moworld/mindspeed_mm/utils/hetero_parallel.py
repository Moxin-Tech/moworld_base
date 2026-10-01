import inspect
from functools import wraps
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn.functional as F
from mindspeed.megatron_adaptor import get_mindspeed_args
from mindspeed.patch_utils import MindSpeedPatchesManager as mspm
from megatron.training import get_args, print_rank_0
from megatron.core.parallel_state import initialize_model_parallel, is_initialized
import megatron.core.parallel_state as mpu
from megatron.training.arguments import parse_args
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.model_parallel_config import ModelParallelConfig


_ParallelStatesDict = {}
_HeteroParallelModules = ['image_encoder', 'audio_encoder', 'text_decoder']


def apply_hetero_parallel_hooks(model):

    if hasattr(model, 'image_encoder') and model.image_encoder is not None:
        model.image_encoder.register_forward_pre_hook(image_encoder_forward_pre_hook)
        model.image_encoder.register_forward_hook(image_encoder_forward_hook)
    if hasattr(model, 'audio_encoder') and model.audio_encoder is not None:
        model.audio_encoder.register_forward_pre_hook(audio_encoder_forward_pre_hook)
        model.audio_encoder.register_forward_hook(audio_encoder_forward_hook)


def image_encoder_forward_pre_hook(module, input):
    pixel_values, image_grid_thw, text_img_num = input
    change_parallel_state('text_decoder')
    pixel_values, _ = all_gather_dp_group(pixel_values, pad_dim=0, remove_padding=True)
    image_grid_thw, _ = all_gather_dp_group(image_grid_thw, pad_dim=0, remove_padding=True)
    text_img_num, _ = all_gather_dp_group(text_img_num, cat_dim=0)
    change_parallel_state('image_encoder')

    pv_lens = []
    thw_num_per_DP_rank = []
    for text_img_num_chunk in torch.chunk(text_img_num, chunks=mpu.get_data_parallel_world_size(), dim=0):
        thw_num_per_DP_rank.append(text_img_num_chunk.sum())
    start = 0
    for thw_num in thw_num_per_DP_rank:
        end = start + thw_num
        block = image_grid_thw[start:end]
        prod = block.prod(dim=1).sum()
        pv_lens.append(prod)
        start = end
    pixel_values = split_tensor_dp_group(pixel_values, pad_dim=0, chunk_seq_lens=pv_lens)  # [B, S]
    image_grid_thw = split_tensor_dp_group(image_grid_thw, split_dim=0, chunk_seq_lens=thw_num_per_DP_rank)
    return pixel_values, image_grid_thw


def image_encoder_forward_hook(module, input, output):
    output, all_lens = all_gather_dp_group(output, cat_dim=0, pad_dim=0, remove_padding=True)

    change_parallel_state('text_decoder')

    chunk_seq_lens = []
    origin_len = len(all_lens)
    for i in range(0, origin_len, origin_len // mpu.get_data_parallel_world_size()):
        length = sum(all_lens[i: i + origin_len // mpu.get_data_parallel_world_size()])
        chunk_seq_lens.append(length)

    output = split_tensor_dp_group(output, pad_dim=0, split_dim=0, chunk_seq_lens=chunk_seq_lens)


    return output


def audio_encoder_forward_pre_hook(module, input):
    input_features, feature_attention_mask = input
    change_parallel_state('text_decoder')
    input_features, _ = all_gather_dp_group(input_features)
    feature_attention_mask, _ = all_gather_dp_group(feature_attention_mask)
    change_parallel_state('audio_encoder')
    input_features = split_tensor_dp_group(input_features)
    feature_attention_mask = split_tensor_dp_group(feature_attention_mask)

    return input_features, feature_attention_mask


def audio_encoder_forward_hook(module, input, output):
    output, all_lens = all_gather_dp_group(output, pad_token_id=0.0, cat_dim=0, pad_dim=0, remove_padding=True)
    change_parallel_state('text_decoder')

    chunk_seq_lens = []
    origin_len = len(all_lens)
    for i in range(0, origin_len, origin_len // mpu.get_data_parallel_world_size()):
        length = sum(all_lens[i: i + origin_len // mpu.get_data_parallel_world_size()])
        chunk_seq_lens.append(length)

    output = split_tensor_dp_group(output, pad_dim=0, split_dim=0, chunk_seq_lens=chunk_seq_lens)

    return output


def destroy_model_parallel_ranks(parallel_state):
    for k, v in vars((mpu)).items():
        is_global_variable = k.startswith('_') and not k.startswith('__') and not inspect.isfunction(v)
        if is_global_variable and '_RANK' in k:
            setattr(parallel_state, k, None)


def initial_modules_mpu(config, kwargs):
    config_dict = config.to_dict()


    extra_args_provider = kwargs.get('extra_args_provider', None)
    ignore_unknown_args = kwargs.get('ignore_unknown_args', False)
    parsed_args = kwargs.get('parsed_args', None)

    if parsed_args is None:
        args = parse_args(extra_args_provider, ignore_unknown_args)
    else:
        args = parsed_args

    module_name = [['image_encoder', 'vision_encoder'], ['audio_encoder', 'audio_encoder'], ['text_decoder']]
    module_config = {}

    for module_group in module_name:
        current_config = config_dict
        for key in module_group:
            if current_config[key] is None:
                continue
            try:
                current_config = current_config[key]
            except KeyError as e:
                raise KeyError(f"Key '{key}' not found in current_config: {current_config}") from e
        module_config[module_group[0]] = current_config

    def pass_hetero_initial_arguments(key, module, default=None, use_args=False, main_module='text_decoder'):
        """
        pass the args by <module config - shell - megatron config - manual default> priority
        """
        if module in module_config and key in module_config[module]:
            return module_config[module][key]

        if module == main_module or use_args:
            return getattr(args, key)

        config_list = [ModelParallelConfig, DistributedDataParallelConfig]
        for config in config_list:
            if hasattr(config, key):
                return getattr(config, key)
        return default



    for module in module_config.keys():

        if module not in _ParallelStatesDict:
            _ParallelStatesDict[module] = {}
            mpu.destroy_model_parallel()
            destroy_model_parallel_ranks(mpu)

            initialize_model_parallel(
                tensor_model_parallel_size=pass_hetero_initial_arguments('tensor_model_parallel_size', module),
                pipeline_model_parallel_size=pass_hetero_initial_arguments('pipeline_model_parallel_size', module),
                virtual_pipeline_model_parallel_size=pass_hetero_initial_arguments('virtual_pipeline_model_parallel_size', module),
                pipeline_model_parallel_split_rank=pass_hetero_initial_arguments('pipeline_model_parallel_split_rank', module),
                pipeline_model_parallel_comm_backend=pass_hetero_initial_arguments('pipeline_model_parallel_comm_backend', module, use_args=True),
                context_parallel_size=pass_hetero_initial_arguments('context_parallel_size', module),
                hierarchical_context_parallel_sizes=pass_hetero_initial_arguments('hierarchical_context_parallel_sizes', module),
                expert_model_parallel_size=pass_hetero_initial_arguments('expert_model_parallel_size', module),
                num_distributed_optimizer_instances=pass_hetero_initial_arguments('num_distributed_optimizer_instances', module, use_args=True),
                expert_tensor_parallel_size=pass_hetero_initial_arguments('expert_tensor_parallel_size', module),
                distributed_timeout_minutes=pass_hetero_initial_arguments('distributed_timeout_minutes', module, use_args=True),
                nccl_communicator_config_path=pass_hetero_initial_arguments('nccl_communicator_config_path', module, use_args=True),
                order='tp-cp-ep-dp-pp' if not pass_hetero_initial_arguments('use_tp_pp_dp_mapping', module, use_args=True) else 'tp-cp-ep-pp-dp',
                encoder_tensor_model_parallel_size=pass_hetero_initial_arguments('encoder_tensor_model_parallel_size', module, use_args=True),
                encoder_pipeline_model_parallel_size=pass_hetero_initial_arguments('encoder_pipeline_model_parallel_size', module, use_args=True),
                get_embedding_ranks=kwargs.get('get_embedding_ranks', None),
                get_position_embedding_ranks=kwargs.get('get_position_embedding_ranks', None),
                create_gloo_process_groups=pass_hetero_initial_arguments('enable_gloo_process_groups', module, use_args=True),
            )

        state_snapshot = {
            k: v for k, v in vars((mpu)).items()
            if k.startswith('_') and not k.startswith('__') and not inspect.isfunction(v)
        }
        _ParallelStatesDict[module].update(state_snapshot)


def change_parallel_state(module):
    target_globals = vars(mpu)
    source_globals = _ParallelStatesDict[module]

    for k, v in source_globals.items():
        if k in target_globals:
            target_globals[k] = v


def initial_megatron_hetero_parallel_wrapper(fn):
    print_rank_0('initial_megatron_hetero_parallel_wrapper activated')

    @wraps(fn)
    def wrapper(*args, **kwargs):
        fn(*args, **kwargs)
        args = get_args()
        vlm_config = deepcopy(args.mm.model)
        from pretrain_vlm import _configure_modules
        _configure_modules(vlm_config, _HeteroParallelModules)

        initial_modules_mpu(config=vlm_config,
                            kwargs=kwargs)
        return
    return wrapper


if hasattr(get_mindspeed_args(), 'hetero_parallel') and get_mindspeed_args().hetero_parallel:
    mspm.register_patch('mindspeed_mm.training.initialize_megatron',
                        initial_megatron_hetero_parallel_wrapper, force_patch=True)
    mspm.apply_patches()


def all_gather_dp_group(tensor,
                        pad_token_id=None,
                        cat_dim=0,
                        pad_dim=1,
                        remove_padding=False,
                        parallel_state=None,
                        ):
    """Gather tensors
        暂时只支持BSH、BD
    """

    if parallel_state is None:
        group = mpu.get_data_parallel_group()
        world_size = mpu.get_data_parallel_world_size()
    else:
        group = parallel_state['_DATA_PARALLEL_GROUP']
        world_size = torch.distributed.get_world_size(group=group)
    if tensor is None:
        return None, None

    if pad_token_id is not None or remove_padding:
        pad_token_id = 0 if pad_token_id is None else pad_token_id
        local_len = torch.tensor([tensor.shape[pad_dim]], device='cuda')
        all_lens = [torch.zeros_like(local_len) for _ in range(world_size)]

        dist.all_gather(all_lens, local_len, group=group)
        all_lens = [length.item() for length in all_lens]
        max_len = max(all_lens)

        pad_size = max_len - local_len
        if pad_size > 0:
            pad_dims = [0] * (2 * tensor.dim())
            # pad_dims: [B, S, H], [D_left, D_right, S_left, S_right, H_left, H_right]
            pad_dims[2 * (tensor.dim() - pad_dim) - 1] = pad_size
            tensor = F.pad(tensor, pad_dims, value=pad_token_id)

    if tensor.requires_grad:
        if remove_padding:
            raise NotImplementedError('tensors that require grad and need removing padding are not implemented')
        output = _AllGatherDp.apply(tensor, cat_dim)
    else:
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor, group=group)

        if remove_padding:
            gathered = [g[:length] for g, length in zip(gathered, all_lens)]
        output = torch.cat(gathered, dim=cat_dim).contiguous()

    if remove_padding:
        return output, all_lens
    return output, None


def split_tensor_dp_group(tensor,
                          split_dim=0,
                          pad_dim=1,
                          chunk_seq_lens=None,
                          all_lens=None,
                          parallel_state=None):
    """split tensors
        暂时只支持bsh
        chunk_seq_lens: split tensor sliding chunk_seq_lens
        all_lens: all tensor origin lens(cat_dim)
                  if all_lens is None, split tensor per device equal or not remove padding,
                  if all_lens is not None, remove padding intra-dp, do not remove padding inter-dp
    """

    if parallel_state is None:
        world_size = mpu.get_data_parallel_world_size()
        group = mpu.get_data_parallel_group()
    else:
        group = parallel_state['_DATA_PARALLEL_GROUP']
        world_size = torch.distributed.get_world_size(group=group)

    if tensor is None:
        return None

    rank = torch.distributed.get_rank(group)

    if chunk_seq_lens:
        chunk = torch.split(tensor, dim=split_dim, split_size_or_sections=chunk_seq_lens)[rank]
    else:
        chunks = torch.chunk(tensor, world_size, dim=split_dim)
        chunk = chunks[rank]
        if all_lens is not None:
            # for not equal split, need remove padding
            local_lens_num = len(all_lens) // world_size
            start_idx = rank * local_lens_num
            end_idx = start_idx + local_lens_num
            local_lens = all_lens[start_idx: end_idx]
            index = [slice(None)] * chunk.ndim
            index[pad_dim] = slice(0, max(local_lens))  # for inner-mbs, not remove padding
            chunk = chunk[tuple(index)]
    return chunk


class _AllGatherDp(torch.autograd.Function):
    """
    all gahter for dp for diff cat dim and padding dim
    """
    @staticmethod
    def forward(ctx, _input, cat_dim=0):
        group = mpu.get_data_parallel_group()
        world_size = mpu.get_data_parallel_world_size()
        group_rank = torch.distributed.get_rank(group)
        ctx.world_size = world_size
        ctx.group = group
        ctx.group_rank = group_rank
        ctx.cat_dim = cat_dim
        ctx.original_batch_size = _input.shape[cat_dim]


        gathered = [torch.zeros_like(_input) for _ in range(world_size)]
        dist.all_gather(gathered, _input, group=group)
        output = torch.cat(gathered, dim=cat_dim).contiguous()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        world_size, group, group_rank, cat_dim, original_batch_size \
            = ctx.world_size, ctx.group, ctx.group_rank, ctx.cat_dim, ctx.original_batch_size, \

        start = group_rank * original_batch_size
        end = start + original_batch_size

        idx = [slice(None)] * grad_output.dim()
        idx[cat_dim] = slice(start, end)
        grad_input = grad_output[tuple(idx)]

        return grad_input, None


def hetero_align_config(config_inner, config_outer):
    config_inner.pipeline_model_parallel_size = config_outer.pp
    config_inner.context_parallel_size = config_outer.cp
    config_inner.tensor_model_parallel_size = config_outer.tp
