"""
lora模型合并脚本，将基础模型权重和LoRA权重进行合并，生成新的权重文件。

保存lora权重目录:
your_ckpt_path_to_save
├── iter_0005000
│   └── mp_rank_00
│       └── model_optim_rng.pt
└── latest_checkpointed_iteration.txt

原始权重目录:
converted_transformer
├── latest_checkpointed_iteration.txt
└── release
    └── mp_rank_00
        └── model_optim_rng.pt

合并后权重目录:
merge_base_lora_weight
├── latest_checkpointed_iteration.txt
└── release
    └── mp_rank_00
        └── model_optim_rng.pt
"""


import argparse
import os
import stat
from pathlib import Path

import mindspeed.megatron_adaptor
import torch
import torch_npu

from checkpoint.common.permissions import set_directory_permissions


def get_latest_iteration(path: Path) -> str:
    """从指定路径读取最新的迭代号."""
    latest_txt = path.joinpath("latest_checkpointed_iteration.txt")
    return latest_txt.read_text().strip() if latest_txt.exists() else 'release'


def save_latest_checkpointed_iteration(save_dir: str, iteration: str):
    """保存最新的迭代号到指定目录."""
    flags = os.O_WRONLY | os.O_CREAT
    mode = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(os.path.join(save_dir, 'latest_checkpointed_iteration.txt'), flags, mode), 'w') as fout:
        fout.write(iteration)


def merge_model(base_dir: str, lora_dir: str, save_dir: str, pp_size, tp_size: int = 1):
    # 获取基础模型和LoRA模型的迭代号
    base_save_dir = Path(base_dir)
    base_iteration = get_latest_iteration(base_save_dir)
    base_save_dir = base_save_dir.joinpath(f"iter_{int(base_iteration):07}" if base_iteration != "release" else base_iteration)

    lora_save_dir = Path(lora_dir)
    lora_iteration = get_latest_iteration(lora_save_dir)
    lora_save_dir = lora_save_dir.joinpath(f"iter_{int(lora_iteration):07}" if lora_iteration != "release" else lora_iteration)

    # 保存最新的迭代号
    save_latest_checkpointed_iteration(save_dir, 'release')

    # 遍历每个 TP 和 PP 组合进行模型合并
    for tp_rank in range(tp_size):
        for pp_rank in range(pp_size):
            # 构建文件路径
            if pp_size > 1:
                base_current_path = base_save_dir.joinpath(f"mp_rank_{int(tp_rank):02}_{int(pp_rank):03}")
                lora_current_path = lora_save_dir.joinpath(f"mp_rank_{int(tp_rank):02}_{int(pp_rank):03}")
                save_pt_path = os.path.join(save_dir, 'release', f"mp_rank_{int(tp_rank):02}_{int(pp_rank):03}", 'model_optim_rng.pt')
                rank_info = f"mp_rank_{int(tp_rank):02}_{int(pp_rank):03}"
            else:
                base_current_path = base_save_dir.joinpath(f"mp_rank_{int(tp_rank):02}")
                lora_current_path = lora_save_dir.joinpath(f"mp_rank_{int(tp_rank):02}")
                save_pt_path = os.path.join(save_dir, 'release', f"mp_rank_{int(tp_rank):02}", 'model_optim_rng.pt')
                rank_info = f"mp_rank_{int(tp_rank):02}"
            base_pt_path = base_current_path.joinpath("model_optim_rng.pt")
            lora_pt_path = lora_current_path.joinpath("model_optim_rng.pt")

            print(f"Base model path: {base_pt_path}".center(100, '_'))
            print(f"Lora model path: {lora_pt_path}".center(100, '_'))

            # 加载模型权重
            if use_npu:
                base_state_dict = torch.load(base_pt_path, map_location='npu')['model']
                lora_state_dict = torch.load(lora_pt_path, map_location='npu')['model']
            else:
                base_state_dict = torch.load(base_pt_path, map_location='cpu')['model']
                lora_state_dict = torch.load(lora_pt_path, map_location='cpu')['model']

            # 合并权重
            print(f"Merging Base model and Lora model in {rank_info}...")
            merge_state_dict = lora_merge_to_base(base_state_dict, lora_state_dict, lora_target_modules, scaling)
            del base_state_dict, lora_state_dict
            # 保存合并后的权重
            os.makedirs(os.path.dirname(save_pt_path), exist_ok=True)
            torch.save({'model': merge_state_dict}, save_pt_path)
            del merge_state_dict
            if use_npu:
                torch.npu.empty_cache()


def lora_merge_to_base(base_state_dict, lora_state_dict, lora_target_modules, scaling):
    """将LoRA的权重合并到基础模型权重中."""
    merge_state_dict = base_state_dict  # 复制基础模型的权重
    target_layers = set()
    for name in lora_state_dict.keys():
        if 'weight' in name and any(lora_target_module in name for lora_target_module in lora_target_modules):
            target_layers.add(name.split('.lora_')[0])
    for target_layer in target_layers:
        lora_a_weight = lora_state_dict.get(target_layer + '.lora_A.default.weight', None)
        lora_b_weight = lora_state_dict.get(target_layer + '.lora_B.default.weight', None)
        if lora_a_weight is not None and lora_b_weight is not None:
            merge_state_dict[target_layer + '.weight'].data.addmm_(lora_b_weight.data, lora_a_weight.data, alpha=scaling)
    return merge_state_dict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_save_dir", type=str, default="./your_converted_ckpt_dir/", help="Source path of checkpoint")
    parser.add_argument("--lora_save_dir", type=str, default="./your_lora_ckpt_path_to_save/", help="Source path of checkpoint")
    parser.add_argument("--merge_save_dir", type=str, default="./your_ckpt_path_to_merge_saved/", help="The path where the base and LoRA weights are merged and saved")
    parser.add_argument("--lora_target_modules", type=str, nargs='+', help="The lora target modules")
    parser.add_argument("--lora_alpha", type=int, default=16, help="The lora_alpha config value")
    parser.add_argument("--lora_r", type=int, default=8, help="The lora_r config value")
    parser.add_argument("--pp_size", type=int, default=1, help="Pipeline parallel model split sizes")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor model parallel world size")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    base_save_dir = args.base_save_dir
    lora_save_dir = args.lora_save_dir
    merge_save_dir = args.merge_save_dir
    lora_target_modules = args.lora_target_modules

    lora_alpha = args.lora_alpha
    lora_r = args.lora_r
    scaling = lora_alpha / lora_r

    pp_size = args.pp_size
    tp_size = args.tp_size

    use_npu = True

    try:
        os.makedirs(merge_save_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory:{e}")

    merge_model(base_save_dir, lora_save_dir, merge_save_dir, pp_size, tp_size)
    set_directory_permissions(Path(merge_save_dir))
    print('Finished!')
