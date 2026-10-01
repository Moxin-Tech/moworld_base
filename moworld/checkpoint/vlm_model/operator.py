import re
from abc import ABC, abstractmethod
from typing import Callable, Iterable, List, Tuple, Any, Dict

import torch
from torch import Tensor

from checkpoint.common.constant import DIGIT_FMT
from checkpoint.common.mm_types import STATE_DICT_T, QV_NAME_T, QKV_NAME_T


def interleaved_qkv_to_concated(megatron_qkv: Tensor, num_key_value_heads: int, split_size: List[int]) -> Tensor:
    """mindspeed-mm中的qkv排布和vllm中qkv的排布不同，因此需要重新转换，qwen2vl的转换逻辑如下：
    qkv = [nq1,k1,v1,  nq2,k2,v2,  nq3,k3,v3,  nq4,k4,v4]
    new_qkv = [nq1,nq2,nq3,nq4,k1,k2,k3,k4,v1,v2,v3,v4]
    """
    groups_qkv = [torch.split(group, split_size) for group in torch.chunk(megatron_qkv, num_key_value_heads)]
    return torch.cat([head for group_head in zip(*groups_qkv) for head in group_head])


def concated_qkv_to_interleaved(qkv: Tensor, num_key_value_heads: int, split_size: List[int]) -> Tensor:
    """
    qkv : [nq1,nq2,nq3,nq4,k1,k2,k3,k4]
    new_qkv : [nq1,k1,v1,  nq2,k2,v2,  nq3,k3,v3,  nq4,k4,v4]
    """
    qkv_split = torch.split(qkv, split_size)
    qkv_chunk_group = [torch.chunk(i, num_key_value_heads) for i in qkv_split]
    return torch.cat([i for j in zip(*qkv_chunk_group) for i in j])


def merge_to_interleaved_qkv(q: Tensor, k: Tensor, v: Tensor, group: int) -> Tensor:
    """原顺序: q=[nq1,nq2,nq3,...], k=[k1,k2,k3,...], v=[v1,v2,v3,...]
    转换后顺序: qkv = [nq1,k1,v1,nq2,k2,v2,nq3,k3,v3,...]
    这里nq1，group query attention时，多个q共享一个k/v， k1/v1即一个key head对应的head_dim维度
    """
    qkv_chunks = [torch.chunk(x, chunks=group, dim=0) for x in [q, k, v]]
    # zip(*[])转换迭代顺序，qkv_group即一组qkv，包含一个k/v头以及对应的多个q，[nq1,k1,v1]
    return torch.concat([head for qkv_group in zip(*qkv_chunks) for head in qkv_group], dim=0)


def split_to_interleaved_qkv(value: Tensor, num_heads: int, q_size: int, k_size: int, v_size: int) -> Tuple[
    Tensor, Tensor, Tensor]:
    # 分块 + zip 重组逻辑（逆向操作）
    qkv_chunks = torch.chunk(value, num_heads, dim=0)
    q_parts, k_parts, v_parts = zip(*[torch.split(chunk, [q_size, k_size, v_size]) for chunk in qkv_chunks])
    return torch.cat(q_parts), torch.cat(k_parts), torch.cat(v_parts)


def get_layer_num(pattern: str, names: Iterable[str]) -> int:
    """在names中找到pattern正则匹配到的数数字的最大值"""
    layer_ids = [int(re.findall(pattern, k)[0]) for k in names if re.match(pattern, k)]
    if len(layer_ids) == 0:
        return 0
    return max(layer_ids) + 1


def get_layer_and_expert_num(pattern: str, names: Iterable[str]) -> Tuple[int, int]:
    """仅在含有expert相关的参数时调用，匹配到的第一个数字为层数，第二个数字为专家数"""
    layer_num = max(int(re.findall(pattern, k)[0][0]) for k in names if re.match(pattern, k)) + 1
    expert_num = max(int(re.findall(pattern, k)[0][1]) for k in names if re.match(pattern, k)) + 1
    return layer_num, expert_num


def resize_embedding(tensor: Tensor, new_vocab_size: int, original_vocab_size: int = None) -> Tensor:
    """
    调整嵌入权重矩阵的大小
    Args:
        tensor: 原始嵌入权重，形状为 [vocab_size, hidden_size]
        new_vocab_size: 目标词汇表大小
        original_vocab_size: 原始词汇表大小（revert时使用）
    """
    old_vocab_size, hidden_size = tensor.shape
    # revert时强制使用原始大小
    target_size = original_vocab_size if original_vocab_size is not None else new_vocab_size

    if target_size == old_vocab_size:
        return tensor.clone()

    # 创建新权重矩阵
    new_tensor = torch.empty(target_size, hidden_size, device=tensor.device, dtype=tensor.dtype)

    # 复制原有权重
    copy_size = min(old_vocab_size, target_size)
    new_tensor[:copy_size] = tensor[:copy_size].clone()

    # 初始化新增部分（使用原分布的均值和标准差）
    if target_size > old_vocab_size:
        mean = tensor.mean().item()
        std = tensor.std().item()
        torch.nn.init.normal_(new_tensor[old_vocab_size:], mean=mean, std=std)

    return new_tensor


class Operator(ABC):

    @abstractmethod
    def apply(self, weights: STATE_DICT_T):
        """原地修改weights字典"""
        pass

    @abstractmethod
    def revert(self, weights: STATE_DICT_T):
        """原地修改weights字典"""
        pass


class ZeroWeightOp(Operator):
    """
    权重置零操作，将权重值赋值为零或向权重字典添加全零权重
    权重转换需要把q,k,v权重合并成一个完整的qkv大矩阵。
    而由于在Qwen2.5-Omni模型的音频模型中，k没有bias，所以需要将k的bias以全零权重的形式添加到权重字典
    """

    def __init__(self, name: str, shape: List[int], layers: int):
        """
        初始化权重信息

        参数:
        - name (str): 权重的名称。
        - shape (List[int]): 权重的的形状，以整数列表形式表示。
        - layers (int): 模型的层数。
        """
        self.name = name
        self.shape = shape
        self.layers = layers

    def apply(self, weights: STATE_DICT_T):
        """向weights字典中添加权重或修改权重为0"""
        for layer_num in range(self.layers):
            tensor = torch.zeros(self.shape)
            name = self.name.replace(DIGIT_FMT, str(layer_num))
            weights[name] = tensor

    def revert(self, weights: STATE_DICT_T) -> None:
        """反向操作：删除由 apply 添加的零权重键"""
        for layer_num in range(self.layers):
            name = self.name.replace(DIGIT_FMT, str(layer_num))
            # 安全删除（仅在键存在时移除）
            weights.pop(name, None)


class RenameOp(Operator):

    def __init__(self, patterns: Tuple[Tuple[str, str], ...]):
        self.patterns = patterns

    @staticmethod
    def replace_parentheses(text):
        """
        使用序号替换文本中的括号内容。

        该方法使用正则表达式匹配文本中所有的括号内容，并使用一个递增的序号替换它们。
        序号以反斜杠开头，如 \1, \2 等。

        参数:
        text -- 待处理的文本字符串。

        返回:
        替换后的文本字符串。

        doctest 示例:

        >>> RenameOp.replace_parentheses("这是一个测试 (示例) 和另一个 (例子)")
        '这是一个测试 \\1 和另一个 \\2'

        >>> RenameOp.replace_parentheses("无括号文本")
        '无括号文本'

        >>> RenameOp.replace_parentheses("(第一个) 包含多个 (括号内容) 示例")
        '\\1 包含多个 \\2 示例'
        """
        count = 0  # 初始化计数器，用于生成序号

        def repl(match):
            """
            替换匹配到的括号内容为序号。

            参数:
            match -- 正则表达式匹配到的对象。

            返回:
            包含序号的字符串，如 \1, \2 等。
            """
            nonlocal count  # 使用 nonlocal 声明，以便能够修改外部的 count 变量
            count += 1  # 计数器递增
            return f'\\{count}'  # 返回如 \1, \2 的字符串

        # 使用正则表达式匹配所有的括号内容，并使用 repl 函数进行替换
        return re.sub(r'\([^)]*\)', repl, text)

    def apply(self, weights: STATE_DICT_T):
        mapping = {}
        for weight_name in weights:
            new_name = weight_name
            for pattern, replacement in self.patterns:
                replacement = self.replace_parentheses(replacement)
                new_name = re.sub(pattern, replacement, new_name)
            if new_name != weight_name:
                mapping[weight_name] = new_name
        for old_name, new_name in mapping.items():
            weights[new_name] = weights.pop(old_name)

    def revert(self, weights: STATE_DICT_T):
        self.patterns = tuple((right, left) for left, right in self.patterns)
        self.apply(weights=weights)


class MergeOp(Operator):
    def __init__(self, raw_names: Any, new_name: str):
        self.raw_names = raw_names
        self.new_name = new_name

    @abstractmethod
    def merge(self, tensors: List[Tensor]) -> Tensor:
        pass

    @abstractmethod
    def split(self, tensor: Tensor) -> List[Tensor]:
        pass

    def apply(self, weights: STATE_DICT_T):
        layer_num = get_layer_num(self.raw_names[0], weights.keys())
        for num in range(layer_num):
            tensors = [weights.pop(name.replace(DIGIT_FMT, str(num)))
                       for name in self.raw_names
                       if name.replace(DIGIT_FMT, str(num)) in weights.keys()]
            # deepseekvl2模型第一层为dense层, 索引非0开始
            if tensors:
                merged_tensor = self.merge(tensors)
                weights[self.new_name.replace(DIGIT_FMT, str(num))] = merged_tensor

    def revert(self, weights: STATE_DICT_T) -> None:
        """反向操作：拆分合并后的权重并恢复原始键"""
        layer_num = get_layer_num(self.new_name, weights.keys())
        for num in range(layer_num):
            # 获取合并后的张量
            merged_name = self.new_name.replace(DIGIT_FMT, str(num))
            merged_tensor = weights.pop(merged_name)  # 移除合并后的键
            # 拆分回原始小矩阵列表
            split_tensors = self.split(merged_tensor)
            if len(split_tensors) != len(self.raw_names):
                raise ValueError(f"Split tensor count {len(split_tensors)} != raw names count {len(self.raw_names)}")
            # 将拆分后的张量写回原始键
            for raw_name, tensor in zip(self.raw_names, split_tensors):
                weights[raw_name.replace(DIGIT_FMT, str(num))] = tensor


class UpGateMergeOp(MergeOp):
    """GLU门控单元中权重合并逻辑，从transformers的gate up两个小矩阵到megatron的mlp中linear_fc1中的一个大矩阵是直接concat的"""

    def merge(self, tensors: List[Tensor]) -> Tensor:
        return torch.concat(tensors, dim=0)

    def split(self, tensor: Tensor) -> List[Tensor]:
        return list(torch.chunk(tensor, chunks=2, dim=0))


class QKVDirectMergeOp(MergeOp):
    """qkv中权重合并逻辑，从transformers的q1q2/k1k2/v1v2（或q1q2/k1k2v1v2）独立小矩阵到megatron的linear_qkv一个大矩阵是q1q2k1k2v1v2直接拼接的"""

    def merge(self, tensors: List[Tensor]) -> Tensor:
        return torch.concat(tensors, dim=0)

    def split(self, tensor: Tensor) -> List[Tensor]:
        raise NotImplementedError("The method has not yet been implemented")


class ExpertUpGateMergeOp(UpGateMergeOp):
    def apply(self, weights: STATE_DICT_T):
        layer_num, expert_num = get_layer_and_expert_num(self.raw_names[0], weights.keys())

        def replace_layer_expert(name, id_layer, id_expert):
            return name.replace(DIGIT_FMT, str(id_layer), 1).replace(DIGIT_FMT, str(id_expert), 1)

        for layer_id in range(layer_num):
            for expert_id in range(expert_num):
                tensors = [weights.pop(replace_layer_expert(name, layer_id, expert_id))
                           for name in self.raw_names
                           if replace_layer_expert(name, layer_id, expert_id) in weights.keys()]
                if tensors:
                    merged_tensor = self.merge(tensors)
                    weights[replace_layer_expert(self.new_name, layer_id, expert_id)] = merged_tensor


class QKVMergeOp(MergeOp):
    """qkv中权重合并逻辑，从transformers的q1q2/k1k2/v1v2三个小矩阵到megatron的linear_qkv一个大矩阵是q1k1v1q2k2v2交织拼接的"""

    def __init__(self, raw_names: QKV_NAME_T, new_name: str, group: int, q_size: int, k_size: int, v_size: int):
        self.group = group
        self.q_size = q_size
        self.k_size = k_size
        self.v_size = v_size
        super().__init__(raw_names, new_name)

    def merge(self, tensors: QKV_NAME_T) -> Tensor:
        return merge_to_interleaved_qkv(*tensors, group=self.group)

    def split(self, tensor: Tensor) -> List[Tensor]:
        qkv = split_to_interleaved_qkv(tensor, self.group, self.q_size, self.k_size, self.v_size)
        return list(qkv)


class QVToQKVMergeOp(MergeOp):
    """构造k权重，并执行qkv中权重合并逻辑，从transformers的q1q2/k1k2/v1v2三个小矩阵到megatron的linear_qkv一个大矩阵是q1k1v1q2k2v2交织拼接的"""

    def __init__(self, raw_names: QV_NAME_T, new_name: str, group: int, q_size: int, k_size: int, v_size: int):
        self.group = group
        self.q_size = q_size
        self.k_size = k_size
        self.v_size = v_size
        super().__init__(raw_names, new_name)

    def merge(self, tensors: List[Tensor]) -> Tensor:
        tensors = [tensors[0], torch.zeros_like(tensors[0]), tensors[1]]
        return merge_to_interleaved_qkv(*tensors, group=self.group)

    def split(self, tensor: Tensor) -> List[Tensor]:
        qkv = split_to_interleaved_qkv(tensor, self.group, self.q_size, self.k_size, self.v_size)
        return [qkv[0], qkv[2]]


class RelocateOp(Operator):
    """处理 QKV 权重在直接拼接和交织结构之间的转换,对于vit，transformers和megatron都是一个大矩阵，但transformer中的qkv是q1q2k1k2v1v2直接拼接，而megatron中是q1k1v1q2k2v2交织拼接"""

    def __init__(self, name: str, new_name: str, group: int, split_size: List[int]):
        """
        Args:
            name: 权重名称模板（如 "decoder.layers.{}.self_attention.linear_qkv.weight"）
            group: 分组数（如多头注意力的头数）
            split_size: 拆分维度 [q_size, k_size, v_size]
        """
        self.name = name
        self.new_name = new_name
        self.group = group
        self.split_size = split_size

    def apply(self, weights: STATE_DICT_T) -> None:
        """正向操作：将直接拼接的 Q/K/V 转换为交织结构"""
        self._apply_transformation(weights, concated_qkv_to_interleaved)

    def revert(self, weights: STATE_DICT_T) -> None:
        self.name = self.new_name
        """反向操作：将交织结构恢复为直接拼接的 Q/K/V"""
        self.split_size = [i // self.group for i in self.split_size]
        self._apply_transformation(weights, interleaved_qkv_to_concated)

    def _apply_transformation(
            self,
            weights: STATE_DICT_T,
            transform_func: Callable[[Tensor, int, List[int]], Tensor]
    ) -> None:
        """统一处理权重重新排布逻辑"""
        layer_num = get_layer_num(self.name, weights.keys())
        for num in range(layer_num):
            name = self.name.replace(DIGIT_FMT, str(num))
            tensor = weights.get(name)
            if tensor is not None:
                weights[name] = transform_func(tensor, self.group, self.split_size)


class TieOp(Operator):
    """对于tie word embedding，其output layer是直接复用的word_embedding的权重"""

    def __init__(self, raw_name: str, new_name: str):
        self.raw_name = raw_name
        self.new_name = new_name

    def apply(self, weights: STATE_DICT_T):
        weights[self.new_name] = weights.get(self.raw_name)

    def revert(self, weights: STATE_DICT_T) -> None:
        pass


class ResizeEmbedOp(Operator):
    """调整词嵌入权重（word_embeddings.weight）的形状至指定词汇表大小"""

    def __init__(self, name: str, new_vocab_size: int):
        """Args: name:嵌入权重名称, new_vocab_size:目标词汇表大小"""
        self.name = name  # 嵌入权重名称（固定单层）
        self.new_vocab_size = new_vocab_size  # 目标大小
        self.original_vocab_size = None  # 保存原始大小，用于反向恢复

    def apply(self, weights: STATE_DICT_T) -> None:
        """正向操作：将嵌入权重调整为新词汇表大小"""
        # 检查权重是否存在且为2D嵌入矩阵
        if self.name not in weights:
            raise KeyError(f"weight: {self.name} not exist in state dict")
        tensor = weights[self.name]
        if tensor.ndim != 2:
            raise ValueError(f"weight: {self.name} must be 2D tensor, but the actual shape is {tensor.shape}")

        # 保存原始大小（仅首次apply时记录）
        if self.original_vocab_size is None:
            self.original_vocab_size = tensor.shape[0]

        # 应用调整
        weights[self.name] = resize_embedding(
            tensor,
            new_vocab_size=self.new_vocab_size
        )

    def revert(self, weights: STATE_DICT_T) -> None:
        """反向操作：将嵌入权重恢复为原始大小"""
        if self.name not in weights:
            raise KeyError(f"weight: {self.name} not exist in state dict")
        if self.original_vocab_size is None:
            raise RuntimeError("No apply operation has been performed, so revert cannot be executed. ")

        tensor = weights[self.name]
        # 恢复为原始大小
        weights[self.name] = resize_embedding(
            tensor,
            new_vocab_size=None,
            original_vocab_size=self.original_vocab_size
        )


class BaseSplit(ABC):
    """Abstract base class for tensor splitting strategies."""

    @staticmethod
    @abstractmethod
    def split(tp_size: int, value: Tensor) -> List[Tensor]:
        """Split tensor into TP_SIZE partitions.

        Args:
            tp_size: Number of tensor parallel partitions
            value: Tensor to split

        Returns:
            List of split tensors, one per TP rank
        """
        pass

    @staticmethod
    @abstractmethod
    def merge(tp_values: List[Tensor]) -> Tensor:
        """Merge split tensors back into original tensor.

        Args:
            tp_values: List of tensors from TP ranks

        Returns:
            Merged tensor
        """
        pass


class ColSplit(BaseSplit):
    """按列切分权重"""

    @staticmethod
    def split(tp_size: int, value: Tensor) -> List[Tensor]:
        size_per_tp = value.shape[1] // tp_size
        return [
            value[:, rank * size_per_tp:(rank + 1) * size_per_tp]
            for rank in range(tp_size)
        ]

    @staticmethod
    def merge(tp_values: List[Tensor]) -> Tensor:
        return torch.cat(tp_values, dim=1)


class RowSplit(BaseSplit):
    """按行切分偏置"""

    @staticmethod
    def split(tp_size: int, value: Tensor) -> List[Tensor]:
        size_per_tp = value.shape[0] // tp_size
        return [
            value[rank * size_per_tp:(rank + 1) * size_per_tp]
            for rank in range(tp_size)
        ]

    @staticmethod
    def merge(tp_values: List[Tensor]) -> Tensor:
        return torch.cat(tp_values, dim=0)


def unaligned_divide(numerator: int, world_size: int, rank: int) -> int:
    res = numerator // world_size
    if rank < numerator % world_size:
        res += 1
    return res


class UnalignedEmbeddingSplit(BaseSplit):
    """按行切分嵌入权重，处理不均匀分布的情况"""

    @staticmethod
    def split(tp_size: int, value: Tensor) -> List[Tensor]:
        value_per_tp = []
        start_idx = 0
        for tp_rank in range(tp_size):
            size_per_tp = unaligned_divide(value.shape[0], tp_size, tp_rank)
            value_per_tp.append(value[start_idx:start_idx + size_per_tp])
            start_idx += size_per_tp
        return value_per_tp

    @staticmethod
    def merge(tp_values: List[Tensor]) -> Tensor:
        return torch.cat(tp_values, dim=0)


class UnalignedRowSplit(BaseSplit):
    """按行切分权重，处理不均匀分布的情况"""

    def __init__(self, num_attention_heads: int):
        self.head = num_attention_heads

    def split(self, tp_size: int, value: Tensor) -> List[Tensor]:
        values = torch.chunk(value, 3, dim=0)
        value_per_tp = []
        start_idx = 0
        for tp_rank in range(tp_size):
            heads_per_tp = unaligned_divide(self.head, tp_size, tp_rank)
            size_per_tp = heads_per_tp * (value.shape[0] // 3 // self.head)
            w_q = values[0][start_idx:start_idx + size_per_tp]
            w_k = values[1][start_idx:start_idx + size_per_tp]
            w_v = values[2][start_idx:start_idx + size_per_tp]
            value_per_tp.append(torch.cat((w_q, w_k, w_v), dim=0))
            start_idx += size_per_tp
        return value_per_tp

    @staticmethod
    def merge(tp_values: List[Tensor]) -> Tensor:
        return torch.cat(tp_values, dim=0)


class UnalignedColSplit(BaseSplit):
    """按列切分权重，处理不均匀分布的情况"""
    def __init__(self, num_attention_heads: int):
        self.head = num_attention_heads

    def split(self, tp_size: int, value: Tensor) -> List[Tensor]:
        value_per_tp = []
        start_idx = 0
        for tp_rank in range(tp_size):
            heads_per_tp = unaligned_divide(self.head, tp_size, tp_rank)
            size_per_tp = heads_per_tp * (value.shape[1] // self.head)
            value_per_tp.append(value[:, start_idx: start_idx + size_per_tp])
            start_idx += size_per_tp
        return value_per_tp

    @staticmethod
    def merge(tp_values: List[Tensor]) -> Tensor:
        return torch.cat(tp_values, dim=1)


class GLUSplit(BaseSplit):
    """GLU 通用切分基类"""

    @staticmethod
    def split(tp_size: int, value: Tensor) -> List[Tensor]:
        # 统一处理逻辑，自动适应不同维度
        size_per_tp = value.shape[0] // tp_size // 2
        gate, up = torch.chunk(value, 2, dim=0)

        return [
            torch.cat((
                gate[rank * size_per_tp: (rank + 1) * size_per_tp],
                up[rank * size_per_tp: (rank + 1) * size_per_tp]
            ), dim=0)
            for rank in range(tp_size)
        ]

    @staticmethod
    def merge(tp_values: List[Tensor]) -> Tensor:
        # 统一合并逻辑
        split_results = [torch.chunk(v, 2, dim=0) for v in tp_values]
        all_gates = [chunk[0] for chunk in split_results]
        all_ups = [chunk[1] for chunk in split_results]
        return torch.cat([
            torch.cat(all_gates, dim=0),
            torch.cat(all_ups, dim=0)
        ], dim=0)


class ExpertSplitOp(Operator):
    """moe experts split op"""
    def __init__(self, raw_name: Any, new_name: str, num_experts: int):
        self.raw_name = raw_name
        self.new_name = new_name
        self.num_experts = num_experts

    def apply(self, weights: STATE_DICT_T):
        if self.num_experts > 0:
            experts_tensor = weights.pop(self.raw_name)
            for num in range(self.num_experts):
                # 拆分专家weight
                split_tensors = experts_tensor[num]
                weights[self.new_name.replace(DIGIT_FMT, str(num))] = split_tensors.T

    def revert(self, weights: STATE_DICT_T):
        if self.num_experts > 0:
            split_tensors_list = []
            for num in range(self.num_experts):
                split_tensors = weights.pop(self.new_name.replace(DIGIT_FMT, str(num))).T
                split_tensors_list.append(split_tensors)

            ori_dtype = split_tensors_list[0].dtype
            need_trans_dtype = ori_dtype == torch.bfloat16
            if need_trans_dtype:
                split_tensors_list = [x.to(torch.float32) for x in split_tensors_list]
            merged_tensor = torch.stack(split_tensors_list, dim=0)
            if need_trans_dtype:
                merged_tensor = merged_tensor.to(ori_dtype)
            weights[self.raw_name] = merged_tensor


TP_PATTERN_T = Dict[str, BaseSplit]
