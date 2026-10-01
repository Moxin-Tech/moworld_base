from typing import Any, Optional, List
import torch


class DebugBwd(torch.autograd.Function):
    """
    通用型反向传播梯度调试自定义函数
    功能：在前向传播原样输出输入，反向传播时打印梯度信息并支持断点调试
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        debug_rank: int = 0,
        print_info: Optional[str] = None,
        extra_info: Optional[List[Any]] = None,
    ) -> torch.Tensor:
        """
        前向传播：保存上下文信息，直接返回输入张量
        Args:
            ctx: 自动微分上下文
            x: 需要监控梯度的输入张量
            debug_rank: 调试的进程所属进程组的rank
            print_info: 打印的标识信息
            extra_info: 额外的参数信息, 比如前向的输入
        Returns:
            原样返回输入张量 x
        """
        # 仅保存必要张量，避免冗余存储（最佳实践）
        ctx.save_for_backward(x)

        # 存储非张量上下文（使用字典统一管理，更整洁）
        ctx.debug_info = {
            "debug_rank": debug_rank,
            "print_info": print_info,
            "extra_info": extra_info,
        }

        return x

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        """
        反向传播：打印梯度信息并触发断点，返回原始梯度
        """

        # 从上下文获取数据
        x = ctx.saved_tensors[0]
        debug_rank = ctx.debug_info["debug_rank"]
        print_info = ctx.debug_info["print_info"]
        extra_info = ctx.debug_info["extra_info"]


        rank_id = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if debug_rank == rank_id or debug_rank < 0:
            log_parts = [f"[DebugBwd] {print_info or ' '}"]
            log_parts.append(f"shape={grad_output.shape}")
            log_parts.append(f"device={grad_output.device}")
            log_parts.append(f"grad_sum={grad_output.sum():.6f}")
            print_str = " | ".join(log_parts)
            print(f"\n{print_str}\nGrad Output: {grad_output}\n", flush=True)

            # 可以在反向加入断点调试

        # 返回对应输入的梯度，多余参数返回 None
        return grad_output, None, None, None


def debug_fn(
    x: torch.Tensor,
    debug_rank: int = 0,
    print_info: Optional[str] = None,
    extra_info: Optional[List[Any]] = None,
) -> torch.Tensor:
    """
    【对外易用接口】梯度调试包装函数
    用法：直接包裹需要监控梯度的张量，不改变前向计算逻辑

    Example:
        x = debug_fn(x, print_info="conv1_input") # 获取x的反向梯度
        out = model(x)
    """
    return DebugBwd.apply(x, debug_rank, print_info, extra_info)
