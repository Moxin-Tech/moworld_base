import math
import random
import argparse

from PIL import Image
from torch.nn import functional as F
from transformers import AutoConfig


def random_pil_image(w, h):
    """
    生成指定宽高的纯色 PIL Image 图像

    Args:
        w: 图像宽度（像素）
        h: 图像高度（像素）

    Returns:
        PIL.Image.Image: 随机纯色的 RGB 图像
    """
    color = tuple(random.randint(0, 255) for _ in range(3))
    return Image.new('RGB', (w, h), color)


def _preprocess_image(image, image_max_pixels, image_min_pixels):
    """
    按指定像素范围缩放 + 格式转换 + 尺寸约束 + 宽高比限制

    Args:
        image (PIL.Image): 输入的 PIL 图像
        image_max_pixels (int): 图像允许的最大像素总数
        image_min_pixels (int): 图像允许的最小像素总数

    Returns:
        PIL.Image: 预处理后的标准 RGB 图像
    """
    if (image.width * image.height) > image_max_pixels:
        resize_factor = math.sqrt(image_max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < image_min_pixels:
        resize_factor = math.sqrt(image_min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    width, height = max(image.width, 28), max(image.height, 28)
    image = image.resize((width, height), resample=Image.NEAREST)

    if image.width / image.height > 200:
        width, height = image.height * 180, image.height
        image = image.resize((width, height), resample=Image.NEAREST)

    if image.height / image.width > 200:
        width, height = image.width, image.width * 180
        image = image.resize((width, height), resample=Image.NEAREST)

    return image


def regularize_images(images, **kwargs):
    """
    图像标准化预处理

    Args:
        images (list): 原始图像列表, 支持PIL对象/图片路径两种输入
        **kwargs: 透传给 _preprocess_image 的关键字参数, 如mage_max_pixels、image_min_pixels

    Returns:
        list[Image.Image]: 经过统一规整、尺寸限制、格式转换后的PIL图像列表
    """
    results = []

    for image in images:
        if not isinstance(image, Image.Image):
            with Image.open(image) as img:
                processed_img = _preprocess_image(img, **kwargs)
            results.append(processed_img)
        else:
            results.append(_preprocess_image(image, **kwargs))

    return results


def get_mm_inputs(images, hf_ckpt_path, image_max_pixels=512 * 512, image_min_pixels=1024):
    """
    加载图像处理器 + 图像正则化 + 图像张量转换

    Args:
        images: 输入图像
        hf_ckpt_path: HuggingFace模型权重路径
        image_max_pixels: 图像最大像素值上限，需与模型配置中的 image_max_pixels 参数对齐
        image_min_pixels: 图像最小像素值下限，需与模型配置中的 image_min_pixels 参数对齐

    Returns:
        经过 image_processor 处理后的 PyTorch 张量格式图像输入
    """
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(hf_ckpt_path, trust_remote_code=True)
    image_processor = processor.image_processor

    if not isinstance(images, list):
        images = [images]
    images = regularize_images(images, image_max_pixels=image_max_pixels, image_min_pixels=image_min_pixels)

    return image_processor(images, return_tensors="pt")


def num_floating_image_encoder_point_operations(hf_cfg, seq_length=None):
    """
    计算 ViT 部分的flops

    Args:
        hf_cfg: huggingface配置
        seq_length: 序列长度

    Returns:
        ViT 部分的flops
    """
    vit_cfg = hf_cfg.vision_config

    in_dim = 3 # 此处 kimi 无传参, 设为默认值 3
    patch_size = vit_cfg.patch_size
    hidden_size = vit_cfg.mm_hidden_size

    # patch_embedding，kimi的卷积核大小为(in_dim * hidden_size * (patch_size ** 2))
    patch_embedding_flops = 2 * seq_length * in_dim * hidden_size * (patch_size ** 2)

    num_layers = vit_cfg.vt_num_hidden_layers

    # attention flops
    qkv_proj_flops = 2 * seq_length * (hidden_size ** 2) * 3 * num_layers
    output_proj_flops = 2 * seq_length * (hidden_size ** 2) * num_layers
    # Q @ KT：前向包含一次矩阵乘法, 反向包含两次矩阵乘法, 且在反向重计算中需额外进行一次前向计算
    # score @ V：前向包含一次矩阵乘法, 反向包含两次矩阵乘法
    full_attention_flops = 2 * (1 + 2 + 1 + 1 + 2) * (seq_length ** 2) * hidden_size * num_layers
    attention_flops = qkv_proj_flops + full_attention_flops + output_proj_flops

    # mlp flops
    mlp_flops = 4 * seq_length * hidden_size * vit_cfg.vt_intermediate_size * num_layers

    # vit flops
    vit_flops = patch_embedding_flops + attention_flops + mlp_flops

    # projector flops
    spatial_merge_size = vit_cfg.merge_kernel_size[0]
    seq_length_projector = seq_length // (spatial_merge_size ** 2)
    hidden_size_projector = hidden_size * (spatial_merge_size ** 2)
    hidden_size_llm = vit_cfg.text_hidden_size

    projector_flops = 6 * seq_length_projector * (hidden_size_projector ** 2) \
        + 6 * seq_length_projector * hidden_size_projector * hidden_size_llm

    return vit_flops + projector_flops


def _estimate_deepseek_v3_flops(text_cfg, tokens_sum, batch_seqlens):
    """
    计算 LLM (deepseek V3) 部分的flops

    Args:
        text_cfg: 文本部分的配置
        tokens_sum: 总token数量
        batch_seqlens: 列表, batch中每条样本的序列长度 [seq_len1, seq_len2, ..., seq_lenB]

    Returns:
        LLM (deepseek V3) 部分的flops
    """
    hidden_size = text_cfg.hidden_size  # 隐藏层维度
    vocab_size = text_cfg.vocab_size  # 词表大小
    moe_intermediate_size = text_cfg.moe_intermediate_size  # MoE 专家中间维度
    num_hidden_layers = text_cfg.num_hidden_layers  # 总层数, 含 dense 层数和 moe 层数
    first_k_dense_replace = text_cfg.first_k_dense_replace  # 稠密层层数
    num_query_heads = text_cfg.num_attention_heads  # 注意力头数
    moe_num_expert = text_cfg.n_routed_experts  # 总专家数
    moe_topk = text_cfg.num_experts_per_tok  # 激活的专家数
    share_expert_num = text_cfg.n_shared_experts  # 共享专家数

    # -------------------------------------- 1.MOE 部分--------------------------------------
    # self.gate 部分, 对应操作 F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
    # self.weight 形状: nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
    moe_gate_N = hidden_size * moe_num_expert

    # self.experts 和 self.shared_experts 部分, * 3 表示 3 次 Linear 操作, 对应 gate_proj, up_proj 和 down_proj
    # gate_proj, up_proj 和 down_proj 形状均为 nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
    # self.experts 处受 Topk 专家数量控制, 执行 moe_topk 次;
    # self.shared_experts 处受 共享专家数量控制, 执行 share_expert_num 次;
    moe_expert_mlp_N = hidden_size * moe_intermediate_size * (moe_topk + share_expert_num) * 3
    # -------------------------------------- 1.MOE 部分 --------------------------------------

    # -------------------------------------- 2.MLA 部分 --------------------------------------
    attn_linear_N = 0
    q_head_dim = text_cfg.qk_nope_head_dim + text_cfg.qk_rope_head_dim  # 不带位置编码和带位置编码的注意力头数

    # 矩阵 Q 计算
    if text_cfg.q_lora_rank is None:
        # 非LoRA场景, 对应操作: q = self.q_proj(hidden_states)
        # self.q_proj 形状为: nn.Linear(self.hidden_size, self.num_heads * self.q_head_dim, bias=False)
        attn_linear_N += hidden_size * num_query_heads * q_head_dim
    else:  # LoRA场景
        # 矩阵 QA 计算, 对应操作: self.q_a_proj(hidden_states)
        # self.q_a_proj 形状为: nn.Linear(self.hidden_size, config.q_lora_rank, bias=config.attention_bias)
        attn_linear_N += hidden_size * text_cfg.q_lora_rank
        # 矩阵 QB 计算, 对应操作: self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        # self.q_b_proj 形状为: nn.Linear(config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)
        attn_linear_N += num_query_heads * q_head_dim * text_cfg.q_lora_rank

    # 矩阵 K/V 计算
    # compressed_kv 计算, 对应操作: self.kv_a_proj_with_mqa(hidden_states)
    # self.kv_a_proj_with_mqa 形状为: nn.Linear(self.hidden_size, config.kv_lora_rank + config.qk_rope_head_dim)
    attn_linear_N += hidden_size * (text_cfg.kv_lora_rank + text_cfg.qk_rope_head_dim)

    # K/V 计算, 对应操作: self.kv_b_proj(self.kv_a_layernorm(compressed_kv)
    # self.kv_b_proj 形状为: nn.Linear(config.kv_lora_rank,self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim))
    attn_linear_N += (
        num_query_heads
        * (q_head_dim - text_cfg.qk_rope_head_dim + text_cfg.v_head_dim)
        * text_cfg.kv_lora_rank
    )

    # 矩阵 O 计算
    # 对应操作: self.o_proj(attn_output)
    # self.o_proj 形状为: nn.Linear(elf.num_heads * self.v_head_dim, self.hidden_size, bias=config.attention_bias)
    attn_linear_N += num_query_heads * text_cfg.v_head_dim * hidden_size
    # -------------------------------------- 2.MLA 部分 --------------------------------------

    # -------------------------------------- 3.lm head 部分 --------------------------------------
    # 对应操作: self.lm_head(hidden_states)
    # self.lm_head 形状为: nn.Linear(config.hidden_size,config.vocab_size, bias=False)
    emd_and_lm_head_N = vocab_size * hidden_size
    # -------------------------------------- 3.lm head 部分 --------------------------------------

    # -------------------------------------- Total Layers --------------------------------------
    # MOE 部分的 flops * MOE 层数 + Dense 部分的 flops * Dense 层数 + lm head 部分的 flops
    moe_N = (
        (moe_gate_N + moe_expert_mlp_N + attn_linear_N) * (num_hidden_layers - first_k_dense_replace)
        + (hidden_size * text_cfg.intermediate_size * 3 + attn_linear_N) * first_k_dense_replace
        + emd_and_lm_head_N
    )
    # -------------------------------------- Total Layers --------------------------------------

    # -------------------------------------- Total Tokens --------------------------------------
    # moe_N 表示单个 token 的flops, 乘以总的 token 数量;
    # 矩阵乘法含乘法和加法(即 * 2), 前向计算一次矩阵乘法, 反向在计算权重梯度和输入梯度时各计算一次矩阵乘法(共计 3 次)
    dense_N_flops = 6 * moe_N * tokens_sum
    # -------------------------------------- Total Tokens --------------------------------------

    # -------------------------------------- 4.Attention --------------------------------------
    # Attention 计算含两次矩阵乘法: Q @ KT 及 score @ V
    # Q: [B, N, S, Dq] KT: [B, N, Dq, S]  矩阵乘法: [B, N, S, Dq] × [B, N, Dq, S] ⇒ [B, N, S, S], 即 2bns^2d_q (乘法 + 加法)
    # score: [B, N, S, S] V: [B, N, S, Dv]  矩阵乘法: [B, N, S, S] × [B, N, S, Dv] ⇒ [B, N, S, Dv], 即 2bns^2d_v (乘法 + 加法)
    # Q @ KT：前向包含一次矩阵乘法, 反向包含两次矩阵乘法, 且在反向重计算中需额外进行一次前向计算
    # score @ V：前向包含一次矩阵乘法, 反向包含两次矩阵乘法
    # 采用了casual mask，计算量减半，因此除以2
    seqlen_square_sum = 0
    # for 循环相当于遍历 batch_size
    for seqlen in batch_seqlens:
        seqlen_square_sum += seqlen * seqlen * num_hidden_layers
    attn_qkv_flops = (2 * (1 + 2 + 1) * seqlen_square_sum * q_head_dim * num_query_heads + \
        2 * (1 + 2) * seqlen_square_sum * text_cfg.v_head_dim * num_query_heads) / 2
    # -------------------------------------- 4.Attention --------------------------------------

    # -------------------------------------- Sum --------------------------------------
    flops_all_token = dense_N_flops + attn_qkv_flops
    # -------------------------------------- Sum --------------------------------------

    return flops_all_token


def main(args):
    # 生成伪图片
    fake_image = random_pil_image(args.width, args.height)

    # 图像预处理
    mm_inputs = get_mm_inputs(fake_image, args.hf_ckpt_path)
    pixel_values, grid_thw = mm_inputs["pixel_values"], mm_inputs["grid_thws"]

    # 读取模型配置
    hf_cfg = AutoConfig.from_pretrained(args.hf_ckpt_path, trust_remote_code=True)

    # 图像编码器 FLOPs
    image_encoder_flops = num_floating_image_encoder_point_operations(hf_cfg, pixel_values.shape[0])
    print(f"Image encoder flops is: {image_encoder_flops}")

    # 文本解码器 FLOPs
    text_decoder_flops = _estimate_deepseek_v3_flops(
        hf_cfg.text_config,
        args.batch_size * args.text_seq_length,
        [args.text_seq_length] * args.batch_size
    )
    print(f"Text decoder flops is: {text_decoder_flops}")

    # 总 FLOPs
    total_flops = image_encoder_flops * args.batch_size * args.image_num + text_decoder_flops

    # 单个 rank 的平均 FLOPs
    average_flops = total_flops / (args.n_npu * args.latency)

    print(f"Average FLOPs is: {average_flops}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kimi-K2.5 FLOPs Calculation Tool")
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--image_num', type=int, help='Number of images')
    parser.add_argument('--width', type=int, help='Image width')
    parser.add_argument('--height', type=int, help='Image height')
    parser.add_argument('--text_seq_length', type=int, help='Text sequence length')
    parser.add_argument('--hf_ckpt_path', type=str, help='HuggingFace config path')
    parser.add_argument('--n_npu', type=int, help='Number of NPU')
    parser.add_argument('--latency', type=float, help='Elapsed time per iteration (s)')

    args = parser.parse_args()
    main(args)


"""
示例:
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python mindspeed_mm/fsdp/tools/flops_tool/flops_kimik2_5.py \
    --batch_size 16 \
    --image_num 10 \
    --width 1024 \
    --height 1024 \
    --text_seq_length 8192 \
    --hf_ckpt_path "./mindspeed_mm/fsdp/models/kimik2_5" \
    --n_npu 16 \
    --latency 34.340
"""
