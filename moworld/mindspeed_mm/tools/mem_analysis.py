"""
************************************************************
driver mem = free + reserved

allocated = OS + CANN + Driver + GE + PTA
(Currently, Driver consume 3GB)

PTA = fragmentation + Multi-stream overhead + allocated

allocated = static_mem + active_mem + worksapce

In a model,
    Optimizer: param_fp32, momentum, variance.  All are FP32
    Model:  model_param. Often is bf16/fp16

In specific module(not precisely),
    Linear: B * S * (C_in + C_out)
    Conv:   B * C_in * H_in * W_in + B * C_out * H_out * W_out
    LayerNorm: B * S * H
    Residual Connection: B * S * H

************************************************************

This code will give a demo about memory usage of Qwen2vl 7B.
"""


GB = 1024 ** 3
B = 1000000000
# train config
model_size = 7.61 * B / GB  # Qwen2VL 7B actually has 7.61B parameters

b = 1  # batch size
s_vit = 1344
s_llm = 496

bf16 = 2
fp32 = 4
tp = 2
dp = 4

# model config
vit_hidden_size = 1280
llm_hidden_size = 3584
vit_layer_num = 32
llm_layer_num = 28


hidden_state = b * bf16 * (vit_layer_num * vit_hidden_size * s_vit + llm_layer_num * llm_hidden_size * s_llm) / GB / tp

# workspace & activation
npu_apply_adam_w = 0.52
max_workspace = 0.8  # embedding_backward + embedding, usually, max workspace comes from embedding. TODO: Need more details
active_mem = hidden_state + max_workspace

# optimizer
m, v = fp32 * model_size, fp32 * model_size  # self.grad_data
fp32_param = fp32 * model_size

grad_data = fp32 * model_size / tp  # megatron/core/distributed/param_and_grad_buffer.py:366
main_params_shard = fp32_param / tp / dp  # fp32_params
optimizer = grad_data + main_params_shard + (m + v) / tp / dp

# model
param_data = bf16 * model_size / tp  # megatron/core/distributed/param_and_grad_buffer.py:360

static_mem = param_data + optimizer


total_model_mem = static_mem + active_mem

torch_reserved = 2
GE_reserved = 2

total_reserved = total_model_mem + torch_reserved + GE_reserved

print("active mem: %.2f GB" % active_mem)
print("static mem: %.2f GB" % static_mem)
print("total model mem: %.2f GB" % total_model_mem)
print("reserved memory: %.2f GB" % total_reserved)
