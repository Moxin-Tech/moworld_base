import torch


def load_balancing_loss_func_optimized(
        gate_logits: torch.Tensor | tuple[torch.Tensor] | None,
        num_experts: int | None = None,
        top_k: int = 2,
        attention_mask: torch.Tensor | None = None,
        context_parallel_group=None,
) -> torch.Tensor | int:
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    num_layers = len(gate_logits)
    if num_layers == 0:
        return 0

    compute_device = gate_logits[0].device

    tokens_selected = torch.zeros(top_k, num_experts, device=compute_device)

    if attention_mask is None:
        tokens_total = torch.zeros(top_k, num_experts, device=compute_device)
    else:
        tokens_total = torch.zeros(top_k, num_experts, device=compute_device, dtype=attention_mask.dtype)

    expert_attention_mask = None

    for layer_gate in gate_logits:
        routing_weights = torch.nn.functional.softmax(layer_gate, dim=-1)  # [batch*seq_len, num_experts]
        _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)  # [batch*seq_len, top_k]
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)  # [batch*seq_len, top_k, num_experts]

        if attention_mask is None:
            num_tokens = layer_gate.shape[0]  # batch_size * sequence_length
            if expert_attention_mask is None or expert_attention_mask.shape[0] != num_tokens:
                expert_attention_mask = torch.ones(
                    num_tokens, top_k, num_experts,
                    device=compute_device, dtype=torch.float32
                ).reshape(-1, top_k, num_experts)
                layer_tokens_total = torch.sum(expert_attention_mask, dim=0)

            layer_tokens_selected = torch.sum(expert_mask.float(), dim=0)
        else:
            batch_size, sequence_length = attention_mask.shape
            if expert_attention_mask is None:
                expert_attention_mask = (
                    attention_mask[None, :, :, None, None]
                    .expand((1, batch_size, sequence_length, top_k, num_experts))
                    .reshape(-1, top_k, num_experts)
                    .to(compute_device)
                )
                layer_tokens_total = torch.sum(expert_attention_mask, dim=0)

            layer_tokens_selected = torch.sum(expert_mask.float() * expert_attention_mask, dim=0)

        tokens_selected += layer_tokens_selected
        tokens_total += layer_tokens_total

    if context_parallel_group is not None and torch.distributed.get_world_size(group=context_parallel_group) > 1:
        torch.distributed.all_reduce(
            tokens_total,
            op=torch.distributed.ReduceOp.SUM,
            group=context_parallel_group
        )

        torch.distributed.all_reduce(
            tokens_selected,
            op=torch.distributed.ReduceOp.SUM,
            group=context_parallel_group
        )

    tokens_per_expert = tokens_selected / tokens_total

    # calculate router_prob_per_expert
    compute_device = gate_logits[0].device
    routing_weights = torch.cat([torch.nn.functional.softmax(layer_gate, dim=-1).to(compute_device) for layer_gate in gate_logits], dim=0)

    if attention_mask is not None:
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )
        router_selected = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0)
        router_total = torch.sum(router_per_expert_attention_mask, dim=0)
    else:
        num_tokens = gate_logits[0].shape[0]  # batch_size * sequence_length
        router_per_expert_attention_mask = torch.ones(
            num_layers, num_tokens, num_experts,
            device=compute_device, dtype=torch.float32
        ).reshape(-1, num_experts)
        router_selected = torch.sum(routing_weights, dim=0)
        router_total = torch.sum(router_per_expert_attention_mask, dim=0)

    if context_parallel_group is not None and torch.distributed.get_world_size(group=context_parallel_group) > 1:
        torch.distributed.all_reduce(
            router_total,
            op=torch.distributed.ReduceOp.SUM,
            group=context_parallel_group
        )

    router_prob_per_expert = router_selected / router_total

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts
