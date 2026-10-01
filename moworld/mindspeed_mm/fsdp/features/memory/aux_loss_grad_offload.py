from mindspeed_mm.fsdp.features.memory.grad_offload import GradientOffload, GradientRestore

"""
Gradient Offload Wrapper for Auxiliary Loss Management

Purpose: Manage gradient offloading for auxiliary loss computation.

Usage:
1. In forward propagation: Call `restore_wrapper` where router_logits are generated
2. In loss computation: Call `offload_wrapper` where router_logits are used for auxiliary loss

Workflow:
  Generate router_logits → restore_wrapper → ... → offload_wrapper → Compute auxiliary loss
                         (Restore gradients)        (Offload gradients to CPU)
"""

_ROUTER_LOGITS_ID = 0


def offload_wrapper(router_logits):
    # The router_logits in record outputs.router_logits are in the same order as the forward execution sequence.
    gate_logits = list(router_logits)
    num_hidden_layers = len(gate_logits)
    for i, layer_gate in enumerate(gate_logits):
        # Keep the final layer gradient resident on device to prevent type-cast overhead from stalling the primary compute stream.
        if i == num_hidden_layers - 1:
            continue
        gate_logits[i] = GradientOffload.apply(layer_gate, i)
    return tuple(gate_logits)


def restore_wrapper(router_logits, num_hidden_layers):
    global _ROUTER_LOGITS_ID
    # Keep the final layer gradient resident on device to prevent type-cast overhead from stalling the primary compute stream.
    if _ROUTER_LOGITS_ID == num_hidden_layers - 1:
        return router_logits
    prefetch_keys = (_ROUTER_LOGITS_ID - 1,) if _ROUTER_LOGITS_ID > 0 else None
    router_logits = GradientRestore.apply(router_logits, _ROUTER_LOGITS_ID, prefetch_keys)
    _ROUTER_LOGITS_ID += 1
    _ROUTER_LOGITS_ID = _ROUTER_LOGITS_ID % num_hidden_layers
    return router_logits
