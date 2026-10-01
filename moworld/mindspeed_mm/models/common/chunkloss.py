from typing import Any, Callable, Optional, Tuple
import torch
import torch.nn.functional as F


class ChunkLoss(torch.autograd.Function):
    """
    A custom autograd function that computes a loss in chunks to reduce memory usage.

    This function splits the input hidden states along the feature dimension into chunks,
    computes the loss and gradients for each chunk separately using a provided loss function,
    and then accumulates the results. It is particularly useful when the full forward pass
    would exceed device memory limits.

    Note: Bias terms in the head (e.g., classifier bias) are not currently supported.
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: Optional[torch.Tensor],
        loss_forward: Callable,
        loss_kwargs_chunks: list[Any],
        chunk_size: int,
    ) -> torch.Tensor:
        """
        Forward pass: compute the total loss by processing hidden states in chunks.

        Args:
            ctx: Context object used to save tensors for backward pass.
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
            head_weight (torch.Tensor): Weight matrix of the final prediction head.
            head_bias (Optional[torch.Tensor]): Bias vector of the head (currently unsupported).
            loss_forward (Callable): A function that takes (hidden_chunk, weight, bias, **kwargs)
                                     and returns (loss, aux_output).
            loss_kwargs_chunks (list[Any]): A list of keyword argument dictionaries, one per chunk.
            chunk_size (int): The size (in feature dimension) of each chunk.

        Returns:
            torch.Tensor: Scalar tensor representing the accumulated loss over all chunks.
        """

        if head_bias is not None:
            raise NotImplementedError(f"head_bias is not supported in ChunkLoss")

        device = hidden_states.device
        # Initialize accumulated scalar loss on the same device
        accumulated_loss = torch.tensor(0.0, device=device)
        # Pre-allocate gradient tensors for inputs and weights
        grad_inputs = torch.empty_like(hidden_states)
        grad_weight = torch.zeros_like(head_weight)

        # Split tensors into chunks along the feature dimension (dim=1 assumed to be sequence dim)
        grad_inputs_chunks = torch.split(grad_inputs, chunk_size, dim=1)
        hidden_states_chunks = torch.split(hidden_states, chunk_size, dim=1)

        # Process each chunk independently
        for hidden_states_chunk, grad_inputs_chunk, loss_kwargs in zip(
            hidden_states_chunks, grad_inputs_chunks, loss_kwargs_chunks
        ):
            # Compute both gradients and loss value for this chunk
            (chunk_grad_input, chunk_grad_weight), (per_chunk_loss, _) = torch.func.grad_and_value(
                loss_forward, argnums=(0, 1), has_aux=True
            )(hidden_states_chunk, head_weight, None, **loss_kwargs)

            # Accumulate loss and gradients
            accumulated_loss.add_(per_chunk_loss)
            grad_inputs_chunk.copy_(chunk_grad_input)
            grad_weight.add_(chunk_grad_weight)

        # Save computed gradients for use in backward pass
        ctx.save_for_backward(grad_inputs, grad_weight)
        return accumulated_loss

    @staticmethod
    def backward(ctx, *grad_output) -> Tuple:
        """
        Backward pass: propagate upstream gradients through the precomputed gradients.

        Args:
            ctx: Context object with saved tensors from forward pass.
            grad_output: Gradient of the loss w.r.t. the output (usually a scalar).

        Returns:
            tuple: Gradients w.r.t. (hidden_states, head_weight, head_bias, loss_forward,
                   loss_kwargs_chunks, chunk_size). Only the first two are non-None.
        """
        grad_input, grad_weight = ctx.saved_tensors
        if torch.ne(grad_output[0], torch.tensor(1.0, device=grad_output[0].device)):
            grad_input = grad_input * grad_output[0]
            grad_weight = grad_weight * grad_output[0]

        return grad_input, grad_weight, None, None, None, None


def fixed_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    reduction: str = "sum",
    **kwargs,
) -> torch.Tensor:
    """
    Compute a modified cross-entropy loss that optionally normalizes the loss by a per-example or global scaling factor (alpha).
    Args:
        source (torch.Tensor): Predicted logits of shape (N, C), where C is the number of classes.
        target (torch.Tensor): Ground truth labels of shape (N,) with values in [0, C-1].
        alpha (Optional[torch.Tensor]): Optional scaling factor.
            - If scalar (0-D tensor), it globally scales the total loss.
            - If 1-D tensor of shape (N,), it provides per-example scaling factors.
            - Must be positive and non-zero.
        ignore_index (int): Specifies a target value that is ignored and does not contribute to the loss.
        reduction (str): Specifies the reduction to apply to the output:
            - `"sum"`: Sum all losses before dividing by `alpha`.
            - `"none"`: No reduction; used here to enable per-example weighting via `alpha`.
            Note: `"mean"` is not supported in this implementation.
        **kwargs: Additional keyword arguments passed to `F.cross_entropy` (though currently unused).

    Returns:
        torch.Tensor: A scalar tensor representing the normalized cross-entropy loss.
    """

    # Compute standard cross-entropy loss
    loss = torch.nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if alpha is not None:
        alpha = alpha.to(loss.device)

    if reduction == "sum":
        # Global normalization: total loss divided by scalar alpha
        loss = loss / alpha

    elif reduction == "none":
        if alpha.ndim == 0:
            # Alpha is a scalar: sum all element-wise losses, then divide
            loss = loss.sum() / alpha
        else:
            # Alpha is 1-D with shape (N,): assume N examples
            # Reshape loss to (N, -1) to group elements per example
            loss = loss.view(alpha.shape[0], -1)
            # Sum over non-batch dimensions (e.g., sequence length in token classification)
            # Normalize each example's loss by its corresponding alpha
            loss = loss.sum(1) / alpha
            loss = loss.sum()
    else:
        raise ValueError(f"Unsupported reduction mode: {reduction}. Use 'sum' or 'none'.")

    return loss


def calculate_lm_loss(
    hidden_states: torch.Tensor,
    head_weight: torch.Tensor,
    head_bias: Optional[torch.Tensor] = None,
    *,
    shift_labels: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    reduction: Optional[str] = None,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the language modeling (LM) loss using a linear output head and a modified cross-entropy function.

    This function is typically used in autoregressive language models where the prediction for each token
    is compared against the next token in the sequence. The input `hidden_states` are projected to logits
    via a linear layer (without bias if `head_bias` is None), then compared to shifted labels.

    Note:
        - The `head_bias` argument is accepted for interface compatibility but **not used** in the current implementation.
        - Label shifting (i.e., aligning predictions with next-token targets) is assumed to have been done externally;
          this function only flattens the tensors for loss computation.

    Args:
        hidden_states (torch.Tensor):
            The hidden representations from the transformer backbone, of shape (batch_size, seq_len, hidden_dim).
        head_weight (torch.Tensor):
            Weight matrix of the output classification head, of shape (vocab_size, hidden_dim).
        head_bias (Optional[torch.Tensor], optional):
            Bias vector for the output head (shape: (vocab_size,)). Currently **ignored**.
        shift_labels (torch.Tensor):
            Ground truth token IDs, already shifted to align with predictions (e.g., target[i] = input[i+1]),
            of shape (batch_size, seq_len).
        alpha (Optional[torch.Tensor], optional):
            Optional scaling factor used in `fixed_cross_entropy` for loss normalization
            (e.g., per-example or global weighting). See `fixed_cross_entropy` for details.
        ignore_index (int, optional):
            Specifies a target value that is ignored and does not contribute to the loss. Default: -100.
        reduction (Optional[str], optional):
            Reduction method for the loss ('none', 'sum', etc.). Passed directly to `fixed_cross_entropy`.
            If None, the default behavior of `fixed_cross_entropy` applies (typically 'sum').
        **kwargs (Any):
            Additional keyword arguments forwarded to `fixed_cross_entropy`.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - **loss**: Scalar tensor representing the computed LM loss.
            - **logits**: The unnormalized prediction scores of shape (batch_size * seq_len, vocab_size).

    Example:
        >>> hidden = torch.randn(2, 5, 768)
        >>> weight = torch.randn(30522, 768)
        >>> labels = torch.randint(0, 30522, (2, 5))
        >>> loss, logits = calculate_lm_loss(hidden, weight, shift_labels=labels, reduction="sum")
    """
    # Flatten labels to 1D: (batch_size * seq_len,)
    shift_labels = shift_labels.reshape(-1)

    # Flatten hidden states to (batch_size * seq_len, hidden_dim)
    hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))

    # Project to logits using only the weight (bias is intentionally omitted here)
    # Cast to float to ensure numerical stability in loss computation
    logits = F.linear(hidden_states, head_weight).float()

    # Compute the modified cross-entropy loss
    loss = fixed_cross_entropy(
        logits,
        shift_labels,
        alpha=alpha,
        ignore_index=ignore_index,
        reduction=reduction,
        **kwargs
    )

    return loss, logits


def chunk_loss(hidden_states, head_weight, head_bias, loss_forward, loss_kwargs_chunks, chunk_size):
    """
    Compute loss in chunks using the custom autograd function `ChunkLoss`.

    Args:
        hidden_states: Input tensor (e.g., from a transformer) to compute loss on.
        head_weight: Weight matrix of the output classification head.
        head_bias: Bias vector of the head.
        loss_forward: Callable that computes the loss for a given chunk.
        loss_kwargs_chunks: List of keyword arguments for `loss_forward`, one per chunk.
        chunk_size: Number of features per chunk (along dim=1).

    Returns:
        The total accumulated loss as a scalar tensor.
    """
    return ChunkLoss.apply(
        hidden_states,
        head_weight,
        head_bias,
        loss_forward,
        loss_kwargs_chunks,
        chunk_size
    )
