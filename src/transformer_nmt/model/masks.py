from __future__ import annotations

import torch
from torch import Tensor


def make_padding_mask(tokens: Tensor, pad_index: int) -> Tensor:
    """Build a key-padding attention mask where ``True`` marks positions that may be attended.

    Args:
        tokens: Token indices of shape ``[batch, length]``.
        pad_index: Index of the padding token.

    Returns:
        Boolean mask of shape ``[batch, 1, 1, length]`` broadcastable over heads and queries.
    """
    return (tokens != pad_index)[:, None, None, :]


def make_causal_mask(length: int, device: torch.device | None = None) -> Tensor:
    """Build a lower-triangular mask that blocks attention to future positions.

    Args:
        length: Sequence length.
        device: Device of the resulting tensor.

    Returns:
        Boolean mask of shape ``[1, 1, length, length]`` with ``True`` where attention is allowed.
    """
    return torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))[None, None]


def make_target_mask(tokens: Tensor, pad_index: int) -> Tensor:
    """Combine causal and padding masks for decoder self-attention.

    Args:
        tokens: Target token indices of shape ``[batch, length]``.
        pad_index: Index of the padding token.

    Returns:
        Boolean mask of shape ``[batch, 1, length, length]``.
    """
    return make_causal_mask(tokens.size(1), tokens.device) & make_padding_mask(tokens, pad_index)
