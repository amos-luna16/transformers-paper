from __future__ import annotations

import torch
from torch import Tensor


def create_padding_mask(
    seq: Tensor,
    pad_idx: int,
) -> Tensor:
    """Build the mask that hides padding tokens from attention.

    Note: ``True`` means "may attend" (the convention of
    ``torch.nn.functional.scaled_dot_product_attention``), the opposite of a
    ``key_padding_mask`` where ``True`` means "ignore".

    Args:
        seq: Token indices of shape ``[batch_size, seq_len]``.
        pad_idx: Index used for padding.

    Returns:
        Boolean mask of shape ``[batch_size, 1, 1, seq_len]`` that broadcasts over heads
        and query positions.
    """
    return (seq != pad_idx)[:, None, None, :]


def make_causal_mask(
    size: int,
    device: torch.device | None = None,
) -> Tensor:
    """Build the look-ahead mask that stops a position from seeing future tokens.

    Args:
        size: Sequence length.
        device: Device of the resulting tensor.

    Returns:
        Lower-triangular boolean mask of shape ``[1, 1, size, size]`` (``True`` = allowed).
    """
    return torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))[None, None]


def make_target_mask(
    tgt: Tensor,
    pad_idx: int,
) -> Tensor:
    """Combine the causal mask and the padding mask for decoder self-attention.

    Args:
        tgt: Target token indices of shape ``[batch_size, tgt_len]``.
        pad_idx: Index used for padding.

    Returns:
        Boolean mask of shape ``[batch_size, 1, tgt_len, tgt_len]``.
    """
    return make_causal_mask(tgt.size(1), tgt.device) & create_padding_mask(tgt, pad_idx)
