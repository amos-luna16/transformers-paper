from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product attention 
    Extended to several heads and to cross-attention 
    (queries from one sequence, keys/values from another).
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
    ) -> None:
        """Create the projection layers.

        Args:
            d_model: Embedding dimension.
            nhead: Number of attention heads; must divide ``d_model``.
            dropout: Dropout applied to the attention weights.
        """
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout_p = dropout
        self.to_q = nn.Linear(d_model, d_model)
        self.to_k = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)
        self.to_out = nn.Linear(d_model, d_model)

    def _split_heads(
        self,
        x: Tensor,
    ) -> Tensor:
        """Reshape [batch, len, d_model] into [batch, nhead, len, head_dim]"""
        batch_size, length, _ = x.shape
        return x.view(batch_size, length, self.nhead, self.head_dim).transpose(1, 2)

    def _merge_heads(
        self,
        x: Tensor,
    ) -> Tensor:
        """Reshape [batch, nhead, len, head_dim] back into [batch, len, d_model]."""
        batch_size, _, length, _ = x.shape
        return x.transpose(1, 2).reshape(batch_size, length, self.d_model)

    def _manual_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Explicit ``softmax(QK^T / sqrt(d_k)) V`` that also returns the attention weights."""
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        if attn_mask is not None:
            # A query with no allowed key gets zero weights instead of NaN.
            attn = attn.masked_fill(~attn_mask.any(dim=-1, keepdim=True), 0.0)
        attn = F.dropout(attn, p=self.dropout_p, training=self.training)
        return torch.matmul(attn, v), attn

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor | None = None,
        *,
        need_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """Attend from ``query`` positions to ``key``/``value`` positions.

        Args:
            query: Tensor of shape ``[batch, query_len, d_model]``.
            key: Tensor of shape ``[batch, key_len, d_model]``.
            value: Tensor of shape ``[batch, key_len, d_model]``.
            attn_mask: Boolean mask broadcastable to ``[batch, nhead, query_len, key_len]``
                where ``True`` allows attention.
            need_weights: Also return the attention weights (uses the slower manual path).

        Returns:
            The attended output ``[batch, query_len, d_model]`` and, when requested, the
            weights ``[batch, nhead, query_len, key_len]``.
        """
        q = self._split_heads(self.to_q(query))
        k = self._split_heads(self.to_k(key))
        v = self._split_heads(self.to_v(value))
        attn: Tensor | None = None
        if need_weights:
            out, attn = self._manual_attention(q, k, v, attn_mask)
        else:
            # Fused GPU kernel (FlashAttention) with the same result as the manual path.
            out = F.scaled_dot_product_attention(
                q, 
                k, 
                v, 
                attn_mask=attn_mask, 
                dropout_p=self.dropout_p if self.training else 0.0
            )
        return self.to_out(self._merge_heads(out)), attn
