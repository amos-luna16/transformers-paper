from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product attention (Vaswani et al., 2017, section 3.2).

    The fast path delegates to ``torch.nn.functional.scaled_dot_product_attention`` so that
    fused kernels (FlashAttention / memory-efficient attention) are used on GPUs. A reference
    path computes the softmax explicitly when attention weights are requested.
    """

    def __init__(
        self, d_model: int, num_heads: int, dropout: float = 0.0, bias: bool = True
    ) -> None:
        """Create the projection layers.

        Args:
            d_model: Model (embedding) dimension.
            num_heads: Number of parallel attention heads; must divide ``d_model``.
            dropout: Dropout probability applied to the attention weights.
            bias: Whether the projections carry bias terms.
        """
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_p = dropout
        self.query_projection = nn.Linear(d_model, d_model, bias=bias)
        self.key_projection = nn.Linear(d_model, d_model, bias=bias)
        self.value_projection = nn.Linear(d_model, d_model, bias=bias)
        self.output_projection = nn.Linear(d_model, d_model, bias=bias)

    def _split_heads(self, x: Tensor) -> Tensor:
        """Reshape ``[batch, length, d_model]`` into ``[batch, heads, length, head_dim]``."""
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """Reshape ``[batch, heads, length, head_dim]`` back into ``[batch, length, d_model]``."""
        batch, _, length, _ = x.shape
        return x.transpose(1, 2).reshape(batch, length, self.d_model)

    def _reference_attention(
        self, query: Tensor, key: Tensor, value: Tensor, attention_mask: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        """Explicit softmax attention that also returns the attention weights."""
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        if attention_mask is not None:
            # Match SDPA's safe softmax: a query with no attendable key gets zero weights.
            weights = weights.masked_fill(~attention_mask.any(dim=-1, keepdim=True), 0.0)
        weights = F.dropout(weights, p=self.dropout_p, training=self.training)
        return torch.matmul(weights, value), weights

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None = None,
        *,
        return_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """Attend from ``query`` positions to ``key``/``value`` positions.

        Args:
            query: Tensor of shape ``[batch, query_len, d_model]``.
            key: Tensor of shape ``[batch, key_len, d_model]``.
            value: Tensor of shape ``[batch, key_len, d_model]``.
            attention_mask: Boolean mask broadcastable to ``[batch, heads, query_len, key_len]``
                where ``True`` allows attention and ``False`` blocks it.
            return_weights: Compute and return per-head attention weights (slower path).

        Returns:
            The attended output of shape ``[batch, query_len, d_model]`` and, when requested,
            the attention weights of shape ``[batch, heads, query_len, key_len]``.
        """
        q = self._split_heads(self.query_projection(query))
        k = self._split_heads(self.key_projection(key))
        v = self._split_heads(self.value_projection(value))

        weights: Tensor | None = None
        if return_weights:
            context, weights = self._reference_attention(q, k, v, attention_mask)
        else:
            context = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
        return self.output_projection(self._merge_heads(context)), weights
