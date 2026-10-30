from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from transformer_nmt.model.attention import MultiHeadAttention


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sine/cosine positional encodings added to token embeddings (paper section 3.5)."""

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.0) -> None:
        """Pre-compute the encoding table.

        Args:
            d_model: Embedding dimension (must be even).
            max_len: Longest sequence the table can serve.
            dropout: Dropout applied after adding the positions.
        """
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10_000.0) / d_model)
        )
        table = torch.zeros(max_len, d_model)
        table[:, 0::2] = torch.sin(position * div_term)
        table[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("table", table.unsqueeze(0), persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings: Tensor) -> Tensor:
        """Add positional information to ``embeddings`` of shape ``[batch, length, d_model]``."""
        length = embeddings.size(1)
        if length > self.table.size(1):
            raise ValueError(f"Sequence length {length} exceeds max_len {self.table.size(1)}")
        return self.dropout(embeddings + self.table[:, :length].to(embeddings.dtype))


class PositionwiseFeedForward(nn.Module):
    """Two-layer position-wise feed-forward network with ReLU (paper section 3.3)."""

    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.0) -> None:
        """Create the two linear maps.

        Args:
            d_model: Input/output dimension.
            dim_feedforward: Hidden dimension.
            dropout: Dropout applied after the activation.
        """
        super().__init__()
        self.linear_in = nn.Linear(d_model, dim_feedforward)
        self.linear_out = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Apply ``linear_out(dropout(relu(linear_in(x))))``."""
        return self.linear_out(self.dropout(torch.relu(self.linear_in(x))))


class TransformerEncoderLayer(nn.Module):
    """Self-attention followed by a feed-forward block, each wrapped in residual + LayerNorm."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        norm_first: bool,
    ) -> None:
        """Create the sub-layers.

        Args:
            d_model: Model dimension.
            num_heads: Attention heads.
            dim_feedforward: Feed-forward hidden size.
            dropout: Dropout probability for residual branches and attention weights.
            norm_first: Apply LayerNorm before each sub-layer (Pre-LN) instead of after (Post-LN).
        """
        super().__init__()
        self.norm_first = norm_first
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, dim_feedforward, dropout)
        self.attention_norm = nn.LayerNorm(d_model)
        self.feed_forward_norm = nn.LayerNorm(d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.feed_forward_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, source_mask: Tensor | None) -> Tensor:
        """Transform ``x`` of shape ``[batch, length, d_model]`` under ``source_mask``."""
        if self.norm_first:
            normed = self.attention_norm(x)
            x = x + self.attention_dropout(
                self.self_attention(normed, normed, normed, source_mask)[0]
            )
            x = x + self.feed_forward_dropout(self.feed_forward(self.feed_forward_norm(x)))
            return x
        x = self.attention_norm(
            x + self.attention_dropout(self.self_attention(x, x, x, source_mask)[0])
        )
        return self.feed_forward_norm(x + self.feed_forward_dropout(self.feed_forward(x)))


class TransformerDecoderLayer(nn.Module):
    """Masked self-attention, encoder-decoder cross-attention and a feed-forward block."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        norm_first: bool,
    ) -> None:
        """Create the sub-layers.

        Args:
            d_model: Model dimension.
            num_heads: Attention heads.
            dim_feedforward: Feed-forward hidden size.
            dropout: Dropout probability for residual branches and attention weights.
            norm_first: Apply LayerNorm before each sub-layer (Pre-LN) instead of after (Post-LN).
        """
        super().__init__()
        self.norm_first = norm_first
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, dim_feedforward, dropout)
        self.self_attention_norm = nn.LayerNorm(d_model)
        self.cross_attention_norm = nn.LayerNorm(d_model)
        self.feed_forward_norm = nn.LayerNorm(d_model)
        self.self_attention_dropout = nn.Dropout(dropout)
        self.cross_attention_dropout = nn.Dropout(dropout)
        self.feed_forward_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        target_mask: Tensor | None,
        memory_mask: Tensor | None,
    ) -> Tensor:
        """Decode ``x`` (``[batch, target_len, d_model]``) against encoder ``memory``.

        Args:
            x: Target embeddings.
            memory: Encoder output of shape ``[batch, source_len, d_model]``.
            target_mask: Causal + padding mask for self-attention.
            memory_mask: Source padding mask for cross-attention.

        Returns:
            Updated target representations with the same shape as ``x``.
        """
        if self.norm_first:
            normed = self.self_attention_norm(x)
            x = x + self.self_attention_dropout(
                self.self_attention(normed, normed, normed, target_mask)[0]
            )
            normed = self.cross_attention_norm(x)
            x = x + self.cross_attention_dropout(
                self.cross_attention(normed, memory, memory, memory_mask)[0]
            )
            return x + self.feed_forward_dropout(self.feed_forward(self.feed_forward_norm(x)))
        x = self.self_attention_norm(
            x + self.self_attention_dropout(self.self_attention(x, x, x, target_mask)[0])
        )
        x = self.cross_attention_norm(
            x
            + self.cross_attention_dropout(self.cross_attention(x, memory, memory, memory_mask)[0])
        )
        return self.feed_forward_norm(x + self.feed_forward_dropout(self.feed_forward(x)))
