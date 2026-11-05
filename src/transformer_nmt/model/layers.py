from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (paper section 3.5), precomputed as a fixed table."""

    def __init__(
        self,
        max_len: int,
        d_model: int,
    ) -> None:
        """Precompute the ``[max_len, d_model]`` sine/cosine table.

        Args:
            max_len: Maximum sequence length the table can serve.
            d_model: Embedding dimension (must be even).
        """
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Return the encodings for the first ``seq_len`` positions of ``x``.

        Args:
            x: Embeddings of shape ``[batch_size, seq_len, d_model]``.

        Returns:
            Tensor of shape ``[1, seq_len, d_model]`` to be added to ``x``.
        """
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")
        return self.pe[:, :seq_len, :].to(x.dtype)


class FeedForward(nn.Module):
    """Position-wise feed-forward network ``Linear -> ReLU -> Dropout -> Linear`` (section 3.3)."""

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        dropout: float = 0.0,
    ) -> None:
        """Create the two linear maps.

        Args:
            d_model: Input and output dimension.
            dim_feedforward: Hidden dimension.
            dropout: Dropout applied after the activation.
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Apply the network to ``x`` of shape ``[batch_size, seq_len, d_model]``."""
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
