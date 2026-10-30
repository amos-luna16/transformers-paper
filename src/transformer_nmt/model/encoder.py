from __future__ import annotations

import math

from torch import Tensor, nn

from transformer_nmt.model.layers import SinusoidalPositionalEncoding, TransformerEncoderLayer
from transformer_nmt.model.masks import make_padding_mask


class TransformerEncoder(nn.Module):
    """Embedding, positional encoding and a stack of encoder layers (paper section 3.1)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_position_embeddings: int,
        norm_first: bool,
        pad_index: int,
    ) -> None:
        """Create the encoder.

        Args:
            vocab_size: Source vocabulary size.
            d_model: Model dimension.
            num_heads: Attention heads per layer.
            num_layers: Number of stacked encoder layers.
            dim_feedforward: Feed-forward hidden size.
            dropout: Dropout probability.
            max_position_embeddings: Length of the positional-encoding table.
            norm_first: Pre-LN (True) or Post-LN (False) residual blocks.
            pad_index: Padding token index (its embedding stays zero and is masked).
        """
        super().__init__()
        self.d_model = d_model
        self.pad_index = pad_index
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_index)
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model, max_position_embeddings, dropout
        )
        self.layers = nn.ModuleList(
            TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout, norm_first)
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()

    def forward(self, source_tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Encode ``source_tokens`` of shape ``[batch, source_len]``.

        Returns:
            The memory tensor ``[batch, source_len, d_model]`` and the source padding mask
            ``[batch, 1, 1, source_len]`` to be reused by decoder cross-attention.
        """
        source_mask = make_padding_mask(source_tokens, self.pad_index)
        x = self.positional_encoding(self.token_embedding(source_tokens) * math.sqrt(self.d_model))
        for layer in self.layers:
            x = layer(x, source_mask)
        return self.final_norm(x), source_mask
