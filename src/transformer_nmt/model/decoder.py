from __future__ import annotations

import math

from torch import Tensor, nn

from transformer_nmt.model.layers import SinusoidalPositionalEncoding, TransformerDecoderLayer
from transformer_nmt.model.masks import make_target_mask


class TransformerDecoder(nn.Module):
    """Embedding, positional encoding, decoder layers and the vocabulary projection."""

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
        tie_embeddings: bool,
    ) -> None:
        """Create the decoder.

        Args:
            vocab_size: Target vocabulary size.
            d_model: Model dimension.
            num_heads: Attention heads per layer.
            num_layers: Number of stacked decoder layers.
            dim_feedforward: Feed-forward hidden size.
            dropout: Dropout probability.
            max_position_embeddings: Length of the positional-encoding table.
            norm_first: Pre-LN (True) or Post-LN (False) residual blocks.
            pad_index: Padding token index.
            tie_embeddings: Share the embedding matrix with the output projection.
        """
        super().__init__()
        self.d_model = d_model
        self.pad_index = pad_index
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_index)
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model, max_position_embeddings, dropout
        )
        self.layers = nn.ModuleList(
            TransformerDecoderLayer(d_model, num_heads, dim_feedforward, dropout, norm_first)
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.output_projection.weight = self.token_embedding.weight

    def forward(self, target_tokens: Tensor, memory: Tensor, memory_mask: Tensor) -> Tensor:
        """Predict next-token logits for ``target_tokens`` of shape ``[batch, target_len]``.

        Args:
            target_tokens: Decoder input tokens (shifted right, starting with ``<sos>``).
            memory: Encoder output ``[batch, source_len, d_model]``.
            memory_mask: Source padding mask ``[batch, 1, 1, source_len]``.

        Returns:
            Logits of shape ``[batch, target_len, vocab_size]``.
        """
        target_mask = make_target_mask(target_tokens, self.pad_index)
        x = self.positional_encoding(self.token_embedding(target_tokens) * math.sqrt(self.d_model))
        for layer in self.layers:
            x = layer(x, memory, target_mask, memory_mask)
        return self.output_projection(self.final_norm(x))
