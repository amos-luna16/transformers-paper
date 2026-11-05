from __future__ import annotations

import math

from torch import Tensor, nn

from transformer_nmt.model.blocks import EncoderBlock
from transformer_nmt.model.layers import PositionalEncoding
from transformer_nmt.model.masks import create_padding_mask


class Encoder(nn.Module):
    """Encoder: embeds the source sentence and runs it through a stack of encoder blocks."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        max_len: int = 256,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ) -> None:
        """Create the embedding, positional encoding and the block stack.

        Args:
            vocab_size: Source vocabulary size.
            d_model: Embedding dimension.
            nhead: Attention heads per block.
            num_layers: Number of stacked ``EncoderBlock``s.
            dim_feedforward: Hidden size of each feed-forward network.
            max_len: Length of the positional-encoding table.
            dropout: Dropout probability.
            pad_idx: Padding index (its embedding is zero and it is masked out).
        """
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(max_len=max_len, d_model=d_model)
        self.dropout = nn.Dropout(dropout)
        # The stack of layers is a list of blocks, one EncoderBlock per layer.
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)]
        )
        self.ln_final = nn.LayerNorm(d_model)  # closes the pre-LN stack

    def forward(
        self,
        src: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Encode ``src`` of shape ``[batch_size, src_len]``.

        Args:
            src: Source token indices.

        Returns:
            ``memory`` of shape ``[batch_size, src_len, d_model]`` and the source padding mask
            of shape ``[batch_size, 1, 1, src_len]`` for the decoder's cross-attention.
        """
        padding_mask = create_padding_mask(src, self.pad_idx)
        x = self.token_emb(src) * math.sqrt(self.d_model)
        x = self.dropout(x + self.pos_enc(x))
        for block in self.blocks:
            x = block(x, padding_mask)
        return self.ln_final(x), padding_mask
