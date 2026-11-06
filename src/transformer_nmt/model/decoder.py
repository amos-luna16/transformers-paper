from __future__ import annotations

import math

from torch import Tensor, nn

from transformer_nmt.model.blocks import DecoderBlock
from transformer_nmt.model.layers import PositionalEncoding
from transformer_nmt.model.masks import make_target_mask


class Decoder(nn.Module):
    """Decoder: embeds the target, runs the decoder blocks and projects to the vocabulary."""

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
        tie_embeddings: bool = True,
    ) -> None:
        """Create the embedding, positional encoding, block stack and output projection.

        Args:
            vocab_size: Target vocabulary size.
            d_model: Embedding dimension.
            nhead: Attention heads per block.
            num_layers: Number of stacked ``DecoderBlock``s.
            dim_feedforward: Hidden size of each feed-forward network.
            max_len: Length of the positional-encoding table.
            dropout: Dropout probability.
            pad_idx: Padding index.
            tie_embeddings: Share the embedding matrix with the output projection (paper 3.4).
        """
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(max_len=max_len, d_model=d_model)
        self.dropout = nn.Dropout(dropout)
        # The stack of layers is a list of blocks, one DecoderBlock per layer.
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(d_model, nhead, dim_feedforward, dropout) 
                for _ in range(num_layers)
            ]
        )
        self.ln_final = nn.LayerNorm(d_model)  # closes the pre-LN stack
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.output_projection.weight = self.token_emb.weight  # weight tying

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        memory_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Predict next-token logits for ``tgt`` of shape ``[batch_size, tgt_len]``.

        Args:
            tgt: Decoder input tokens (``<sos>`` followed by the target so far).
            memory: Encoder output of shape ``[batch_size, src_len, d_model]``.
            memory_padding_mask: Source padding mask of shape ``[batch_size, 1, 1, src_len]``.

        Returns:
            Logits of shape ``[batch_size, tgt_len, vocab_size]``.
        """
        tgt_mask = make_target_mask(tgt, self.pad_idx)
        x = self.token_emb(tgt) * math.sqrt(self.d_model)
        x = self.dropout(x + self.pos_enc(x))
        for block in self.blocks:
            x = block(x, memory, tgt_mask, memory_padding_mask)
        return self.output_projection(self.ln_final(x))
