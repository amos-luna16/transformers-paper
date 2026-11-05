from __future__ import annotations

from torch import Tensor, nn

from transformer_nmt.model.attention import MultiHeadAttention
from transformer_nmt.model.layers import FeedForward


class EncoderBlock(nn.Module):
    """A single layer of the Transformer encoder: self-attention then feed-forward.

    Layer normalisation is applied before each sub-layer (pre-LN) and a residual connection
    with dropout wraps it, as in the reference encoder block.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        """Create the sub-layers.

        Args:
            d_model: Embedding dimension.
            nhead: Number of attention heads.
            dim_feedforward: Hidden size of the feed-forward network.
            dropout: Dropout probability.
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, nhead, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dim_feedforward, dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Transform ``x`` of shape ``[batch_size, seq_len, d_model]``.

        Args:
            x: Input representations.
            padding_mask: Mask from ``create_padding_mask`` hiding padded keys.

        Returns:
            Tensor with the same shape as ``x``.
        """
        x_norm = self.ln1(x)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm, padding_mask)
        x = x + self.dropout1(attn_out)
        x = x + self.dropout2(self.ffn(self.ln2(x)))
        return x


class DecoderBlock(nn.Module):
    """A single layer of the Transformer decoder.

    Masked self-attention over the target, cross-attention over the encoder memory, then the
    feed-forward network; each sub-layer is pre-normalised and wrapped in a residual connection.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        """Create the sub-layers.

        Args:
            d_model: Embedding dimension.
            nhead: Number of attention heads.
            dim_feedforward: Hidden size of the feed-forward network.
            dropout: Dropout probability.
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, nhead, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_mha = MultiHeadAttention(d_model, nhead, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dim_feedforward, dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Decode ``x`` against the encoder ``memory``.

        Args:
            x: Target representations of shape ``[batch_size, tgt_len, d_model]``.
            memory: Encoder output of shape ``[batch_size, src_len, d_model]``.
            tgt_mask: Causal + padding mask from ``make_target_mask``.
            memory_padding_mask: Source padding mask for cross-attention.

        Returns:
            Tensor with the same shape as ``x``.
        """
        x_norm = self.ln1(x)
        self_attn_out, _ = self.mha(x_norm, x_norm, x_norm, tgt_mask)
        x = x + self.dropout1(self_attn_out)
        cross_attn_out, _ = self.cross_mha(self.ln2(x), memory, memory, memory_padding_mask)
        x = x + self.dropout2(cross_attn_out)
        x = x + self.dropout3(self.ffn(self.ln3(x)))
        return x
