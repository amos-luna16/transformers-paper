from __future__ import annotations

import torch
from torch import Tensor, nn

from transformer_nmt.config import ModelSettings
from transformer_nmt.model.decoder import Decoder
from transformer_nmt.model.encoder import Encoder


class EncoderDecoder(nn.Module):
    """Complete encoder-decoder translation model combining ``Encoder`` and ``Decoder``."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        config: ModelSettings,
        pad_idx: int = 0,
    ) -> None:
        """Assemble the encoder and decoder from ``config``.

        Args:
            src_vocab_size: Source vocabulary size.
            tgt_vocab_size: Target vocabulary size.
            config: Architecture hyper-parameters.
            pad_idx: Padding index shared by both vocabularies.
        """
        super().__init__()
        self.config = config
        self.pad_idx = pad_idx
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=config.d_model,
            nhead=config.num_heads,
            num_layers=config.num_encoder_layers,
            dim_feedforward=config.dim_feedforward,
            max_len=config.max_position_embeddings,
            dropout=config.dropout,
            pad_idx=pad_idx,
        )
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=config.d_model,
            nhead=config.num_heads,
            num_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            max_len=config.max_position_embeddings,
            dropout=config.dropout,
            pad_idx=pad_idx,
            tie_embeddings=config.tie_target_embeddings,
        )
        self.reset_parameters()

    def reset_parameters(
        self,
    ) -> None:
        """Xavier-initialise the weight matrices and zero the padding embeddings."""
        for name, parameter in self.named_parameters():
            if parameter.dim() > 1 and "token_emb" not in name:
                nn.init.xavier_uniform_(parameter)
        for embedding in (self.encoder.token_emb, self.decoder.token_emb):
            nn.init.normal_(embedding.weight, mean=0.0, std=self.config.d_model**-0.5)
            with torch.no_grad():
                embedding.weight[self.pad_idx].zero_()

    def count_parameters(
        self,
    ) -> int:
        """Return the number of unique trainable parameters (tied weights counted once)."""
        unique = {id(p): p for p in self.parameters() if p.requires_grad}
        return sum(p.numel() for p in unique.values())

    def encode(
        self,
        src: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run the encoder; see :meth:`Encoder.forward`."""
        return self.encoder(src)

    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        memory_padding_mask: Tensor,
    ) -> Tensor:
        """Run the decoder; see :meth:`Decoder.forward`."""
        return self.decoder(tgt, memory, memory_padding_mask)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
    ) -> Tensor:
        """Teacher-forced forward pass.

        Args:
            src: Source token indices of shape ``[batch_size, src_len]``.
            tgt: Decoder input tokens of shape ``[batch_size, tgt_len]``.

        Returns:
            Logits of shape ``[batch_size, tgt_len, tgt_vocab_size]``.
        """
        memory, src_padding_mask = self.encode(src)
        return self.decode(tgt, memory, src_padding_mask)

    @torch.no_grad()
    def greedy_decode(
        self,
        src: Tensor,
        *,
        sos_idx: int,
        eos_idx: int,
        max_len: int,
    ) -> Tensor:
        """Translate a batch token by token, always taking the most likely next token.

        Args:
            src: Source token indices of shape ``[batch_size, src_len]``.
            sos_idx: Start-of-sequence index that primes the decoder.
            eos_idx: End-of-sequence index that stops a sentence.
            max_len: Maximum number of generated tokens.

        Returns:
            Generated indices of shape ``[batch_size, generated_len]`` without ``<sos>``;
            positions after a sentence's ``<eos>`` are filled with ``pad_idx``.
        """
        was_training = self.training
        self.eval()
        try:
            memory, src_padding_mask = self.encode(src)
            batch_size, device = src.size(0), src.device
            generated = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            for _ in range(max_len):
                logits = self.decode(generated, memory, src_padding_mask)[:, -1, :]
                next_tokens = logits.argmax(dim=-1)
                next_tokens = torch.where(
                    finished, torch.full_like(next_tokens, self.pad_idx), next_tokens
                )
                generated = torch.cat([generated, next_tokens[:, None]], dim=1)
                finished |= next_tokens == eos_idx
                if bool(finished.all()):
                    break
        finally:
            self.train(was_training)
        return generated[:, 1:]
