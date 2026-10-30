from __future__ import annotations

import torch
from torch import Tensor, nn

from transformer_nmt.config import ModelSettings
from transformer_nmt.model.decoder import TransformerDecoder
from transformer_nmt.model.encoder import TransformerEncoder


class Seq2SeqTransformer(nn.Module):
    """Encoder-decoder Transformer for sequence-to-sequence translation."""

    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        config: ModelSettings,
        pad_index: int,
    ) -> None:
        """Assemble the encoder and decoder from ``config``.

        Args:
            source_vocab_size: Size of the source vocabulary.
            target_vocab_size: Size of the target vocabulary.
            config: Architecture hyper-parameters.
            pad_index: Padding token index shared by both vocabularies.
        """
        super().__init__()
        self.config = config
        self.pad_index = pad_index
        self.encoder = TransformerEncoder(
            vocab_size=source_vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_encoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            max_position_embeddings=config.max_position_embeddings,
            norm_first=config.norm_first,
            pad_index=pad_index,
        )
        self.decoder = TransformerDecoder(
            vocab_size=target_vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            max_position_embeddings=config.max_position_embeddings,
            norm_first=config.norm_first,
            pad_index=pad_index,
            tie_embeddings=config.tie_target_embeddings,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise weight matrices with Xavier-uniform and zero the padding embeddings."""
        for name, parameter in self.named_parameters():
            if parameter.dim() > 1 and "embedding" not in name:
                nn.init.xavier_uniform_(parameter)
        for embedding in (self.encoder.token_embedding, self.decoder.token_embedding):
            nn.init.normal_(embedding.weight, mean=0.0, std=self.config.d_model**-0.5)
            with torch.no_grad():
                embedding.weight[self.pad_index].zero_()

    def count_parameters(self, *, trainable_only: bool = True) -> int:
        """Return the number of (unique) parameters in the model."""
        seen: set[int] = set()
        total = 0
        for parameter in self.parameters():
            if trainable_only and not parameter.requires_grad:
                continue
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            total += parameter.numel()
        return total

    def encode(self, source_tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Run the encoder; see :meth:`TransformerEncoder.forward`."""
        return self.encoder(source_tokens)

    def decode(self, target_tokens: Tensor, memory: Tensor, memory_mask: Tensor) -> Tensor:
        """Run the decoder; see :meth:`TransformerDecoder.forward`."""
        return self.decoder(target_tokens, memory, memory_mask)

    def forward(self, source_tokens: Tensor, target_tokens: Tensor) -> Tensor:
        """Teacher-forced forward pass.

        Args:
            source_tokens: ``[batch, source_len]`` source indices.
            target_tokens: ``[batch, target_len]`` decoder inputs (``<sos>`` + shifted target).

        Returns:
            Logits of shape ``[batch, target_len, target_vocab_size]``.
        """
        memory, memory_mask = self.encode(source_tokens)
        return self.decode(target_tokens, memory, memory_mask)

    @torch.no_grad()
    def greedy_decode(
        self,
        source_tokens: Tensor,
        *,
        sos_index: int,
        eos_index: int,
        max_len: int,
    ) -> Tensor:
        """Translate a batch autoregressively, always picking the most likely next token.

        Args:
            source_tokens: ``[batch, source_len]`` source indices.
            sos_index: Start-of-sequence index used to prime the decoder.
            eos_index: End-of-sequence index that stops a hypothesis.
            max_len: Maximum number of generated tokens (excluding ``<sos>``).

        Returns:
            Generated indices of shape ``[batch, generated_len]`` without the leading ``<sos>``;
            positions after a hypothesis' ``<eos>`` are filled with ``pad_index``.
        """
        was_training = self.training
        self.eval()
        try:
            memory, memory_mask = self.encode(source_tokens)
            batch_size = source_tokens.size(0)
            device = source_tokens.device
            generated = torch.full((batch_size, 1), sos_index, dtype=torch.long, device=device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            for _ in range(max_len):
                logits = self.decode(generated, memory, memory_mask)[:, -1, :]
                next_tokens = logits.argmax(dim=-1)
                next_tokens = torch.where(
                    finished, torch.full_like(next_tokens, self.pad_index), next_tokens
                )
                generated = torch.cat([generated, next_tokens[:, None]], dim=1)
                finished |= next_tokens == eos_index
                if bool(finished.all()):
                    break
        finally:
            self.train(was_training)
        return generated[:, 1:]
