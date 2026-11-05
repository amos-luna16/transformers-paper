from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path

import torch
from torch import Tensor

from transformer_nmt.config import ModelSettings
from transformer_nmt.data.preprocessing import WordTokenizer, normalize_text
from transformer_nmt.data.vocabulary import Vocabulary
from transformer_nmt.model.encoder_decoder import EncoderDecoder
from transformer_nmt.training.checkpoint import (
    SETTINGS_NAME,
    SOURCE_VOCAB_NAME,
    TARGET_VOCAB_NAME,
    load_checkpoint,
)
from transformer_nmt.utils.hardware import resolve_device

logger = logging.getLogger(__name__)


def _length_penalty(
    length: int,
    alpha: float,
) -> float:
    """GNMT length normalisation ``((5 + length) / 6) ** alpha``."""
    return ((5.0 + length) / 6.0) ** alpha


class Translator:
    """Load a trained checkpoint and translate sentences with greedy or beam-search decoding."""

    def __init__(
        self,
        model: EncoderDecoder,
        source_vocab: Vocabulary,
        target_vocab: Vocabulary,
        tokenizer: WordTokenizer,
        device: torch.device,
        *,
        max_decode_len: int = 40,
        max_source_len: int | None = None,
    ) -> None:
        """Wrap an in-memory model.

        Args:
            model: Trained model (moved to ``device`` and set to eval mode here).
            source_vocab: Source-language vocabulary.
            target_vocab: Target-language vocabulary.
            tokenizer: Tokenizer used during training.
            device: Device for inference.
            max_decode_len: Maximum generated tokens per sentence.
            max_source_len: Maximum source tokens fed to the encoder (defaults to the size of
                the positional table; :meth:`from_checkpoint` uses the training-time limit).
        """
        self.model = model.to(device).eval()
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.tokenizer = tokenizer
        self.device = device
        table_len = model.config.max_position_embeddings
        self.max_decode_len = min(max_decode_len, table_len - 1)
        self.max_source_len = min(max_source_len or table_len, table_len)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        device: torch.device | None = None,
        **kwargs: int,
    ) -> Translator:
        """Rebuild the model from a checkpoint and the metadata saved next to it.

        Args:
            checkpoint_path: ``best.pt``/``last.pt`` inside a run directory that also holds
                ``settings.json``, ``source_vocab.json`` and ``target_vocab.json``.
            device: Inference device (auto-detected when omitted).
            **kwargs: Forwarded to the constructor (for example ``max_decode_len``).

        Returns:
            A ready-to-use translator.
        """
        run_dir = checkpoint_path.parent
        settings_payload = json.loads((run_dir / SETTINGS_NAME).read_text(encoding="utf-8"))
        model_settings = ModelSettings.model_validate(settings_payload["model"])
        source_vocab = Vocabulary.load(run_dir / SOURCE_VOCAB_NAME)
        target_vocab = Vocabulary.load(run_dir / TARGET_VOCAB_NAME)
        resolved_device = device or resolve_device()
        model = EncoderDecoder(
            len(source_vocab), len(target_vocab), model_settings, source_vocab.pad_index
        )
        model.load_state_dict(
            load_checkpoint(checkpoint_path, map_location="cpu")["model_state_dict"]
        )
        logger.info(
            "Loaded %s (%s parameters) on %s",
            checkpoint_path,
            f"{model.count_parameters():,}",
            resolved_device,
        )
        kwargs.setdefault("max_source_len", int(settings_payload["data"]["max_seq_len"]))
        return cls(model, source_vocab, target_vocab, WordTokenizer(), resolved_device, **kwargs)

    def _encode_source(
        self,
        sentence: str,
    ) -> Tensor | None:
        """Normalise, tokenise and index a source sentence into a ``[1, length]`` tensor.

        Returns ``None`` when nothing survives normalisation (digits, emoji, other scripts).
        """
        tokens = self.tokenizer(normalize_text(sentence))
        if not tokens:
            return None
        if len(tokens) > self.max_source_len:
            logger.warning(
                "Source truncated from %d to %d tokens", len(tokens), self.max_source_len
            )
            tokens = tokens[: self.max_source_len]
        return torch.tensor(
            [self.source_vocab.encode(tokens)], dtype=torch.long, device=self.device
        )

    def _indices_to_sentence(
        self,
        indices: Sequence[int],
    ) -> str:
        """Map target indices back to a detokenised sentence."""
        return self.tokenizer.detokenize(self.target_vocab.decode(indices))

    def translate_batch(
        self,
        sentences: Sequence[str],
    ) -> list[str]:
        """Translate several sentences with batched greedy decoding.

        Args:
            sentences: Source sentences.

        Returns:
            Translations aligned with ``sentences``; an empty string for sentences that
            contain no translatable tokens.
        """
        translations = [""] * len(sentences)
        kept: list[tuple[int, Tensor]] = []
        for index, sentence in enumerate(sentences):
            encoded = self._encode_source(sentence)
            if encoded is not None:
                kept.append((index, encoded[0]))
        if not kept:
            return translations
        length = max(seq.numel() for _, seq in kept)
        batch = torch.full(
            (len(kept), length), self.source_vocab.pad_index, dtype=torch.long, device=self.device
        )
        for row, (_, seq) in enumerate(kept):
            batch[row, : seq.numel()] = seq
        generated = self.model.greedy_decode(
            batch,
            sos_idx=self.target_vocab.sos_index,
            eos_idx=self.target_vocab.eos_index,
            max_len=self.max_decode_len,
        )
        for (index, _), row in zip(kept, generated, strict=True):
            translations[index] = self._indices_to_sentence(row.tolist())
        return translations

    @torch.no_grad()
    def beam_search(
        self,
        source: Tensor,
        *,
        beam_size: int,
        length_penalty: float,
    ) -> list[int]:
        """Beam-search decode a single encoded sentence.

        Args:
            source: ``[1, source_len]`` source indices.
            beam_size: Number of hypotheses kept per step.
            length_penalty: Exponent of the GNMT length normalisation (0 disables it).

        Returns:
            Target indices of the best hypothesis including its ``<eos>`` (if produced).
        """
        sos, eos = self.target_vocab.sos_index, self.target_vocab.eos_index
        memory, memory_mask = self.model.encode(source)
        memory = memory.expand(beam_size, -1, -1)
        memory_mask = memory_mask.expand(beam_size, -1, -1, -1)

        sequences = torch.full((1, 1), sos, dtype=torch.long, device=self.device)
        scores = torch.zeros(1, device=self.device)
        finished: list[tuple[float, list[int]]] = []

        for _ in range(self.max_decode_len):
            alive = sequences.size(0)
            logits = self.model.decode(sequences, memory[:alive], memory_mask[:alive])[:, -1, :]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            candidates = (scores[:, None] + log_probs).view(-1)
            top_scores, top_indices = candidates.topk(min(beam_size, candidates.numel()))
            vocab_size = log_probs.size(-1)
            beam_indices, token_indices = top_indices // vocab_size, top_indices % vocab_size

            next_sequences, next_scores = [], []
            for score, beam_index, token in zip(
                top_scores.tolist(), beam_indices.tolist(), token_indices.tolist(), strict=True
            ):
                hypothesis = [*sequences[beam_index].tolist(), token]
                if token == eos:
                    finished.append(
                        (score / _length_penalty(len(hypothesis) - 1, length_penalty), hypothesis)
                    )
                else:
                    next_sequences.append(hypothesis)
                    next_scores.append(score)
            finished = sorted(finished, key=lambda item: item[0], reverse=True)[:beam_size]
            if not next_sequences:
                break
            if len(finished) >= beam_size:
                # Stop once no alive beam can still beat the worst kept finished hypothesis.
                generated_len = len(next_sequences[0]) - 1
                best_alive = max(next_scores) / _length_penalty(generated_len, length_penalty)
                if finished[-1][0] >= best_alive:
                    break
            sequences = torch.tensor(next_sequences, dtype=torch.long, device=self.device)
            scores = torch.tensor(next_scores, device=self.device)
        else:
            # Length budget exhausted: alive beams compete with the finished hypotheses.
            finished.extend(
                (score / _length_penalty(len(seq) - 1, length_penalty), seq)
                for score, seq in zip(scores.tolist(), sequences.tolist(), strict=True)
            )
        _, best = max(finished, key=lambda item: item[0])
        return best[1:]

    @torch.no_grad()
    def stream_greedy(
        self,
        sentence: str,
    ) -> Iterator[str]:
        """Greedy-decode ``sentence`` and yield each target token as soon as it is produced.

        Args:
            sentence: Source sentence.

        Yields:
            Target tokens in generation order (``<eos>`` is consumed, not yielded).
        """
        source = self._encode_source(sentence)
        if source is None:
            return
        memory, memory_mask = self.model.encode(source)
        generated = [self.target_vocab.sos_index]
        for _ in range(self.max_decode_len):
            target = torch.tensor([generated], dtype=torch.long, device=self.device)
            next_token = int(self.model.decode(target, memory, memory_mask)[0, -1].argmax())
            if next_token == self.target_vocab.eos_index:
                break
            generated.append(next_token)
            yield self.target_vocab.index_to_token(next_token)

    def translate(
        self,
        sentence: str,
        *,
        beam_size: int = 4,
        length_penalty: float = 0.6,
    ) -> str:
        """Translate one sentence.

        Args:
            sentence: Source sentence in natural casing/punctuation.
            beam_size: ``1`` uses greedy decoding; larger values use beam search.
            length_penalty: Beam-search length-normalisation exponent.

        Returns:
            The translated sentence.
        """
        if beam_size <= 1:
            return self.translate_batch([sentence])[0]
        source = self._encode_source(sentence)
        if source is None:
            return ""
        indices = self.beam_search(source, beam_size=beam_size, length_penalty=length_penalty)
        return self._indices_to_sentence(indices)
