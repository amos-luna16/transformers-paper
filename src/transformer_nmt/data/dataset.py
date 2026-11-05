from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from transformer_nmt.data.preprocessing import normalize_text
from transformer_nmt.data.vocabulary import Vocabulary

SentencePair = tuple[str, str]


def load_sentence_pairs(
    path: Path,
    max_pairs: int,
) -> list[SentencePair]:
    """Read ``(source, target)`` pairs from a tab-separated file.

    Args:
        path: TSV file with at least two columns per line.
        max_pairs: Maximum number of pairs to read from the top of the file.

    Returns:
        The raw sentence pairs.
    """
    pairs: list[SentencePair] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            columns = line.rstrip("\n").split("\t")
            if len(columns) < 2:
                continue
            pairs.append((columns[0], columns[1]))
            if len(pairs) >= max_pairs:
                break
    return pairs


def normalize_and_filter_pairs(
    pairs: Sequence[SentencePair],
    tokenizer: Callable[[str], list[str]],
    max_seq_len: int,
) -> list[SentencePair]:
    """Normalise both sides of every pair and drop pairs that are too long.

    The target side must fit together with ``<sos>`` and ``<eos>`` in ``max_seq_len``
    positions, so it is limited to ``max_seq_len - 2`` tokens.

    Args:
        pairs: Raw sentence pairs.
        tokenizer: Tokenizer used to measure sentence length.
        max_seq_len: Maximum sequence length handled by the model.

    Returns:
        Normalised pairs within the length budget, in the original order.
    """
    kept: list[SentencePair] = []
    for source, target in pairs:
        source_norm, target_norm = normalize_text(source), normalize_text(target)
        source_tokens, target_tokens = tokenizer(source_norm), tokenizer(target_norm)
        if not source_tokens or not target_tokens:
            continue
        if len(source_tokens) > max_seq_len or len(target_tokens) > max_seq_len - 2:
            continue
        kept.append((source_norm, target_norm))
    return kept


def deduplicate_pairs(
    pairs: Sequence[SentencePair],
) -> list[SentencePair]:
    """Drop exact duplicate pairs, keeping the first occurrence in the original order."""
    return list(dict.fromkeys(pairs))


def split_pairs(
    pairs: Sequence[SentencePair],
    validation_fraction: float,
    seed: int,
) -> tuple[list[SentencePair], list[SentencePair]]:
    """Deterministic split in which no source sentence appears in both train and validation.

    Tatoeba contains many English sentences with several French translations. Pairs are grouped
    by source sentence, the groups are shuffled with ``seed`` and whole groups are assigned to
    validation until ``validation_fraction`` of the pairs is reached, so validation metrics are
    never inflated by memorised sources.

    Args:
        pairs: Pairs to split.
        validation_fraction: Fraction of pairs assigned to validation.
        seed: Seed of the shuffle.

    Returns:
        ``(train_pairs, validation_pairs)``.
    """
    groups: dict[str, list[SentencePair]] = {}
    for pair in pairs:
        groups.setdefault(pair[0], []).append(pair)
    sources = list(groups)
    random.Random(seed).shuffle(sources)
    validation_size = int(round(len(pairs) * validation_fraction))
    train: list[SentencePair] = []
    validation: list[SentencePair] = []
    for source in sources:
        bucket = validation if len(validation) < validation_size else train
        bucket.extend(groups[source])
    return train, validation


class TranslationDataset(Dataset[tuple[Tensor, Tensor]]):
    """Pre-tokenised sentence pairs stored as index tensors.

    Source sequences carry no special tokens; target sequences are wrapped in
    ``<sos>``/``<eos>`` so the decoder learns to start and stop.
    """

    def __init__(
        self,
        pairs: Sequence[SentencePair],
        tokenizer: Callable[[str], list[str]],
        source_vocab: Vocabulary,
        target_vocab: Vocabulary,
        max_seq_len: int,
    ) -> None:
        """Encode every pair eagerly so ``__getitem__`` is a cheap lookup.

        Args:
            pairs: Normalised sentence pairs.
            tokenizer: Tokenizer shared by both languages.
            source_vocab: Vocabulary of the source language.
            target_vocab: Vocabulary of the target language.
            max_seq_len: Hard limit on encoded length (sequences are truncated).
        """
        self.pairs = list(pairs)
        self._sources: list[Tensor] = []
        self._targets: list[Tensor] = []
        for source, target in self.pairs:
            source_ids = source_vocab.encode(tokenizer(source))[:max_seq_len]
            target_ids = target_vocab.encode(tokenizer(target), add_special_tokens=True)
            if len(target_ids) > max_seq_len:
                target_ids = [*target_ids[: max_seq_len - 1], target_vocab.eos_index]
            self._sources.append(torch.tensor(source_ids, dtype=torch.long))
            self._targets.append(torch.tensor(target_ids, dtype=torch.long))

    def __len__(self) -> int:
        """Return the number of pairs."""
        return len(self._sources)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[Tensor, Tensor]:
        """Return the ``(source_ids, target_ids)`` tensors of one pair."""
        return self._sources[index], self._targets[index]


@dataclass(frozen=True)
class PadCollator:
    """Pad variable-length pairs into rectangular ``(source, target)`` batches."""

    pad_index: int
    pad_to_multiple_of: int = 1

    def _padded_length(
        self,
        length: int,
    ) -> int:
        """Round ``length`` up to the configured multiple."""
        multiple = self.pad_to_multiple_of
        return ((length + multiple - 1) // multiple) * multiple

    def _pad(
        self,
        sequences: Sequence[Tensor],
    ) -> Tensor:
        """Right-pad sequences into a ``[batch, length]`` tensor."""
        length = self._padded_length(max(seq.numel() for seq in sequences))
        batch = torch.full((len(sequences), length), self.pad_index, dtype=torch.long)
        for row, seq in enumerate(sequences):
            batch[row, : seq.numel()] = seq
        return batch

    def __call__(
        self,
        examples: Sequence[tuple[Tensor, Tensor]],
    ) -> tuple[Tensor, Tensor]:
        """Collate dataset items into padded source and target tensors."""
        sources, targets = zip(*examples, strict=True)
        return self._pad(sources), self._pad(targets)


def build_dataloader(
    dataset: TranslationDataset,
    *,
    batch_size: int,
    shuffle: bool,
    collator: PadCollator,
    num_workers: int,
    distributed: bool,
    seed: int,
    pin_memory: bool,
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DistributedSampler[int] | None]:
    """Create a DataLoader, sharding the dataset across ranks when ``distributed`` is set.

    Args:
        dataset: Dataset to iterate.
        batch_size: Per-device batch size.
        shuffle: Whether to shuffle every epoch.
        collator: Padding collate function.
        num_workers: DataLoader worker processes.
        distributed: Use a ``DistributedSampler`` (requires an initialised process group).
        seed: Shuffle seed shared by all ranks.
        pin_memory: Pin host memory for faster host-to-device copies.

    Returns:
        The loader and its sampler (``None`` when not distributed). Call
        ``sampler.set_epoch(epoch)`` before each epoch for distinct shuffles.
    """
    sampler: DistributedSampler[int] | None = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle, seed=seed, drop_last=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
    return loader, sampler
