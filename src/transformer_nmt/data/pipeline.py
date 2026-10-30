from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from transformer_nmt.config import Settings
from transformer_nmt.data.dataset import (
    SentencePair,
    TranslationDataset,
    deduplicate_pairs,
    load_sentence_pairs,
    normalize_and_filter_pairs,
    split_pairs,
)
from transformer_nmt.data.download import ensure_dataset_file
from transformer_nmt.data.preprocessing import WordTokenizer
from transformer_nmt.data.vocabulary import Vocabulary

logger = logging.getLogger(__name__)


@dataclass
class DataBundle:
    """Everything the trainer needs about the data, built once per process."""

    tokenizer: WordTokenizer
    source_vocab: Vocabulary
    target_vocab: Vocabulary
    train_dataset: TranslationDataset
    validation_dataset: TranslationDataset
    train_pairs: list[SentencePair]
    validation_pairs: list[SentencePair]
    statistics: dict[str, Any] = field(default_factory=dict)


def _length_statistics(pairs: list[SentencePair], tokenizer: WordTokenizer) -> dict[str, float]:
    """Compute mean and maximum token lengths of both sides of ``pairs``."""
    source_lengths = [len(tokenizer(src)) for src, _ in pairs]
    target_lengths = [len(tokenizer(tgt)) for _, tgt in pairs]
    return {
        "source_mean_tokens": round(statistics.fmean(source_lengths), 2),
        "source_max_tokens": max(source_lengths),
        "target_mean_tokens": round(statistics.fmean(target_lengths), 2),
        "target_max_tokens": max(target_lengths),
    }


def prepare_data(settings: Settings, data_dir: Path | None = None) -> DataBundle:
    """Download, normalise, split and encode the translation corpus.

    Args:
        settings: Global settings (data section drives every choice here).
        data_dir: Cache directory; defaults to ``settings.tracking.data_dir``.

    Returns:
        A :class:`DataBundle` with datasets, vocabularies and corpus statistics.
    """
    data_settings = settings.data
    pairs_file = ensure_dataset_file(data_settings, data_dir or settings.tracking.data_dir)
    tokenizer = WordTokenizer()

    raw_pairs = load_sentence_pairs(pairs_file, data_settings.max_pairs)
    filtered_pairs = normalize_and_filter_pairs(raw_pairs, tokenizer, data_settings.max_seq_len)
    pairs = deduplicate_pairs(filtered_pairs)
    train_pairs, validation_pairs = split_pairs(
        pairs, data_settings.validation_fraction, data_settings.split_seed
    )

    source_vocab = Vocabulary.build(
        (src for src, _ in train_pairs), tokenizer, data_settings.min_token_freq
    )
    target_vocab = Vocabulary.build(
        (tgt for _, tgt in train_pairs), tokenizer, data_settings.min_token_freq
    )

    train_dataset = TranslationDataset(
        train_pairs, tokenizer, source_vocab, target_vocab, data_settings.max_seq_len
    )
    validation_dataset = TranslationDataset(
        validation_pairs, tokenizer, source_vocab, target_vocab, data_settings.max_seq_len
    )

    stats: dict[str, Any] = {
        "pairs_file": str(pairs_file),
        "raw_pairs": len(raw_pairs),
        "filtered_pairs": len(filtered_pairs),
        "duplicate_pairs_dropped": len(filtered_pairs) - len(pairs),
        "unique_pairs": len(pairs),
        "train_pairs": len(train_pairs),
        "validation_pairs": len(validation_pairs),
        "source_vocab_size": len(source_vocab),
        "target_vocab_size": len(target_vocab),
        **_length_statistics(pairs, tokenizer),
    }
    logger.info("Dataset ready: %s", stats)
    return DataBundle(
        tokenizer=tokenizer,
        source_vocab=source_vocab,
        target_vocab=target_vocab,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        train_pairs=train_pairs,
        validation_pairs=validation_pairs,
        statistics=stats,
    )
