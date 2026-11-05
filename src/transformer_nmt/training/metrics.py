from __future__ import annotations

import csv
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from sacrebleu.metrics import BLEU
from torch import Tensor

MAX_LOSS_FOR_PERPLEXITY = 20.0


def count_correct_tokens(
    logits: Tensor,
    targets: Tensor,
    pad_index: int,
) -> tuple[int, int]:
    """Count greedy next-token hits over non-padding positions.

    Args:
        logits: ``[batch, length, vocab]`` or ``[tokens, vocab]`` predictions.
        targets: Matching gold indices.
        pad_index: Padding index excluded from the count.

    Returns:
        ``(correct, total)`` token counts.
    """
    predictions = logits.argmax(dim=-1)
    mask = targets != pad_index
    correct = int(((predictions == targets) & mask).sum().item())
    return correct, int(mask.sum().item())


def perplexity_from_loss(
    mean_token_loss: float,
) -> float:
    """Convert a mean per-token cross-entropy into perplexity, clamped to avoid overflow."""
    return math.exp(min(mean_token_loss, MAX_LOSS_FOR_PERPLEXITY))


def corpus_bleu(
    hypotheses: Sequence[str],
    references: Sequence[str],
) -> float:
    """Compute corpus-level BLEU with sacreBLEU on already tokenised, lower-cased text.

    Args:
        hypotheses: System outputs, one sentence per entry.
        references: Gold translations aligned with ``hypotheses``.

    Returns:
        BLEU score in ``[0, 100]``.
    """
    if not hypotheses:
        return 0.0
    return float(
        BLEU(tokenize="13a", lowercase=True)
        .corpus_score(list(hypotheses), [list(references)])
        .score
    )


@dataclass
class TokenLossAccumulator:
    """Running sums of token-level loss and accuracy that can be reduced across ranks."""

    loss_sum: float = 0.0
    token_count: int = 0
    correct_tokens: int = 0

    def update(
        self,
        loss_sum: float,
        token_count: int,
        correct_tokens: int,
    ) -> None:
        """Add the statistics of one batch."""
        self.loss_sum += loss_sum
        self.token_count += token_count
        self.correct_tokens += correct_tokens

    def as_tensor(
        self,
        device: torch.device,
    ) -> Tensor:
        """Pack the sums into a float64 tensor for ``all_reduce``."""
        return torch.tensor(
            [self.loss_sum, self.token_count, self.correct_tokens],
            dtype=torch.float64,
            device=device,
        )

    @classmethod
    def from_tensor(
        cls,
        values: Tensor,
    ) -> TokenLossAccumulator:
        """Rebuild an accumulator from a tensor produced by :meth:`as_tensor`."""
        loss_sum, token_count, correct = values.tolist()
        return cls(loss_sum=loss_sum, token_count=int(token_count), correct_tokens=int(correct))

    @property
    def mean_loss(self) -> float:
        """Mean per-token loss."""
        return self.loss_sum / max(self.token_count, 1)

    @property
    def accuracy(self) -> float:
        """Fraction of correctly predicted non-padding tokens."""
        return self.correct_tokens / max(self.token_count, 1)

    @property
    def perplexity(self) -> float:
        """Perplexity derived from the mean loss."""
        return perplexity_from_loss(self.mean_loss)


@dataclass
class TrainingHistory:
    """Per-epoch metric rows, exportable to JSON/CSV and consumed by the plotting module."""

    rows: list[dict[str, Any]] = field(default_factory=list)

    def append(
        self,
        row: dict[str, Any],
    ) -> None:
        """Record the metrics of one epoch."""
        self.rows.append(dict(row))

    def series(
        self,
        key: str,
    ) -> list[float]:
        """Return the values of ``key`` across epochs, skipping epochs where it is missing."""
        return [float(row[key]) for row in self.rows if key in row and row[key] is not None]

    def best(
        self,
        key: str,
        *,
        minimize: bool,
    ) -> tuple[int, float] | None:
        """Return ``(epoch, value)`` of the best value for ``key``, or ``None`` if absent."""
        candidates = [
            (int(row["epoch"]), float(row[key])) for row in self.rows if row.get(key) is not None
        ]
        if not candidates:
            return None
        return (
            min(candidates, key=lambda item: item[1])
            if minimize
            else max(candidates, key=lambda item: item[1])
        )

    def save_json(
        self,
        path: Path,
    ) -> None:
        """Write the history as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.rows, indent=2), encoding="utf-8")

    def save_csv(
        self,
        path: Path,
    ) -> None:
        """Write the history as CSV with the union of all columns."""
        path.parent.mkdir(parents=True, exist_ok=True)
        columns: list[str] = []
        for row in self.rows:
            columns.extend(key for key in row if key not in columns)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.rows)
