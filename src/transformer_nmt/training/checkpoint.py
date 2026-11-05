from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from transformer_nmt.config import Settings
from transformer_nmt.data.vocabulary import Vocabulary

logger = logging.getLogger(__name__)

LAST_CHECKPOINT_NAME = "last.pt"
BEST_CHECKPOINT_NAME = "best.pt"
BEST_SUMMARY_NAME = "best.json"
SOURCE_VOCAB_NAME = "source_vocab.json"
TARGET_VOCAB_NAME = "target_vocab.json"
SETTINGS_NAME = "settings.json"
SMOKE_RUN_TYPE = "smoke"


def unwrap_model(
    model: nn.Module,
) -> nn.Module:
    """Return the plain module behind DDP and ``torch.compile`` wrappers."""
    if isinstance(model, DistributedDataParallel):
        model = model.module
    return getattr(model, "_orig_mod", model)


@dataclass(frozen=True)
class BestCheckpointSummary:
    """Small JSON sidecar describing the best checkpoint of a run."""

    run_name: str
    metric: str
    value: float
    epoch: int
    checkpoint_file: str = BEST_CHECKPOINT_NAME
    run_type: str = "full"

    def save(
        self,
        path: Path,
    ) -> None:
        """Write the summary as JSON."""
        path.write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        path: Path,
    ) -> BestCheckpointSummary:
        """Read a summary written by :meth:`save`."""
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    @property
    def minimize(self) -> bool:
        """Whether lower values of ``metric`` are better."""
        return self.metric.endswith("loss")


def save_run_metadata(
    run_dir: Path,
    settings: Settings,
    source_vocab: Vocabulary,
    target_vocab: Vocabulary,
) -> None:
    """Persist vocabularies and settings next to the checkpoints so inference is self-contained.

    Args:
        run_dir: Directory of the run's checkpoints.
        settings: Settings used for training.
        source_vocab: Source vocabulary.
        target_vocab: Target vocabulary.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    source_vocab.save(run_dir / SOURCE_VOCAB_NAME)
    target_vocab.save(run_dir / TARGET_VOCAB_NAME)
    (run_dir / SETTINGS_NAME).write_text(
        json.dumps(settings.model_dump(mode="json", exclude={"modal"}), indent=2),
        encoding="utf-8",
    )


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    global_step: int,
    best_metric_value: float | None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a resumable checkpoint containing model, optimiser and schedule state.

    Args:
        path: Destination ``.pt`` file.
        model: Model (DDP/compiled wrappers are unwrapped).
        optimizer: Optimiser whose state is saved.
        scheduler: LR scheduler whose state is saved.
        scaler: Optional fp16 gradient scaler.
        epoch: Last completed epoch (1-based).
        global_step: Optimisation steps performed so far.
        best_metric_value: Best validation metric observed so far.
        extra: Additional JSON-compatible metadata.

    Returns:
        The written path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_metric_value": best_metric_value,
        "torch_version": str(torch.__version__),
        **(extra or {}),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def load_checkpoint(
    path: Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint written by :func:`save_checkpoint` with ``weights_only=True``."""
    return torch.load(path, map_location=map_location, weights_only=True)


@dataclass(frozen=True)
class CheckpointCandidate:
    """A run directory with a usable best checkpoint."""

    summary: BestCheckpointSummary
    run_dir: Path

    @property
    def checkpoint_path(self) -> Path:
        """Path of the run's best checkpoint."""
        return self.run_dir / self.summary.checkpoint_file


def discover_checkpoints(
    search_root: Path,
) -> list[CheckpointCandidate]:
    """Collect every run whose ``best.json`` sits next to its checkpoint and ``settings.json``.

    The copies MLflow keeps under ``mlartifacts`` have no ``settings.json`` beside them and are
    therefore skipped.

    Args:
        search_root: Directory searched recursively (for example ``artifacts/``).

    Returns:
        Candidates in path order.
    """
    candidates: list[CheckpointCandidate] = []
    for sidecar in sorted(search_root.rglob(BEST_SUMMARY_NAME)):
        summary = BestCheckpointSummary.load(sidecar)
        run_dir = sidecar.parent
        if (run_dir / summary.checkpoint_file).exists() and (run_dir / SETTINGS_NAME).exists():
            candidates.append(CheckpointCandidate(summary, run_dir))
    return candidates


def rank_checkpoints(
    candidates: Sequence[CheckpointCandidate],
    *,
    exclude: Sequence[str] = (SMOKE_RUN_TYPE,),
    metric: str | None = None,
) -> list[CheckpointCandidate]:
    """Order candidates best-first on a single selection metric.

    Smoke runs, identified by the sidecar's ``run_type`` or by a run name containing one of
    ``exclude``, are dropped unless nothing else exists because their validation sets differ.
    Only runs selected with the same metric are comparable: ``metric`` picks it, otherwise the
    single metric present is used.

    Args:
        candidates: Discovered checkpoints.
        exclude: Substrings identifying smoke runs by name (fallback for old sidecars).
        metric: Selection metric to compare, for example ``"val/loss"``.

    Returns:
        Candidates sorted best-first.

    Raises:
        ValueError: If runs with different selection metrics exist and ``metric`` is not given.
    """
    full_runs = [
        candidate
        for candidate in candidates
        if candidate.summary.run_type != SMOKE_RUN_TYPE
        and not any(marker in candidate.summary.run_name for marker in exclude)
    ]
    pool = list(full_runs or candidates)
    by_metric: dict[str, list[CheckpointCandidate]] = {}
    for candidate in pool:
        by_metric.setdefault(candidate.summary.metric, []).append(candidate)
    if not by_metric:
        return []
    if metric in by_metric:
        pool = by_metric[metric]
    elif len(by_metric) == 1:
        pool = next(iter(by_metric.values()))
    else:
        raise ValueError(
            f"Runs were selected with different metrics {sorted(by_metric)}; pass metric=..."
        )

    def sort_key(
        candidate: CheckpointCandidate,
    ) -> float:
        summary = candidate.summary
        return summary.value if summary.minimize else -summary.value

    return sorted(pool, key=sort_key)
