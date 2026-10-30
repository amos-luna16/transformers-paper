from __future__ import annotations

import logging
import threading
from pathlib import Path

import torch

from transformer_nmt.api.schemas import ModelInfo
from transformer_nmt.inference.translator import Translator
from transformer_nmt.training.checkpoint import (
    CheckpointCandidate,
    discover_checkpoints,
    rank_checkpoints,
)
from transformer_nmt.utils.hardware import resolve_device

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Discovers trained runs on disk and holds the translator selected for inference."""

    def __init__(
        self, models_root: Path, device: torch.device | None = None, *, max_decode_len: int = 40
    ) -> None:
        """Create an empty registry.

        Args:
            models_root: Directory scanned recursively for run checkpoints.
            device: Inference device (auto-detected when omitted).
            max_decode_len: Maximum generated tokens per sentence.
        """
        self.models_root = models_root
        self.device = device or resolve_device()
        self.max_decode_len = max_decode_len
        self.active_run_name: str | None = None
        self.translator: Translator | None = None
        self._lock = threading.Lock()

    @property
    def gpu_name(self) -> str | None:
        """Name of the CUDA device in use, if any."""
        if self.device.type != "cuda":
            return None
        return torch.cuda.get_device_name(self.device)

    def candidates(self) -> list[CheckpointCandidate]:
        """Return the runs on disk ranked best-first (smoke runs last, see ``rank_checkpoints``)."""
        found = discover_checkpoints(self.models_root)
        try:
            ranked = rank_checkpoints(found)
        except ValueError:
            # Mixed selection metrics: fall back to grouping by metric, loss-based runs first.
            ranked = sorted(found, key=lambda c: (c.summary.metric, c.summary.value))
        remaining = [candidate for candidate in found if candidate not in ranked]
        return [*ranked, *remaining]

    def list_models(self) -> list[ModelInfo]:
        """Describe every run for the API, flagging the active one."""
        return [
            ModelInfo(
                run_name=candidate.summary.run_name,
                run_type=candidate.summary.run_type,
                metric=candidate.summary.metric,
                value=candidate.summary.value,
                epoch=candidate.summary.epoch,
                checkpoint_path=str(candidate.checkpoint_path),
                is_active=candidate.summary.run_name == self.active_run_name,
            )
            for candidate in self.candidates()
        ]

    def active_model(self) -> ModelInfo | None:
        """Describe the active run, or ``None`` when nothing is loaded."""
        return next((model for model in self.list_models() if model.is_active), None)

    def select(self, run_name: str) -> ModelInfo:
        """Load ``run_name`` and make it the active model.

        Args:
            run_name: Name of a run listed by :meth:`list_models`.

        Returns:
            The description of the newly active model.

        Raises:
            KeyError: If no run with that name exists under ``models_root``.
        """
        candidate = next((c for c in self.candidates() if c.summary.run_name == run_name), None)
        if candidate is None:
            raise KeyError(run_name)
        with self._lock:
            self.translator = Translator.from_checkpoint(
                candidate.checkpoint_path, self.device, max_decode_len=self.max_decode_len
            )
            self.active_run_name = run_name
        logger.info("Active model: %s (%s)", run_name, candidate.checkpoint_path)
        info = self.active_model()
        assert info is not None  # noqa: S101 - just selected
        return info

    def select_best(self) -> ModelInfo | None:
        """Load the best-ranked run if any exists."""
        ranked = self.candidates()
        return self.select(ranked[0].summary.run_name) if ranked else None

    def require_translator(self) -> Translator:
        """Return the active translator or raise ``LookupError`` when none is selected."""
        if self.translator is None:
            raise LookupError("No model selected; call POST /models/select first")
        return self.translator
