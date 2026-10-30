from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import torch

from transformer_nmt.api.schemas import LanguagePairInfo, ModelInfo
from transformer_nmt.inference.translator import Translator
from transformer_nmt.training.checkpoint import (
    SETTINGS_NAME,
    CheckpointCandidate,
    discover_checkpoints,
    rank_checkpoints,
)
from transformer_nmt.utils.hardware import resolve_device

logger = logging.getLogger(__name__)

LANGUAGE_CODES: dict[str, str] = {
    "English": "en",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
}

LanguagePair = tuple[str, str]


def language_code(language_name: str) -> str:
    """Map a language name from the training settings to a two-letter code."""
    return LANGUAGE_CODES.get(language_name, language_name[:2].lower())


@dataclass(frozen=True)
class RegisteredModel:
    """A discovered run together with the language direction it was trained for."""

    candidate: CheckpointCandidate
    source_language: str
    target_language: str

    @property
    def run_name(self) -> str:
        """Name of the run."""
        return self.candidate.summary.run_name

    @property
    def pair(self) -> LanguagePair:
        """``(source_code, target_code)``."""
        return (self.source_language, self.target_language)


class ModelRegistry:
    """Discovers trained runs on disk and keeps one loaded translator per language pair."""

    def __init__(
        self,
        models_root: Path,
        device: torch.device | None = None,
        *,
        max_decode_len: int = 40,
        autoload: bool = True,
    ) -> None:
        """Create an empty registry.

        Args:
            models_root: Directory scanned recursively for run checkpoints.
            device: Inference device (auto-detected when omitted).
            max_decode_len: Maximum generated tokens per sentence.
            autoload: Load the best run of a pair on demand when none was selected by hand.
        """
        self.models_root = models_root
        self.device = device or resolve_device()
        self.max_decode_len = max_decode_len
        self.autoload = autoload
        self._translators: dict[LanguagePair, Translator] = {}
        self._active_run_names: dict[LanguagePair, str] = {}
        self._lock = threading.Lock()

    @property
    def gpu_name(self) -> str | None:
        """Name of the CUDA device in use, if any."""
        if self.device.type != "cuda":
            return None
        return torch.cuda.get_device_name(self.device)

    @staticmethod
    def _languages_of(candidate: CheckpointCandidate) -> LanguagePair | None:
        """Read the language direction recorded in the run's ``settings.json``."""
        settings_path = candidate.run_dir / SETTINGS_NAME
        if not settings_path.exists():
            return None
        data = json.loads(settings_path.read_text(encoding="utf-8")).get("data", {})
        return (
            language_code(data.get("source_language", "English")),
            language_code(data.get("target_language", "French")),
        )

    def models(self) -> list[RegisteredModel]:
        """Return every usable run, grouped by language pair and ranked best-first within a pair."""
        by_pair: dict[LanguagePair, list[CheckpointCandidate]] = {}
        for candidate in discover_checkpoints(self.models_root):
            pair = self._languages_of(candidate)
            if pair is not None:
                by_pair.setdefault(pair, []).append(candidate)

        registered: list[RegisteredModel] = []
        for pair in sorted(by_pair):
            candidates = by_pair[pair]
            try:
                ranked = rank_checkpoints(candidates)
            except ValueError:  # mixed selection metrics: group by metric, lowest first
                ranked = sorted(candidates, key=lambda c: (c.summary.metric, c.summary.value))
            leftovers = [candidate for candidate in candidates if candidate not in ranked]
            registered.extend(RegisteredModel(c, *pair) for c in [*ranked, *leftovers])
        return registered

    def _info(self, model: RegisteredModel) -> ModelInfo:
        """Describe a registered model for the API."""
        summary = model.candidate.summary
        return ModelInfo(
            run_name=model.run_name,
            source_language=model.source_language,
            target_language=model.target_language,
            metric=summary.metric,
            value=summary.value,
            epoch=summary.epoch,
            run_type=summary.run_type,
            active=self._active_run_names.get(model.pair) == model.run_name,
        )

    def list_models(self) -> list[ModelInfo]:
        """Describe every run for ``GET /models``."""
        return [self._info(model) for model in self.models()]

    def language_pairs(self) -> list[LanguagePairInfo]:
        """Summarise the directions that have at least one model."""
        counts: dict[LanguagePair, int] = {}
        for model in self.models():
            counts[model.pair] = counts.get(model.pair, 0) + 1
        return [
            LanguagePairInfo(
                source_language=source,
                target_language=target,
                models=count,
                active_model=self._active_run_names.get((source, target)),
            )
            for (source, target), count in sorted(counts.items())
        ]

    def active_run_name(self, source_language: str, target_language: str) -> str | None:
        """Name of the run currently serving a direction, if any."""
        return self._active_run_names.get((source_language, target_language))

    def active_models(self) -> list[ModelInfo]:
        """Describe the model currently serving each language pair."""
        return [info for info in self.list_models() if info.active]

    def select(self, run_name: str) -> ModelInfo:
        """Load ``run_name`` and make it the active model for its language pair.

        Raises:
            KeyError: If no run with that name exists under ``models_root``.
        """
        model = next((m for m in self.models() if m.run_name == run_name), None)
        if model is None:
            raise KeyError(run_name)
        with self._lock:
            self._translators[model.pair] = Translator.from_checkpoint(
                model.candidate.checkpoint_path, self.device, max_decode_len=self.max_decode_len
            )
            self._active_run_names[model.pair] = run_name
        logger.info(
            "Active model for %s->%s: %s", model.source_language, model.target_language, run_name
        )
        return self._info(model)

    def select_best(self) -> list[ModelInfo]:
        """Load the best-ranked run of every language pair."""
        best_per_pair: dict[LanguagePair, str] = {}
        for model in self.models():
            best_per_pair.setdefault(model.pair, model.run_name)
        return [self.select(run_name) for run_name in best_per_pair.values()]

    def translator_for(self, source_language: str, target_language: str) -> Translator:
        """Return the translator serving a direction, loading the best run on demand.

        Raises:
            KeyError: If no run translates that direction.
            LookupError: If a run exists but none was selected and autoload is disabled.
        """
        pair = (source_language, target_language)
        if pair in self._translators:
            return self._translators[pair]
        candidates = [m for m in self.models() if m.pair == pair]
        if not candidates:
            raise KeyError(f"No model translates {source_language} -> {target_language}")
        if not self.autoload:
            raise LookupError(
                f"No model selected for {source_language} -> {target_language}; "
                "call POST /models/select first"
            )
        self.select(candidates[0].run_name)
        return self._translators[pair]
