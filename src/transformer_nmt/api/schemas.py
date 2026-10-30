from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

LanguageCode = str


def _normalise_code(value: str) -> str:
    """Lower-case a language code and reject anything that is not two letters."""
    code = value.strip().lower()
    if len(code) != 2 or not code.isalpha():
        raise ValueError("language codes are two letters, for example 'en', 'es' or 'fr'")
    return code


class ModelInfo(BaseModel):
    """One trained run that can serve a language pair."""

    run_name: str
    source_language: LanguageCode
    target_language: LanguageCode
    metric: str
    value: float
    epoch: int
    run_type: str
    active: bool


class ModelsResponse(BaseModel):
    """Runs available on disk, best first within each language pair."""

    models: list[ModelInfo]


class LanguagePairInfo(BaseModel):
    """A translation direction and the run currently serving it."""

    source_language: LanguageCode
    target_language: LanguageCode
    models: int
    active_model: str | None


class LanguagesResponse(BaseModel):
    """Body of ``GET /languages``."""

    pairs: list[LanguagePairInfo]


class SelectModelRequest(BaseModel):
    """Body of ``POST /models/select``."""

    run_name: str = Field(description="``run_name`` from ``GET /models``.")


class HealthResponse(BaseModel):
    """Body of ``GET /health``."""

    status: str
    device: str
    gpu_name: str | None
    models_root: str
    available_models: int
    active_models: list[ModelInfo]


class TranslationRequest(BaseModel):
    """Body of ``POST /translate``."""

    text: str = Field(min_length=1, description="One sentence in the source language.")
    source_language: LanguageCode = Field(description="Two-letter code, e.g. 'en'.")
    target_language: LanguageCode = Field(description="Two-letter code, e.g. 'es'.")
    beam_size: int = Field(default=4, ge=1, le=10, description="1 = greedy, >1 = beam search.")

    _normalise = field_validator("source_language", "target_language")(_normalise_code)


class TranslationResponse(BaseModel):
    """Body returned by ``POST /translate``."""

    translation: str
    model: str = Field(description="``run_name`` of the model that produced the translation.")


class StreamRequest(BaseModel):
    """Body of ``POST /translate/stream`` (greedy decoding, one token per event)."""

    text: str = Field(min_length=1, description="One sentence in the source language.")
    source_language: LanguageCode
    target_language: LanguageCode

    _normalise = field_validator("source_language", "target_language")(_normalise_code)
