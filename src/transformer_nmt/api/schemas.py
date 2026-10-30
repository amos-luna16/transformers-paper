from __future__ import annotations

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """One trained run that can be selected for inference."""

    run_name: str
    run_type: str
    metric: str
    value: float
    epoch: int
    checkpoint_path: str
    is_active: bool = False


class ModelsResponse(BaseModel):
    """Runs available on disk, best first."""

    models: list[ModelInfo]
    active_run_name: str | None


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
    active_model: ModelInfo | None


class TranslationRequest(BaseModel):
    """Body of ``POST /translate``."""

    text: str = Field(min_length=1, description="One English sentence.")
    beam_size: int = Field(default=4, ge=1, le=10, description="1 = greedy, >1 = beam search.")
    length_penalty: float = Field(default=0.6, ge=0.0, le=2.0)


class TranslationResponse(BaseModel):
    """Body returned by ``POST /translate``."""

    text: str
    translation: str
    run_name: str
    device: str


class StreamRequest(BaseModel):
    """Body of ``POST /translate/stream`` (greedy decoding, one token per event)."""

    text: str = Field(min_length=1, description="One English sentence.")
