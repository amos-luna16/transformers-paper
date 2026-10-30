from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from transformer_nmt.api.registry import ModelRegistry
from transformer_nmt.api.schemas import (
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    SelectModelRequest,
    StreamRequest,
    TranslationRequest,
    TranslationResponse,
)
from transformer_nmt.config import Settings, get_settings
from transformer_nmt.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def get_registry(request: Request) -> ModelRegistry:
    """FastAPI dependency returning the registry created at startup."""
    return request.app.state.registry


def create_app(settings: Settings | None = None, *, models_root: Path | None = None) -> FastAPI:
    """Build the inference API.

    Args:
        settings: Settings to use (defaults to the environment-driven settings).
        models_root: Directory scanned for runs; overrides ``INFERENCE_MODELS_ROOT``.

    Returns:
        A configured ``FastAPI`` application.
    """
    resolved = settings or get_settings()
    root = models_root or resolved.inference.models_root or resolved.tracking.artifacts_dir

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        registry = ModelRegistry(root, max_decode_len=resolved.inference.max_decode_len)
        if resolved.inference.autoload_best_model:
            loaded = registry.select_best()
            logger.info("Startup model: %s", loaded.run_name if loaded else "none found")
        app.state.registry = registry
        yield
        registry.translator = None

    app = FastAPI(
        title="transformer-nmt inference API",
        description="English -> French translation with the Transformer trained in this repo.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health(registry: ModelRegistry = Depends(get_registry)) -> HealthResponse:  # noqa: B008
        """Report device, GPU, available runs and the active model."""
        models = registry.list_models()
        active = registry.active_model()
        return HealthResponse(
            status="ok" if active is not None else "no-model-selected",
            device=str(registry.device),
            gpu_name=registry.gpu_name,
            models_root=str(registry.models_root),
            available_models=len(models),
            active_model=active,
        )

    @app.get("/models", response_model=ModelsResponse)
    def list_models(registry: ModelRegistry = Depends(get_registry)) -> ModelsResponse:  # noqa: B008
        """List the trained runs on disk, best first."""
        return ModelsResponse(
            models=registry.list_models(), active_run_name=registry.active_run_name
        )

    @app.post("/models/select", response_model=ModelInfo)
    def select_model(
        body: SelectModelRequest,
        registry: ModelRegistry = Depends(get_registry),  # noqa: B008
    ) -> ModelInfo:
        """Load the chosen run and make it the active model for inference."""
        try:
            return registry.select(body.run_name)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=f"Unknown run {body.run_name!r}") from error

    @app.post("/translate", response_model=TranslationResponse)
    def translate(
        body: TranslationRequest,
        registry: ModelRegistry = Depends(get_registry),  # noqa: B008
    ) -> TranslationResponse:
        """Translate one sentence with the active model."""
        translator = _active_translator(registry)
        translation = translator.translate(
            body.text, beam_size=body.beam_size, length_penalty=body.length_penalty
        )
        return TranslationResponse(
            text=body.text,
            translation=translation,
            run_name=registry.active_run_name or "",
            device=str(registry.device),
        )

    @app.post("/translate/stream")
    def translate_stream(
        body: StreamRequest,
        registry: ModelRegistry = Depends(get_registry),  # noqa: B008
    ) -> StreamingResponse:
        """Stream the greedy translation token by token as server-sent events."""
        translator = _active_translator(registry)

        def events() -> Iterator[str]:
            for token in translator.stream_greedy(body.text):
                yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def _active_translator(registry: ModelRegistry):  # noqa: ANN202
    """Return the active translator or a 409 if none has been selected."""
    try:
        return registry.require_translator()
    except LookupError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


app = create_app()
