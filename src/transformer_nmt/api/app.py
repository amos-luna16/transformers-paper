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
    LanguagesResponse,
    ModelInfo,
    ModelsResponse,
    SelectModelRequest,
    StreamRequest,
    TranslationRequest,
    TranslationResponse,
)
from transformer_nmt.config import Settings, get_settings
from transformer_nmt.inference.translator import Translator
from transformer_nmt.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def get_registry(
    request: Request,
) -> ModelRegistry:
    """FastAPI dependency returning the registry created at startup."""
    return request.app.state.registry


def _translator_or_error(
    registry: ModelRegistry,
    source: str,
    target: str,
) -> Translator:
    """Resolve the translator for a direction, mapping registry errors to HTTP errors."""
    try:
        return registry.translator_for(source, target)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error).strip("'")) from error
    except LookupError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


def create_app(
    settings: Settings | None = None,
    *,
    models_root: Path | None = None,
) -> FastAPI:
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
    async def lifespan(
        app: FastAPI,
    ) -> AsyncIterator[None]:
        configure_logging()
        registry = ModelRegistry(
            root,
            max_decode_len=resolved.inference.max_decode_len,
            autoload=resolved.inference.autoload_best_model,
        )
        if resolved.inference.autoload_best_model:
            loaded = registry.select_best()
            logger.info("Startup models: %s", [m.run_name for m in loaded] or "none found")
        app.state.registry = registry
        yield

    app = FastAPI(
        title="transformer-nmt inference API",
        description=(
            "Translation with the Transformers trained in this repo, one model per language pair."
        ),
        version="0.2.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health(
        registry: ModelRegistry = Depends(get_registry),  # noqa: B008
    ) -> HealthResponse:  # noqa: B008
        """Report device, GPU, how many runs exist and which model serves each language pair."""
        active = registry.active_models()
        return HealthResponse(
            status="ok" if active else "no-model-selected",
            device=str(registry.device),
            gpu_name=registry.gpu_name,
            models_root=str(registry.models_root),
            available_models=len(registry.list_models()),
            active_models=active,
        )

    @app.get("/languages", response_model=LanguagesResponse)
    def languages(
        registry: ModelRegistry = Depends(get_registry),  # noqa: B008
    ) -> LanguagesResponse:  # noqa: B008
        """List the translation directions that have at least one trained model."""
        return LanguagesResponse(pairs=registry.language_pairs())

    @app.get("/models", response_model=ModelsResponse)
    def list_models(
        registry: ModelRegistry = Depends(get_registry),  # noqa: B008
    ) -> ModelsResponse:  # noqa: B008
        """List every run with its language direction, best first within each direction."""
        return ModelsResponse(models=registry.list_models())

    @app.post("/models/select", response_model=ModelInfo)
    def select_model(
        body: SelectModelRequest,
        registry: ModelRegistry = Depends(get_registry),  # noqa: B008
    ) -> ModelInfo:
        """Load the chosen run and make it the active model for its language direction."""
        try:
            return registry.select(body.run_name)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=f"Unknown run {body.run_name!r}") from error

    @app.post("/translate", response_model=TranslationResponse)
    def translate(
        body: TranslationRequest,
        registry: ModelRegistry = Depends(get_registry),  # noqa: B008
    ) -> TranslationResponse:
        """Translate one sentence with the model serving the requested direction."""
        translator = _translator_or_error(registry, body.source_language, body.target_language)
        translation = translator.translate(
            body.text, beam_size=body.beam_size, length_penalty=resolved.inference.length_penalty
        )
        run_name = registry.active_run_name(body.source_language, body.target_language) or ""
        return TranslationResponse(translation=translation, model=run_name)

    @app.post("/translate/stream")
    def translate_stream(
        body: StreamRequest,
        registry: ModelRegistry = Depends(get_registry),  # noqa: B008
    ) -> StreamingResponse:
        """Stream the greedy translation token by token as server-sent events."""
        translator = _translator_or_error(registry, body.source_language, body.target_language)

        def events() -> Iterator[str]:
            for token in translator.stream_greedy(body.text):
                yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


app = create_app()
