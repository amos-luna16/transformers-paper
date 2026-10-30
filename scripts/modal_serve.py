from __future__ import annotations

import os
from pathlib import Path

from transformer_nmt.config import Settings, get_settings

settings: Settings = get_settings()
for _key, _value in settings.modal.credential_environment().items():
    os.environ.setdefault(_key, _value)

import modal  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
VOLUME_ARTIFACTS_DIR = settings.modal.volume_mount_path / "artifacts"

app = modal.App(f"{settings.modal.app_name}-api")
volume = modal.Volume.from_name(settings.modal.volume_name, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version=settings.modal.python_version)
    .uv_sync(str(REPO_ROOT), frozen=True)
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": "/root",
            "MLFLOW_DISABLE_AGENT_HINT": "1",
            "GIT_PYTHON_REFRESH": "quiet",
        }
    )
    .add_local_python_source("transformer_nmt")
)


@app.cls(
    image=image,
    gpu=settings.modal.inference_gpu,
    volumes={str(settings.modal.volume_mount_path): volume},
    scaledown_window=settings.modal.inference_scaledown_seconds,
    timeout=300,
)
@modal.concurrent(max_inputs=16)
class TranslationService:
    """Hosts the same FastAPI inference app as ``uv run fastapi dev``, backed by the Volume.

    Routes: ``GET /health``, ``GET /models``, ``POST /models/select``, ``POST /translate``,
    ``POST /translate/stream`` and the OpenAPI UI at ``/docs``.
    """

    @modal.enter()
    def refresh_volume(self) -> None:
        """Make sure the container sees the latest committed runs."""
        volume.reload()

    @modal.asgi_app()
    def web(self):  # noqa: ANN201 - FastAPI app
        """Create the ASGI app scanning the checkpoints persisted on the Volume."""
        from transformer_nmt.api.app import create_app

        return create_app(models_root=VOLUME_ARTIFACTS_DIR / "checkpoints")
