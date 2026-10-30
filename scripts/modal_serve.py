from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

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


class TranslationRequest(BaseModel):
    """Body of ``POST /translate``."""

    text: str | list[str] = Field(description="One English sentence or a list of sentences.")
    beam_size: int = Field(default=4, ge=1, le=10, description="1 = greedy, >1 = beam search.")
    length_penalty: float = Field(default=0.6, ge=0.0, le=2.0)


class TranslationResponse(BaseModel):
    """Body returned by ``POST /translate``."""

    translations: list[str]
    checkpoint: str
    device: str


@app.cls(
    image=image,
    gpu=settings.modal.inference_gpu,
    volumes={str(settings.modal.volume_mount_path): volume},
    scaledown_window=settings.modal.inference_scaledown_seconds,
    timeout=120,
)
@modal.concurrent(max_inputs=16)
class TranslationService:
    """Hosted translation API that serves the best checkpoint stored on the Modal Volume."""

    @modal.enter()
    def load_model(self) -> None:
        """Load the best checkpoint once per container."""
        import torch

        from transformer_nmt.inference.translator import Translator
        from transformer_nmt.training.checkpoint import find_best_checkpoint

        volume.reload()
        self.checkpoint_path = find_best_checkpoint(VOLUME_ARTIFACTS_DIR / "checkpoints")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.translator = Translator.from_checkpoint(
            self.checkpoint_path, self.device, max_decode_len=settings.inference.max_decode_len
        )
        device_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "cpu"
        print(f"[api] serving {self.checkpoint_path} on {device_name}", flush=True)

    @modal.fastapi_endpoint(method="POST", docs=True)
    def translate(self, request: TranslationRequest) -> TranslationResponse:
        """Translate one or many English sentences into French."""
        sentences = [request.text] if isinstance(request.text, str) else request.text
        if request.beam_size == 1:
            translations = self.translator.translate_batch(sentences)
        else:
            translations = [
                self.translator.translate(
                    sentence, beam_size=request.beam_size, length_penalty=request.length_penalty
                )
                for sentence in sentences
            ]
        return TranslationResponse(
            translations=translations,
            checkpoint=str(self.checkpoint_path.relative_to(settings.modal.volume_mount_path)),
            device=str(self.device),
        )

    @modal.fastapi_endpoint(method="GET")
    def health(self) -> dict[str, Any]:
        """Liveness probe reporting the checkpoint in use."""
        return {"status": "ok", "checkpoint": str(self.checkpoint_path), "device": str(self.device)}


@app.local_entrypoint()
def main(text: str = "I love this movie.", beam_size: int = 4) -> None:
    """Send one request to the (ephemeral or deployed) endpoint to verify it end to end."""
    import json
    import urllib.request

    url = TranslationService().translate.get_web_url()
    payload = json.dumps({"text": text, "beam_size": beam_size}).encode("utf-8")
    request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        print(
            f"POST {url}\n{json.dumps(json.loads(response.read()), indent=2, ensure_ascii=False)}"
        )
