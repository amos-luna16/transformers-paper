from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from transformer_nmt.config import Settings, get_settings

settings: Settings = get_settings()
# The Modal client reads credentials from the process environment; export the ones in .env
# before any Modal object is created so `uv run modal run ...` works without extra setup.
for _key, _value in settings.modal.credential_environment().items():
    os.environ.setdefault(_key, _value)

import modal  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root")
REMOTE_ARTIFACTS_DIR = REMOTE_ROOT / "artifacts"
VOLUME_ARTIFACTS_DIR = settings.modal.volume_mount_path / "artifacts"
LOCAL_SYNC_DIR = settings.tracking.artifacts_dir / "modal"
SUMMARY_FILE_NAME = "summary.json"
DATA_FILE_NAME = Path(settings.data.archive_member).name

app = modal.App(settings.modal.app_name)
volume = modal.Volume.from_name(settings.modal.volume_name, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version=settings.modal.python_version)
    .uv_sync(str(REPO_ROOT), frozen=True)
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(REMOTE_ROOT),
            "MLFLOW_DISABLE_AGENT_HINT": "1",
            "NCCL_CUMEM_HOST_ENABLE": "0",
            "GIT_PYTHON_REFRESH": "quiet",
        }
    )
    .add_local_python_source("transformer_nmt")
    .add_local_file(
        REPO_ROOT / "scripts" / "train.py", remote_path=str(REMOTE_ROOT / "scripts/train.py")
    )
)


def _copy_if_exists(source: Path, destination: Path) -> bool:
    """Copy a file or directory tree when ``source`` exists; return whether anything was copied."""
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        shutil.copy2(source, destination)
    return True


def restore_state_from_volume() -> list[str]:
    """Seed the container-local artifacts dir with the MLflow database and cached corpus.

    The SQLite store is copied to local disk because Modal Volumes are optimised for
    write-once files, not for the many small writes of a database.
    """
    restored = []
    for relative in (Path("mlflow.db"), Path("data") / DATA_FILE_NAME):
        if _copy_if_exists(VOLUME_ARTIFACTS_DIR / relative, REMOTE_ARTIFACTS_DIR / relative):
            restored.append(str(relative))
    return restored


def persist_state_to_volume(run_name: str) -> list[str]:
    """Copy the MLflow store, run artifacts, checkpoints and plots of ``run_name`` to the Volume."""
    persisted = []
    for relative in (
        Path("mlflow.db"),
        Path("mlartifacts"),
        Path("checkpoints") / run_name,
        Path("plots") / run_name,
        Path("data") / DATA_FILE_NAME,
        Path(SUMMARY_FILE_NAME),
    ):
        if _copy_if_exists(REMOTE_ARTIFACTS_DIR / relative, VOLUME_ARTIFACTS_DIR / relative):
            persisted.append(str(relative))
    volume.commit()
    return persisted


def describe_gpus() -> list[str]:
    """Return one description line per visible CUDA device."""
    import torch

    descriptions = []
    for index in range(torch.cuda.device_count()):
        memory_gb = torch.cuda.get_device_properties(index).total_memory / 1024**3
        descriptions.append(f"{torch.cuda.get_device_name(index)} ({memory_gb:.0f} GB)")
    return descriptions


@app.function(
    image=image,
    gpu=settings.modal.gpu_spec,
    cpu=settings.modal.cpu_cores,
    memory=settings.modal.memory_mb,
    timeout=settings.modal.timeout_seconds,
    volumes={str(settings.modal.volume_mount_path): volume},
)
def train_on_gpu(
    train_args: list[str], environment: dict[str, str], gpu_count: int
) -> dict[str, Any]:
    """Run ``scripts/train.py`` under ``torchrun`` on every GPU of the container.

    Args:
        train_args: Extra CLI arguments for ``scripts/train.py`` (``--smoke``, ``--epochs`` ...).
        environment: ``PREFIX_FIELD`` settings exported from the local machine.
        gpu_count: Number of DDP ranks (one per GPU).

    Returns:
        The training summary plus the list of paths persisted to the Volume.
    """
    gpus = describe_gpus()
    print(f"[modal] GPUs: {gpus}", flush=True)
    print(f"[modal] restored from volume: {restore_state_from_volume()}", flush=True)

    summary_path = REMOTE_ARTIFACTS_DIR / SUMMARY_FILE_NAME
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={gpu_count}",
        str(REMOTE_ROOT / "scripts/train.py"),
        *train_args,
        "--launcher",
        "modal",
        "--summary-json",
        str(summary_path),
    ]
    run_environment = {
        **os.environ,
        **environment,
        "TRACKING_ARTIFACTS_DIR": str(REMOTE_ARTIFACTS_DIR),
        "TRACKING_MLFLOW_TRACKING_URI": "",
    }
    run_environment.pop("TRACKING_MLFLOW_TRACKING_URI")
    print(f"[modal] launching: {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, cwd=str(REMOTE_ROOT), env=run_environment)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    persisted = persist_state_to_volume(summary["run_name"])
    print(f"[modal] persisted to volume: {persisted}", flush=True)
    return {"summary": summary, "gpus": gpus, "persisted": persisted}


def _download_entries(remote_prefix: str, local_root: Path, *, skip: Iterable[str] = ()) -> int:
    """Download every file below ``remote_prefix`` on the Volume into ``local_root``."""
    from modal.volume import FileEntryType

    downloaded = 0
    for entry in volume.listdir(remote_prefix, recursive=True):
        if entry.type != FileEntryType.FILE or any(pattern in entry.path for pattern in skip):
            continue
        relative = Path(entry.path).relative_to(remote_prefix)
        destination = local_root / relative if entry.path != remote_prefix else local_root
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            for chunk in volume.read_file(entry.path):
                handle.write(chunk)
        downloaded += 1
    return downloaded


def download_results(run_name: str, *, all_checkpoints: bool) -> Path:
    """Pull the MLflow store and the run's files to ``artifacts/modal`` and fix artifact URIs."""
    from transformer_nmt.training.tracking import relocate_mlflow_store

    LOCAL_SYNC_DIR.mkdir(parents=True, exist_ok=True)
    remote_root = VOLUME_ARTIFACTS_DIR.relative_to(settings.modal.volume_mount_path).as_posix()
    skip = () if all_checkpoints else ("/epoch-", "/last.pt")
    count = 0
    for relative in ("mlflow.db", "mlartifacts", f"checkpoints/{run_name}", f"plots/{run_name}"):
        count += _download_entries(
            f"{remote_root}/{relative}", LOCAL_SYNC_DIR / relative, skip=skip
        )
    rewritten = relocate_mlflow_store(LOCAL_SYNC_DIR)
    print(f"[modal] downloaded {count} files to {LOCAL_SYNC_DIR} (rewrote {rewritten} MLflow rows)")
    return LOCAL_SYNC_DIR


@app.local_entrypoint()
def main(
    epochs: int | None = None,
    batch_size: int | None = None,
    precision: str | None = None,
    run_name: str | None = None,
    smoke: bool = False,
    download: bool = True,
    all_checkpoints: bool = False,
) -> None:
    """Train on Modal with the GPU configured by ``MODAL_GPU``/``MODAL_GPU_COUNT``.

    Example::

        uv run modal run scripts/modal_train.py --smoke
        uv run modal run --detach scripts/modal_train.py --epochs 25
    """
    train_args: list[str] = ["--smoke"] if smoke else []
    for flag, value in (
        ("--epochs", epochs),
        ("--batch-size", batch_size),
        ("--precision", precision),
        ("--run-name", run_name),
    ):
        if value is not None:
            train_args.extend([flag, str(value)])

    print(
        f"[modal] app={settings.modal.app_name} gpu={settings.modal.gpu_spec} "
        f"volume={settings.modal.volume_name} args={train_args}"
    )
    result = train_on_gpu.spawn(
        train_args, settings.to_environment(), settings.modal.gpu_count
    ).get()
    summary = result["summary"]
    print(json.dumps(result, indent=2))

    if download:
        local_dir = download_results(summary["run_name"], all_checkpoints=all_checkpoints)
        db_path = local_dir / "mlflow.db"
        print("\nNext steps:")
        print(f"  uv run mlflow ui --backend-store-uri sqlite:///{db_path} --port 5001")
        print('  uv run python scripts/translate.py "I love this movie."')
