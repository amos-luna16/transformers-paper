from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from transformer_nmt.config import Settings, get_settings

settings: Settings = get_settings()
# Export the Modal token pair from .env: the client reads MODAL_TOKEN_* from the environment at
# call time. The profile is resolved at `import modal`, so select it with
# `uv run --env-file .env modal run ...` when several profiles exist.
for _key, _value in settings.modal.credential_environment().items():
    os.environ.setdefault(_key, _value)

import modal  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root")
REMOTE_ARTIFACTS_DIR = REMOTE_ROOT / "artifacts"
VOLUME_ARTIFACTS_DIR = settings.modal.volume_mount_path / "artifacts"
LOCAL_SYNC_DIR = settings.tracking.artifacts_dir / "modal"
SUMMARY_FILE_NAME = "summary.json"
DEFAULT_DATA_FILE_NAME = Path(settings.data.archive_member).name
LOCK_KEY = "mlflow.db"
LOCK_STALE_SECONDS = float(settings.modal.timeout_seconds + 600)

app = modal.App(settings.modal.app_name)
volume = modal.Volume.from_name(settings.modal.volume_name, create_if_missing=True)
locks = modal.Dict.from_name(f"{settings.modal.volume_name}-locks", create_if_missing=True)
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


def _copy_if_exists(
    source: Path,
    destination: Path,
) -> bool:
    """Copy a file or directory tree when ``source`` exists; return whether anything was copied."""
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        shutil.copy2(source, destination)
    return True


def acquire_volume_lock(
    owner: str,
) -> None:
    """Block until this call owns the Volume write lock; stale holders are evicted.

    The MLflow SQLite database is copied in and out of the Volume, so two concurrent runs
    would silently overwrite each other's rows without this account-wide lock.
    """
    while not locks.put(LOCK_KEY, {"owner": owner, "since": time.time()}, skip_if_exists=True):
        holder = locks.get(LOCK_KEY)
        if holder is None:
            continue
        if time.time() - float(holder["since"]) > LOCK_STALE_SECONDS:
            print(f"[modal] evicting stale lock held by {holder['owner']}", flush=True)
            locks.pop(LOCK_KEY, None)
            continue
        print(f"[modal] waiting for Volume lock held by {holder['owner']}", flush=True)
        time.sleep(30)


def release_volume_lock(
    owner: str,
) -> None:
    """Release the Volume write lock if this call still owns it."""
    holder = locks.get(LOCK_KEY)
    if holder is not None and holder.get("owner") == owner:
        locks.pop(LOCK_KEY, None)


def restore_state_from_volume(
    volume_artifacts_dir: Path,
    data_file_name: str,
) -> list[str]:
    """Seed the container-local artifacts dir with the MLflow database and cached corpus.

    The SQLite store is copied to local disk because Modal Volumes are optimised for
    write-once files, not for the many small writes of a database.
    """
    restored = []
    for relative in (Path("mlflow.db"), Path("data") / data_file_name):
        if _copy_if_exists(volume_artifacts_dir / relative, REMOTE_ARTIFACTS_DIR / relative):
            restored.append(str(relative))
    return restored


def persist_state_to_volume(
    volume_artifacts_dir: Path,
    run_name: str | None,
    data_file_name: str,
) -> list[str]:
    """Copy the MLflow store, run artifacts, checkpoints and plots to the Volume.

    Without ``run_name`` (a failed run, before ``summary.json`` exists) the whole checkpoints
    and plots trees are copied; they only contain the current run because
    :func:`restore_state_from_volume` never seeds them.
    """
    run_paths = (
        (Path("checkpoints") / run_name, Path("plots") / run_name)
        if run_name
        else (Path("checkpoints"), Path("plots"))
    )
    persisted = []
    for relative in (
        Path("mlflow.db"),
        Path("mlartifacts"),
        *run_paths,
        Path("data") / data_file_name,
        Path(SUMMARY_FILE_NAME),
    ):
        if _copy_if_exists(REMOTE_ARTIFACTS_DIR / relative, volume_artifacts_dir / relative):
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
    train_args: list[str],
    environment: dict[str, str],
    gpu_count: int,
    volume_artifacts_dir: str,
) -> dict[str, Any]:
    """Run ``scripts/train.py`` under ``torchrun`` on every GPU of the container.

    Args:
        train_args: Extra CLI arguments for ``scripts/train.py`` (``--smoke``, ``--epochs`` ...).
        environment: ``PREFIX_FIELD`` settings exported from the local machine.
        gpu_count: Number of DDP ranks (one per GPU).
        volume_artifacts_dir: Container path of the persisted artifacts on the Volume.

    Returns:
        The training summary plus the list of paths persisted to the Volume.
    """
    volume_dir = Path(volume_artifacts_dir)
    data_file_name = Path(environment.get("DATA_ARCHIVE_MEMBER", DEFAULT_DATA_FILE_NAME)).name
    owner = os.environ.get("MODAL_TASK_ID", f"pid-{os.getpid()}")
    gpus = describe_gpus()
    print(f"[modal] GPUs: {gpus}", flush=True)

    acquire_volume_lock(owner)
    summary: dict[str, Any] | None = None
    try:
        volume.reload()
        restored = restore_state_from_volume(volume_dir, data_file_name)
        print(f"[modal] restored from volume: {restored}", flush=True)
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
        }
        run_environment.pop("TRACKING_MLFLOW_TRACKING_URI", None)
        print(f"[modal] launching: {' '.join(command)}", flush=True)
        try:
            subprocess.run(command, check=True, cwd=str(REMOTE_ROOT), env=run_environment)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        finally:
            persisted = persist_state_to_volume(
                volume_dir, summary["run_name"] if summary else None, data_file_name
            )
            print(f"[modal] persisted to volume: {persisted}", flush=True)
    finally:
        release_volume_lock(owner)
    return {"summary": summary, "gpus": gpus, "persisted": persisted}


def _download_entries(
    remote_prefix: str,
    local_root: Path,
    *,
    skip: Iterable[str] = (),
) -> int:
    """Download every file below ``remote_prefix`` on the Volume into ``local_root``."""
    from modal.volume import FileEntryType

    downloaded = 0
    for entry in volume.listdir(remote_prefix, recursive=True):
        if entry.type != FileEntryType.FILE or any(pattern in entry.path for pattern in skip):
            continue
        destination = (
            local_root
            if entry.path == remote_prefix
            else local_root / Path(entry.path).relative_to(remote_prefix)
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            for chunk in volume.read_file(entry.path):
                handle.write(chunk)
        downloaded += 1
    return downloaded


def _remote_artifacts_prefix() -> str:
    """Volume-relative path of the persisted artifacts directory."""
    return VOLUME_ARTIFACTS_DIR.relative_to(settings.modal.volume_mount_path).as_posix()


def latest_run_name_on_volume() -> str:
    """Return the most recently modified run directory under ``checkpoints/`` on the Volume."""
    from modal.volume import FileEntryType

    runs = [
        entry
        for entry in volume.listdir(f"{_remote_artifacts_prefix()}/checkpoints")
        if entry.type == FileEntryType.DIRECTORY
    ]
    if not runs:
        raise SystemExit("[modal] no finished run found on the Volume; pass --run-name")
    return Path(max(runs, key=lambda entry: entry.mtime).path).name


def download_results(
    run_name: str,
    *,
    all_checkpoints: bool,
) -> Path:
    """Pull the MLflow store and the run's files to ``artifacts/modal`` and fix artifact URIs."""
    from transformer_nmt.training.tracking import relocate_mlflow_store

    LOCAL_SYNC_DIR.mkdir(parents=True, exist_ok=True)
    remote_root = _remote_artifacts_prefix()
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
    train: bool = True,
    download: bool = True,
    all_checkpoints: bool = False,
) -> None:
    """Train on Modal with the GPU configured by ``MODAL_GPU``/``MODAL_GPU_COUNT``.

    Example::

        uv run --env-file .env modal run scripts/modal_train.py --smoke
        uv run --env-file .env modal run --detach scripts/modal_train.py --epochs 25
        uv run --env-file .env modal run scripts/modal_train.py --no-train   # fetch last run
    """
    if train:
        train_args: list[str] = ["--smoke"] if smoke else []
        for flag, value in (
            ("--epochs", epochs),
            ("--batch-size", batch_size),
            ("--precision", precision),
            ("--run-name", run_name),
        ):
            if value is not None:
                train_args.extend([flag, str(value)])
        environment = settings.to_environment()
        if environment.pop("DATA_LOCAL_FILE", None):
            print("[modal] DATA_LOCAL_FILE is local-only; the container downloads the corpus")
        print(
            f"[modal] app={settings.modal.app_name} gpu={settings.modal.gpu_spec} "
            f"volume={settings.modal.volume_name} args={train_args}"
        )
        result = train_on_gpu.spawn(
            train_args, environment, settings.modal.gpu_count, str(VOLUME_ARTIFACTS_DIR)
        ).get()
        print(json.dumps(result, indent=2))
        if result["summary"] is None:
            raise SystemExit("[modal] training failed; partial artifacts were persisted")
        run_name = result["summary"]["run_name"]

    if download:
        run_name = run_name or latest_run_name_on_volume()
        local_dir = download_results(run_name, all_checkpoints=all_checkpoints)
        db_path = local_dir / "mlflow.db"
        print("\nNext steps:")
        print(f"  uv run mlflow ui --backend-store-uri sqlite:///{db_path} --port 5001")
        print("  uv run fastapi dev   # then POST /translate, see README")
