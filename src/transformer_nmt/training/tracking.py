from __future__ import annotations

import contextlib
import logging
import sqlite3
from collections.abc import Iterator, Mapping
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

import mlflow
from matplotlib.figure import Figure
from mlflow.tracking import MlflowClient

from transformer_nmt.config import TrackingSettings

logger = logging.getLogger(__name__)

MAX_PARAM_VALUE_LENGTH = 500


def _stringify_params(params: Mapping[str, Any]) -> dict[str, str]:
    """Render parameter values as strings within MLflow's length limit."""
    return {key: str(value)[:MAX_PARAM_VALUE_LENGTH] for key, value in params.items()}


def ensure_experiment(settings: TrackingSettings) -> str:
    """Create the MLflow experiment if needed, pinning its artifact location, and return its id.

    Args:
        settings: Tracking settings (tracking URI, experiment name, artifact directory).

    Returns:
        The experiment id.
    """
    client = MlflowClient(tracking_uri=settings.resolved_tracking_uri)
    experiment = client.get_experiment_by_name(settings.experiment_name)
    if experiment is not None:
        return experiment.experiment_id
    artifact_location = (
        (settings.mlflow_artifacts_dir / settings.experiment_name).resolve().as_uri()
    )
    return client.create_experiment(settings.experiment_name, artifact_location=artifact_location)


class ExperimentTracker:
    """Thin MLflow wrapper that becomes a no-op on non-main ranks or when disabled."""

    def __init__(self, settings: TrackingSettings, *, enabled: bool = True) -> None:
        """Configure the tracking store and experiment.

        Args:
            settings: Tracking settings (URI, experiment name, system-metrics options).
            enabled: When False every method returns immediately (used on DDP ranks > 0).
        """
        self.settings = settings
        self.enabled = enabled
        self.run_id: str | None = None
        if not enabled:
            return
        settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(settings.resolved_tracking_uri)
        ensure_experiment(settings)
        mlflow.set_experiment(settings.experiment_name)
        if settings.log_system_metrics:
            try:
                mlflow.set_system_metrics_sampling_interval(
                    settings.system_metrics_interval_seconds
                )
                mlflow.enable_system_metrics_logging()
            except Exception as error:  # pragma: no cover - optional dependency path
                logger.warning("System metrics logging unavailable: %s", error)

    @contextlib.contextmanager
    def run(
        self, run_name: str, *, tags: Mapping[str, str] | None = None
    ) -> Iterator[ExperimentTracker]:
        """Open an MLflow run for the duration of the ``with`` block.

        The run is marked ``FAILED`` if the block raises and ``FINISHED`` otherwise.

        Args:
            run_name: Human-readable run name.
            tags: Tags attached at creation time.
        """
        if not self.enabled:
            yield self
            return
        with mlflow.start_run(run_name=run_name, tags=dict(tags or {})) as active_run:
            self.run_id = active_run.info.run_id
            logger.info(
                "MLflow run '%s' (%s) | experiment '%s' | store %s",
                run_name,
                self.run_id,
                self.settings.experiment_name,
                self.settings.resolved_tracking_uri,
            )
            yield self

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Log parameters (values are stringified and truncated)."""
        if self.enabled and params:
            mlflow.log_params(_stringify_params(params))

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        """Attach or update run tags."""
        if self.enabled and tags:
            mlflow.set_tags({key: str(value) for key, value in tags.items()})

    def log_metrics(self, metrics: Mapping[str, float | int | None], *, step: int) -> None:
        """Log numeric metrics at ``step``; ``None`` values are skipped."""
        if not self.enabled:
            return
        clean = {key: float(value) for key, value in metrics.items() if value is not None}
        if clean:
            mlflow.log_metrics(clean, step=step)

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        """Upload a file to the run's artifact store."""
        if self.enabled:
            mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_figure(self, figure: Figure | Path, artifact_file: str) -> None:
        """Upload a matplotlib figure or an already rendered image file."""
        if not self.enabled:
            return
        if isinstance(figure, Path):
            parent = str(Path(artifact_file).parent)
            mlflow.log_artifact(str(figure), artifact_path=None if parent == "." else parent)
        else:
            mlflow.log_figure(figure, artifact_file)

    def log_dict(self, payload: Mapping[str, Any] | list[Any], artifact_file: str) -> None:
        """Upload a JSON/YAML-serialisable object as an artifact."""
        if self.enabled:
            mlflow.log_dict(payload, artifact_file)

    def log_text(self, text: str, artifact_file: str) -> None:
        """Upload free text as an artifact."""
        if self.enabled:
            mlflow.log_text(text, artifact_file)


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    """Return whether ``table`` exists in the SQLite database."""
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone()
    return row is not None


def relocate_mlflow_store(artifacts_dir: Path) -> int:
    """Point a copied MLflow SQLite store at the artifact files in their new location.

    MLflow records absolute ``file://`` artifact URIs. A store produced inside a remote
    container (for example under ``/root/artifacts`` on Modal) therefore references paths
    that do not exist on the machine it is copied to. This function rewrites the experiment,
    run and logged-model URIs so ``mlflow ui`` can serve the artifacts from ``artifacts_dir``.

    Args:
        artifacts_dir: Local directory holding ``mlflow.db`` and ``mlartifacts/``.

    Returns:
        Number of database rows rewritten.
    """
    settings = TrackingSettings(_env_file=None, artifacts_dir=artifacts_dir)
    new_root = settings.mlflow_artifacts_dir.resolve().as_uri()
    rewritten = 0
    with sqlite3.connect(settings.mlflow_db_path) as connection:
        prefixes: dict[str, str] = {}
        for experiment_id, location in connection.execute(
            "SELECT experiment_id, artifact_location FROM experiments"
        ).fetchall():
            leaf = PurePosixPath(urlparse(location).path or location).name
            new_location = f"{new_root}/{leaf}"
            if location != new_location:
                prefixes[location] = new_location
                connection.execute(
                    "UPDATE experiments SET artifact_location = ? WHERE experiment_id = ?",
                    (new_location, experiment_id),
                )
                rewritten += 1
        for table, id_column, uri_column in (
            ("runs", "run_uuid", "artifact_uri"),
            ("logged_models", "model_id", "artifact_location"),
        ):
            if not _table_exists(connection, table):
                continue
            for row_id, value in connection.execute(
                f"SELECT {id_column}, {uri_column} FROM {table}"  # noqa: S608 - constant identifiers
            ).fetchall():
                for old_prefix, new_prefix in prefixes.items():
                    if isinstance(value, str) and value.startswith(old_prefix):
                        connection.execute(
                            f"UPDATE {table} SET {uri_column} = ? WHERE {id_column} = ?",  # noqa: S608
                            (new_prefix + value[len(old_prefix) :], row_id),
                        )
                        rewritten += 1
                        break
    return rewritten
