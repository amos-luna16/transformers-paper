from mlflow.tracking import MlflowClient

from transformer_nmt.config import TrackingSettings
from transformer_nmt.training.tracking import ExperimentTracker, relocate_mlflow_store


def _settings(tmp_path, **overrides) -> TrackingSettings:
    return TrackingSettings(
        _env_file=None,
        artifacts_dir=tmp_path / "artifacts",
        experiment_name="unit",
        log_system_metrics=False,
        **overrides,
    )


def test_tracker_logs_params_metrics_and_artifacts(tmp_path):
    settings = _settings(tmp_path)
    tracker = ExperimentTracker(settings)
    with tracker.run("run-1", tags={"run_type": "test"}):
        tracker.log_params({"a": 1, "b": "x" * 600})
        tracker.log_metrics({"val/loss": 1.0, "skipped": None}, step=1)
        tracker.log_dict({"k": "v"}, "payload.json")
        run_id = tracker.run_id
    assert settings.mlflow_db_path.exists()
    client = MlflowClient(tracking_uri=settings.resolved_tracking_uri)
    run = client.get_run(run_id)
    assert run.data.params["a"] == "1" and len(run.data.params["b"]) == 500
    assert run.data.metrics["val/loss"] == 1.0 and "skipped" not in run.data.metrics
    assert run.data.tags["run_type"] == "test"
    assert run.info.artifact_uri.startswith(settings.mlflow_artifacts_dir.resolve().as_uri())
    assert any(f.path == "payload.json" for f in client.list_artifacts(run_id))


def test_disabled_tracker_is_noop(tmp_path):
    settings = _settings(tmp_path)
    tracker = ExperimentTracker(settings, enabled=False)
    with tracker.run("ignored"):
        tracker.log_metrics({"x": 1.0}, step=1)
    assert tracker.run_id is None and not settings.mlflow_db_path.exists()


def test_relocate_store_rewrites_experiment_and_run_uris(tmp_path):
    remote_settings = _settings(tmp_path)
    client = MlflowClient(tracking_uri=remote_settings.resolved_tracking_uri)
    experiment_id = client.create_experiment(
        "remote", artifact_location="file:///root/artifacts/mlartifacts/remote"
    )
    run = client.create_run(experiment_id)
    assert run.info.artifact_uri.startswith("file:///root/artifacts")

    rewritten = relocate_mlflow_store(remote_settings.artifacts_dir)
    assert rewritten == 3  # Default experiment, remote experiment and its run
    new_root = remote_settings.mlflow_artifacts_dir.resolve().as_uri()
    client = MlflowClient(tracking_uri=remote_settings.resolved_tracking_uri)
    assert client.get_experiment(experiment_id).artifact_location == f"{new_root}/remote"
    assert (
        client.get_run(run.info.run_id).info.artifact_uri
        == f"{new_root}/remote/{run.info.run_id}/artifacts"
    )
    assert relocate_mlflow_store(remote_settings.artifacts_dir) == 0
