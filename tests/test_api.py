import json

import pytest
import torch
from fastapi.testclient import TestClient

from tests.conftest import make_tiny_bundle, make_tiny_settings
from transformer_nmt.api.app import create_app
from transformer_nmt.training.distributed import DistributedContext
from transformer_nmt.training.tracking import ExperimentTracker
from transformer_nmt.training.trainer import Trainer


@pytest.fixture(scope="module")
def trained_runs(tmp_path_factory):
    """Train two tiny runs (one smoke, one full) so the API has models to choose from."""
    root = tmp_path_factory.mktemp("api-artifacts")
    settings = make_tiny_settings(root)
    bundle = make_tiny_bundle(settings)
    context = DistributedContext(rank=0, local_rank=0, world_size=1, device=torch.device("cpu"))
    for run_name, run_type in (("api-full", "full"), ("api-smoke", "smoke")):
        tracker = ExperimentTracker(settings.tracking, enabled=False)
        Trainer(settings, bundle, context, tracker, run_name, run_type=run_type).fit()
    return settings, root


@pytest.fixture
def client(trained_runs):
    settings, root = trained_runs
    with TestClient(create_app(settings, models_root=root)) as test_client:
        yield test_client


def test_health_reports_device_models_and_active_run(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["available_models"] == 2
    assert body["device"] in {"cpu", "mps"} and body["active_model"]["run_name"] == "api-full"


def test_models_are_listed_best_first_with_smoke_runs_last(client):
    body = client.get("/models").json()
    assert [m["run_name"] for m in body["models"]] == ["api-full", "api-smoke"]
    assert body["models"][0]["is_active"] and body["active_run_name"] == "api-full"


def test_manual_selection_switches_the_model_used_for_translation(client):
    assert client.post("/models/select", json={"run_name": "missing"}).status_code == 404
    selected = client.post("/models/select", json={"run_name": "api-smoke"}).json()
    assert selected["run_name"] == "api-smoke" and selected["is_active"]
    response = client.post("/translate", json={"text": "i am happy .", "beam_size": 2}).json()
    assert response["run_name"] == "api-smoke" and isinstance(response["translation"], str)
    assert client.get("/health").json()["active_model"]["run_name"] == "api-smoke"


def test_translate_validates_input(client):
    assert client.post("/translate", json={"text": ""}).status_code == 422
    assert client.post("/translate", json={"text": "hi", "beam_size": 0}).status_code == 422


def test_stream_emits_one_event_per_token_then_done(client):
    with client.stream("POST", "/translate/stream", json={"text": "i am happy ."}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        lines = [line for line in response.iter_lines() if line]
    assert lines[-2:] == ["event: done", "data: {}"]
    tokens = [
        json.loads(line[len("data: ") :])["token"]
        for line in lines[:-2]
        if line.startswith("data: ")
    ]
    assert all(isinstance(token, str) and token for token in tokens)


def test_requests_without_a_selected_model_return_409(trained_runs):
    settings, root = trained_runs
    settings = settings.with_overrides(inference={"autoload_best_model": False})
    with TestClient(create_app(settings, models_root=root)) as test_client:
        assert test_client.get("/health").json()["status"] == "no-model-selected"
        assert test_client.post("/translate", json={"text": "hello"}).status_code == 409
