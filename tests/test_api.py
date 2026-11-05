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
def trained_runs(
    tmp_path_factory,
):
    """Train three tiny runs: en->fr (full), en->fr (smoke) and es->en (full)."""
    root = tmp_path_factory.mktemp("api-artifacts")
    base = make_tiny_settings(root)
    bundle = make_tiny_bundle(base)
    context = DistributedContext(rank=0, local_rank=0, world_size=1, device=torch.device("cpu"))
    runs = (
        ("api-en-fr", "full", "English", "French"),
        ("api-en-fr-smoke", "smoke", "English", "French"),
        ("api-es-en", "full", "Spanish", "English"),
    )
    for run_name, run_type, source, target in runs:
        settings = base.with_overrides(data={"source_language": source, "target_language": target})
        tracker = ExperimentTracker(settings.tracking, enabled=False)
        Trainer(settings, bundle, context, tracker, run_name, run_type=run_type).fit()
    return base, root


@pytest.fixture
def client(
    trained_runs,
):
    settings, root = trained_runs
    with TestClient(create_app(settings, models_root=root)) as test_client:
        yield test_client


def test_health_reports_one_active_model_per_language_pair(
    client,
):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["available_models"] == 3
    assert body["device"] in {"cpu", "mps"}
    active = {
        (m["source_language"], m["target_language"]): m["run_name"] for m in body["active_models"]
    }
    assert active == {("en", "fr"): "api-en-fr", ("es", "en"): "api-es-en"}


def test_languages_lists_directions_with_counts(
    client,
):
    pairs = client.get("/languages").json()["pairs"]
    assert pairs == [
        {
            "source_language": "en",
            "target_language": "fr",
            "models": 2,
            "active_model": "api-en-fr",
        },
        {
            "source_language": "es",
            "target_language": "en",
            "models": 1,
            "active_model": "api-es-en",
        },
    ]


def test_models_are_grouped_by_pair_best_first_with_minimal_fields(
    client,
):
    models = client.get("/models").json()["models"]
    assert [m["run_name"] for m in models] == ["api-en-fr", "api-en-fr-smoke", "api-es-en"]
    assert set(models[0]) == {
        "run_name",
        "source_language",
        "target_language",
        "metric",
        "value",
        "epoch",
        "run_type",
        "active",
    }
    assert models[0]["active"] and not models[1]["active"] and models[2]["active"]


def test_translate_uses_the_model_of_the_requested_direction(
    client,
):
    body = {
        "text": "i am happy .",
        "source_language": "EN",
        "target_language": "fr",
        "beam_size": 2,
    }
    response = client.post("/translate", json=body).json()
    assert set(response) == {"translation", "model"} and response["model"] == "api-en-fr"
    body = {"text": "estoy feliz .", "source_language": "es", "target_language": "en"}
    assert client.post("/translate", json=body).json()["model"] == "api-es-en"


def test_manual_selection_switches_only_its_language_pair(
    client,
):
    assert client.post("/models/select", json={"run_name": "missing"}).status_code == 404
    selected = client.post("/models/select", json={"run_name": "api-en-fr-smoke"}).json()
    assert selected["run_name"] == "api-en-fr-smoke" and selected["active"]
    en_fr = {"text": "i am happy .", "source_language": "en", "target_language": "fr"}
    es_en = {"text": "estoy feliz .", "source_language": "es", "target_language": "en"}
    assert client.post("/translate", json=en_fr).json()["model"] == "api-en-fr-smoke"
    assert client.post("/translate", json=es_en).json()["model"] == "api-es-en"


def test_unknown_direction_and_bad_bodies_are_rejected(
    client,
):
    body = {"text": "hello", "source_language": "en", "target_language": "de"}
    assert client.post("/translate", json=body).status_code == 404
    assert (
        client.post(
            "/translate", json={"text": "", "source_language": "en", "target_language": "fr"}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/translate", json={"text": "hi", "source_language": "eng", "target_language": "fr"}
        ).status_code
        == 422
    )
    assert client.post("/translate", json={"text": "hi"}).status_code == 422


def test_stream_emits_one_event_per_token_then_done(
    client,
):
    body = {"text": "i am happy .", "source_language": "en", "target_language": "fr"}
    with client.stream("POST", "/translate/stream", json=body) as response:
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


def test_without_autoload_a_model_must_be_selected_first(
    trained_runs,
):
    settings, root = trained_runs
    settings = settings.with_overrides(inference={"autoload_best_model": False})
    with TestClient(create_app(settings, models_root=root)) as test_client:
        assert test_client.get("/health").json()["status"] == "no-model-selected"
        body = {"text": "hello", "source_language": "en", "target_language": "fr"}
        assert test_client.post("/translate", json=body).status_code == 409
        test_client.post("/models/select", json={"run_name": "api-en-fr"})
        assert test_client.post("/translate", json=body).json()["model"] == "api-en-fr"
