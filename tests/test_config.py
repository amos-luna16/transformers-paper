import pytest
from pydantic import ValidationError

from transformer_nmt.config import ModalSettings, ModelSettings, Settings, TrainingSettings


def test_defaults_match_reference_architecture():
    settings = Settings()
    assert (
        settings.model.d_model,
        settings.model.num_heads,
        settings.model.num_encoder_layers,
    ) == (256, 8, 3)
    assert settings.data.max_seq_len == 40 and settings.data.min_token_freq == 2
    assert settings.tracking.resolved_tracking_uri.startswith("sqlite:///")


def test_environment_overrides(monkeypatch):
    monkeypatch.setenv("TRAINING_EPOCHS", "3")
    monkeypatch.setenv("MODAL_GPU", "A100-40GB")
    monkeypatch.setenv("MODAL_GPU_COUNT", "2")
    settings = Settings()
    assert settings.training.epochs == 3
    assert settings.modal.gpu_spec == "A100-40GB:2"


def test_model_validation_rejects_bad_dimensions():
    with pytest.raises(ValueError):
        ModelSettings(_env_file=None, d_model=30, num_heads=8)
    with pytest.raises(ValueError):
        ModelSettings(_env_file=None, d_model=33, num_heads=3)


def test_gpu_spec_rejects_embedded_count():
    with pytest.raises(ValueError):
        ModalSettings(_env_file=None, gpu="H100:2")


def test_with_overrides_and_flat_dict_hide_secrets(monkeypatch):
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-secret")
    settings = Settings().with_overrides(training={"epochs": 7}, data={"max_pairs": 10})
    assert settings.training.epochs == 7 and settings.data.max_pairs == 10
    flat = settings.to_flat_dict()
    assert flat["training.epochs"] == 7 and "modal.token_id" not in flat
    assert settings.to_flat_dict(include_secrets=True)["modal.token_id"] == "ak-secret"


def test_smoke_variant_is_small():
    smoke = Settings().smoke_test_variant()
    assert smoke.training.epochs == 2 and smoke.data.max_pairs == 3000
    assert smoke.tracking.experiment_name.endswith("-smoke")


def test_training_settings_are_frozen():
    training = TrainingSettings(_env_file=None)
    with pytest.raises(ValidationError):
        training.epochs = 10  # type: ignore[misc]


def test_to_environment_round_trips_through_settings(monkeypatch):
    original = Settings().with_overrides(
        training={"epochs": 9, "precision": "bf16", "compile_model": True},
        model={"d_model": 64, "num_heads": 4},
        data={"max_pairs": 123},
    )
    environment = original.to_environment()
    assert environment["TRAINING_EPOCHS"] == "9" and environment["TRAINING_COMPILE_MODEL"] == "true"
    assert "TRACKING_RUN_NAME" not in environment  # None values are skipped
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    restored = Settings()
    assert restored.training == original.training
    assert restored.model == original.model
    assert restored.data == original.data


def test_credential_environment_only_contains_present_values(monkeypatch):
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-1")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-1")
    assert ModalSettings().credential_environment() == {
        "MODAL_TOKEN_ID": "ak-1",
        "MODAL_TOKEN_SECRET": "as-1",
    }
    monkeypatch.delenv("MODAL_TOKEN_ID")
    monkeypatch.delenv("MODAL_TOKEN_SECRET")
    assert ModalSettings(_env_file=None).credential_environment() == {}
