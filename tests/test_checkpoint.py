import json

import pytest
import torch

from transformer_nmt.training.checkpoint import (
    BestCheckpointSummary,
    find_best_checkpoint,
    load_checkpoint,
    save_checkpoint,
    unwrap_model,
)


def _write_run(root, name, *, value, metric="val/loss", run_type="full", with_checkpoint=True):
    run_dir = root / name
    run_dir.mkdir(parents=True)
    BestCheckpointSummary(
        run_name=name, metric=metric, value=value, epoch=1, run_type=run_type
    ).save(run_dir / "best.json")
    if with_checkpoint:
        (run_dir / "best.pt").write_bytes(b"stub")
    return run_dir


def test_save_and_load_checkpoint_roundtrip(tmp_path):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5)
    path = save_checkpoint(
        tmp_path / "ckpt.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        epoch=3,
        global_step=42,
        best_metric_value=1.23,
        extra={"run_name": "r"},
    )
    payload = load_checkpoint(path)
    assert payload["epoch"] == 3 and payload["global_step"] == 42 and payload["run_name"] == "r"
    restored = torch.nn.Linear(3, 2)
    restored.load_state_dict(payload["model_state_dict"])
    assert torch.equal(restored.weight, model.weight)


def test_find_best_checkpoint_prefers_lowest_loss(tmp_path):
    _write_run(tmp_path, "run-a", value=2.0)
    _write_run(tmp_path, "run-b", value=1.5)
    assert find_best_checkpoint(tmp_path) == tmp_path / "run-b" / "best.pt"


def test_find_best_checkpoint_prefers_highest_bleu(tmp_path):
    for name, value in (("run-a", 20.0), ("run-b", 15.0)):
        run_dir = tmp_path / name
        run_dir.mkdir()
        (run_dir / "best.json").write_text(
            json.dumps({"run_name": name, "metric": "val/bleu", "value": value, "epoch": 1})
        )
        (run_dir / "best.pt").write_bytes(b"stub")
    assert find_best_checkpoint(tmp_path) == tmp_path / "run-a" / "best.pt"


def test_find_best_checkpoint_raises_when_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_best_checkpoint(tmp_path)


def test_find_best_checkpoint_skips_smoke_runs_when_full_runs_exist(tmp_path):
    _write_run(tmp_path, "quick", value=0.5, run_type="smoke")
    _write_run(tmp_path, "named-smoke-run", value=0.6)
    _write_run(tmp_path, "full-run", value=1.5)
    assert find_best_checkpoint(tmp_path) == tmp_path / "full-run" / "best.pt"
    # Without the name heuristic only the recorded run_type is used.
    assert find_best_checkpoint(tmp_path, exclude=()) == tmp_path / "named-smoke-run" / "best.pt"


def test_find_best_checkpoint_falls_back_to_smoke_runs(tmp_path):
    _write_run(tmp_path, "quick", value=0.5, run_type="smoke")
    assert find_best_checkpoint(tmp_path) == tmp_path / "quick" / "best.pt"


def test_find_best_checkpoint_ignores_sidecars_without_weights(tmp_path):
    _write_run(tmp_path / "checkpoints", "new", value=2.0)
    _write_run(
        tmp_path / "mlartifacts" / "exp" / "id" / "artifacts" / "run",
        "new",
        value=1.5,
        with_checkpoint=False,
    )
    assert find_best_checkpoint(tmp_path) == tmp_path / "checkpoints" / "new" / "best.pt"


def test_find_best_checkpoint_requires_a_single_metric_or_an_explicit_one(tmp_path):
    _write_run(tmp_path, "loss-run", value=1.0, metric="val/loss")
    _write_run(tmp_path, "bleu-run", value=30.0, metric="val/bleu")
    with pytest.raises(ValueError):
        find_best_checkpoint(tmp_path)
    assert find_best_checkpoint(tmp_path, metric="val/bleu") == tmp_path / "bleu-run" / "best.pt"


def test_unwrap_model_returns_plain_module():
    linear = torch.nn.Linear(2, 2)
    assert unwrap_model(linear) is linear
