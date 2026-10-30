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
    for name, value in (("run-a", 2.0), ("run-b", 1.5)):
        run_dir = tmp_path / name
        run_dir.mkdir()
        BestCheckpointSummary(run_name=name, metric="val/loss", value=value, epoch=1).save(
            run_dir / "best.json"
        )
    assert find_best_checkpoint(tmp_path) == tmp_path / "run-b" / "best.pt"


def test_find_best_checkpoint_prefers_highest_bleu(tmp_path):
    for name, value in (("run-a", 20.0), ("run-b", 15.0)):
        run_dir = tmp_path / name
        run_dir.mkdir()
        (run_dir / "best.json").write_text(
            json.dumps(
                {
                    "run_name": name,
                    "metric": "val/bleu",
                    "value": value,
                    "epoch": 1,
                    "checkpoint_file": "best.pt",
                }
            )
        )
    assert find_best_checkpoint(tmp_path) == tmp_path / "run-a" / "best.pt"


def test_find_best_checkpoint_raises_when_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_best_checkpoint(tmp_path)


def test_unwrap_model_returns_plain_module():
    linear = torch.nn.Linear(2, 2)
    assert unwrap_model(linear) is linear


def test_find_best_checkpoint_skips_smoke_runs_when_full_runs_exist(tmp_path):
    for name, value in (("smoke-run", 0.5), ("full-run", 1.5)):
        run_dir = tmp_path / name
        run_dir.mkdir()
        BestCheckpointSummary(run_name=name, metric="val/loss", value=value, epoch=1).save(
            run_dir / "best.json"
        )
    assert find_best_checkpoint(tmp_path) == tmp_path / "full-run" / "best.pt"
    assert find_best_checkpoint(tmp_path, exclude=()) == tmp_path / "smoke-run" / "best.pt"
