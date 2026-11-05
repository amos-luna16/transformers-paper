import json

import pytest
import torch

from transformer_nmt.training.checkpoint import (
    BestCheckpointSummary,
    discover_checkpoints,
    load_checkpoint,
    rank_checkpoints,
    save_checkpoint,
    unwrap_model,
)


def _write_run(
    root,
    name,
    *,
    value,
    metric="val/loss",
    run_type="full",
    complete=True,
):
    run_dir = root / name
    run_dir.mkdir(parents=True)
    BestCheckpointSummary(
        run_name=name, metric=metric, value=value, epoch=1, run_type=run_type
    ).save(run_dir / "best.json")
    if complete:
        (run_dir / "best.pt").write_bytes(b"stub")
        (run_dir / "settings.json").write_text(json.dumps({"data": {}}))
    return run_dir


def _names(
    candidates,
):
    return [candidate.summary.run_name for candidate in candidates]


def test_save_and_load_checkpoint_roundtrip(
    tmp_path,
):
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


def test_discover_requires_weights_and_settings_next_to_the_card(
    tmp_path,
):
    _write_run(tmp_path / "checkpoints", "complete", value=2.0)
    _write_run(
        tmp_path / "mlartifacts" / "exp" / "run" / "artifacts" / "checkpoints",
        "copy",
        value=1.5,
        complete=False,
    )
    assert _names(discover_checkpoints(tmp_path)) == ["complete"]
    assert discover_checkpoints(tmp_path / "nothing") == []


def test_rank_prefers_lowest_loss_or_highest_bleu(
    tmp_path,
):
    _write_run(tmp_path, "loss-a", value=2.0)
    _write_run(tmp_path, "loss-b", value=1.5)
    assert _names(rank_checkpoints(discover_checkpoints(tmp_path))) == ["loss-b", "loss-a"]
    bleu_root = tmp_path / "bleu"
    _write_run(bleu_root, "bleu-a", value=20.0, metric="val/bleu")
    _write_run(bleu_root, "bleu-b", value=15.0, metric="val/bleu")
    assert _names(rank_checkpoints(discover_checkpoints(bleu_root))) == ["bleu-a", "bleu-b"]


def test_rank_sets_smoke_runs_aside_unless_nothing_else_exists(
    tmp_path,
):
    _write_run(tmp_path, "quick", value=0.5, run_type="smoke")
    _write_run(tmp_path, "full-run", value=1.5)
    assert _names(rank_checkpoints(discover_checkpoints(tmp_path))) == ["full-run"]
    only_smoke = tmp_path / "only"
    _write_run(only_smoke, "quick", value=0.5, run_type="smoke")
    assert _names(rank_checkpoints(discover_checkpoints(only_smoke))) == ["quick"]


def test_rank_rejects_mixed_metrics_unless_one_is_chosen(
    tmp_path,
):
    _write_run(tmp_path, "loss-run", value=1.0)
    _write_run(tmp_path, "bleu-run", value=30.0, metric="val/bleu")
    candidates = discover_checkpoints(tmp_path)
    with pytest.raises(ValueError):
        rank_checkpoints(candidates)
    assert _names(rank_checkpoints(candidates, metric="val/bleu")) == ["bleu-run"]


def test_unwrap_model_returns_plain_module():
    linear = torch.nn.Linear(2, 2)
    assert unwrap_model(linear) is linear
