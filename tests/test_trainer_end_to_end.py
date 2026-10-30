import json

import torch

from transformer_nmt.inference.translator import Translator
from transformer_nmt.training.checkpoint import (
    BEST_CHECKPOINT_NAME,
    BEST_SUMMARY_NAME,
    find_best_checkpoint,
)
from transformer_nmt.training.distributed import DistributedContext
from transformer_nmt.training.tracking import ExperimentTracker
from transformer_nmt.training.trainer import Trainer, generate_run_name


def test_generate_run_name_slug():
    name = generate_run_name("NVIDIA L40S", "bf16", 256, run_type="full", language_pair="eng-spa")
    assert name.endswith("-eng-spa-l40s-bf16-bs256-full")
    assert generate_run_name("", "fp32", 8, run_type="smoke").endswith(
        "-eng-fra-cpu-fp32-bs8-smoke"
    )


def test_trainer_runs_end_to_end_and_translator_loads(tiny_settings, tiny_bundle):
    settings = tiny_settings.with_overrides(training={"epochs": 2})
    context = DistributedContext(rank=0, local_rank=0, world_size=1, device=torch.device("cpu"))
    tracker = ExperimentTracker(settings.tracking)
    with tracker.run("unit-run", tags={"run_type": "test"}):
        trainer = Trainer(settings, tiny_bundle, context, tracker, run_name="unit-run")
        summary = trainer.fit()

    assert summary.epochs_completed == 2 and summary.best_epoch in (1, 2)
    assert len(trainer.history.rows) == 2 and "val/bleu" in trainer.history.rows[0]
    assert trainer.history.rows[1]["train/loss"] < trainer.history.rows[0]["train/loss"] * 1.5
    run_dir = settings.tracking.checkpoints_dir / "unit-run"
    assert (run_dir / BEST_CHECKPOINT_NAME).exists() and (run_dir / "last.pt").exists()
    assert json.loads((run_dir / BEST_SUMMARY_NAME).read_text())["metric"] == "val/loss"
    assert (settings.tracking.plots_dir / "unit-run" / "training_curves.png").exists()
    assert find_best_checkpoint(settings.tracking.artifacts_dir) == run_dir / BEST_CHECKPOINT_NAME

    translator = Translator.from_best_run(
        settings.tracking.artifacts_dir, torch.device("cpu"), max_decode_len=8
    )
    greedy = translator.translate("i am happy .", beam_size=1)
    beam = translator.translate("i am happy .", beam_size=3)
    assert isinstance(greedy, str) and isinstance(beam, str)
    assert len(translator.translate_batch(["i am happy .", "come here !"])) == 2
