import torch

from transformer_nmt.training.distributed import DistributedContext
from transformer_nmt.training.precision import PrecisionPolicy
from transformer_nmt.training.tracking import ExperimentTracker
from transformer_nmt.training.trainer import Trainer


def test_grad_scaler_path_trains_and_reports_finite_grad_norms(
    tiny_settings,
    tiny_bundle,
):
    context = DistributedContext(rank=0, local_rank=0, world_size=1, device=torch.device("cpu"))
    tracker = ExperimentTracker(tiny_settings.tracking, enabled=False)
    trainer = Trainer(tiny_settings, tiny_bundle, context, tracker, run_name="scaler-run")
    # Exercise the loss-scaling branch (scale -> unscale -> clip -> step -> update) on CPU.
    trainer.precision = PrecisionPolicy("fp32", None, use_grad_scaler=True)
    trainer.scaler = torch.amp.GradScaler("cpu", init_scale=2.0**4)
    summary = trainer.fit()
    assert summary.epochs_completed == 1
    row = trainer.history.rows[0]
    assert row["train/grad_norm"] > 0 and row["train/skipped_steps"] == 0
    assert all(record["step/grad_norm"] is not None for record in trainer.step_records)
