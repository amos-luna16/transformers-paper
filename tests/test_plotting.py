from transformer_nmt.training.metrics import TrainingHistory
from transformer_nmt.training.plotting import plot_step_dynamics, plot_training_curves


def test_plots_are_written(tmp_path):
    history = TrainingHistory()
    for epoch in (1, 2, 3):
        history.append(
            {
                "epoch": epoch,
                "train/loss": 3.0 / epoch,
                "val/loss": 3.5 / epoch,
                "val/perplexity": 10.0 / epoch,
                "train/token_accuracy": 0.2 * epoch,
                "val/token_accuracy": 0.15 * epoch,
                "val/bleu": 5.0 * epoch,
                "train/learning_rate": 1e-3 / epoch,
                "train/grad_norm": 1.0,
            }
        )
    curves = plot_training_curves(history, tmp_path / "plots" / "curves.png", title="test")
    dynamics = plot_step_dynamics(
        [{"step": s, "step/loss": 3.0 / s, "step/learning_rate": 1e-4 * s} for s in (1, 2, 3)],
        tmp_path / "plots" / "steps.png",
        title="test",
    )
    assert curves.exists() and curves.stat().st_size > 1000
    assert dynamics.exists() and dynamics.stat().st_size > 1000
