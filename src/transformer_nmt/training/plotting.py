from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from transformer_nmt.training.metrics import TrainingHistory  # noqa: E402

TRAIN_COLOR = "#2a78d6"
VALIDATION_COLOR = "#eb6834"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GRID_COLOR = "#d9d8d4"
LINE_WIDTH = 2.0
MARKER_SIZE = 6.0


def _style_axes(
    axes: Axes,
    title: str,
    ylabel: str,
    xlabel: str = "epoch",
) -> None:
    """Apply the recessive grid/spine styling shared by every panel."""
    axes.set_facecolor(SURFACE)
    axes.set_title(title, loc="left", fontsize=11, color=TEXT_PRIMARY, fontweight="bold")
    axes.set_xlabel(xlabel, color=TEXT_SECONDARY)
    axes.set_ylabel(ylabel, color=TEXT_SECONDARY)
    axes.grid(True, color=GRID_COLOR, linewidth=0.8, alpha=0.6)
    axes.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID_COLOR)


def _plot_series(
    axes: Axes,
    x: Sequence[float],
    y: Sequence[float],
    *,
    label: str,
    color: str,
) -> None:
    """Draw one line-with-markers series."""
    axes.plot(
        x, y, label=label, color=color, linewidth=LINE_WIDTH, marker="o", markersize=MARKER_SIZE
    )


def _epochs_for(
    history: TrainingHistory,
    key: str,
) -> list[int]:
    """Return the epochs where ``key`` is present."""
    return [int(row["epoch"]) for row in history.rows if row.get(key) is not None]


def plot_training_curves(
    history: TrainingHistory,
    output_path: Path,
    *,
    title: str,
) -> Path:
    """Render the per-epoch training dashboard as small multiples and save it as PNG.

    Panels: loss (train vs validation, best epoch annotated), validation perplexity, token
    accuracy (train vs validation), BLEU, learning rate and gradient norm. Each panel keeps a
    single y-axis.

    Args:
        history: Per-epoch metrics.
        output_path: PNG destination; parent directories are created.
        title: Figure title (typically the run name).

    Returns:
        The written path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, panels = plt.subplots(2, 3, figsize=(16, 8.5), facecolor=SURFACE)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=TEXT_PRIMARY, fontweight="bold")

    loss_axes = panels[0][0]
    _plot_series(
        loss_axes,
        _epochs_for(history, "train/loss"),
        history.series("train/loss"),
        label="train (label-smoothed)",
        color=TRAIN_COLOR,
    )
    _plot_series(
        loss_axes,
        _epochs_for(history, "val/loss"),
        history.series("val/loss"),
        label="validation (NLL)",
        color=VALIDATION_COLOR,
    )
    best = history.best("val/loss", minimize=True)
    if best is not None:
        best_epoch, best_value = best
        loss_axes.annotate(
            f"best {best_value:.3f} @ epoch {best_epoch}",
            xy=(best_epoch, best_value),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=9,
            color=TEXT_SECONDARY,
        )
    _style_axes(loss_axes, "Cross-entropy loss", "loss")
    loss_axes.legend(frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY)

    perplexity_axes = panels[0][1]
    _plot_series(
        perplexity_axes,
        _epochs_for(history, "val/perplexity"),
        history.series("val/perplexity"),
        label="validation",
        color=VALIDATION_COLOR,
    )
    _style_axes(perplexity_axes, "Validation perplexity", "perplexity")
    perplexity_axes.set_yscale("log")

    accuracy_axes = panels[0][2]
    _plot_series(
        accuracy_axes,
        _epochs_for(history, "train/token_accuracy"),
        history.series("train/token_accuracy"),
        label="train",
        color=TRAIN_COLOR,
    )
    _plot_series(
        accuracy_axes,
        _epochs_for(history, "val/token_accuracy"),
        history.series("val/token_accuracy"),
        label="validation",
        color=VALIDATION_COLOR,
    )
    _style_axes(accuracy_axes, "Token accuracy", "accuracy")
    accuracy_axes.set_ylim(0, 1)
    accuracy_axes.legend(frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY)

    bleu_axes = panels[1][0]
    _plot_series(
        bleu_axes,
        _epochs_for(history, "val/bleu"),
        history.series("val/bleu"),
        label="validation",
        color=VALIDATION_COLOR,
    )
    _style_axes(bleu_axes, "Validation BLEU (greedy, sacreBLEU)", "BLEU")
    bleu_axes.set_ylim(bottom=0)

    lr_axes = panels[1][1]
    _plot_series(
        lr_axes,
        _epochs_for(history, "train/learning_rate"),
        history.series("train/learning_rate"),
        label="lr",
        color=TRAIN_COLOR,
    )
    _style_axes(lr_axes, "Learning rate at epoch end", "learning rate")
    lr_axes.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))

    grad_axes = panels[1][2]
    _plot_series(
        grad_axes,
        _epochs_for(history, "train/grad_norm"),
        history.series("train/grad_norm"),
        label="grad norm",
        color=TRAIN_COLOR,
    )
    _style_axes(grad_axes, "Mean gradient norm (pre-clip)", "L2 norm")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return output_path


def plot_step_dynamics(
    step_records: Sequence[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
) -> Path:
    """Render step-level training loss and the learning-rate schedule actually followed.

    Args:
        step_records: Dicts with ``step``, ``step/loss`` and ``step/learning_rate``.
        output_path: PNG destination.
        title: Figure title.

    Returns:
        The written path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    steps = [int(record["step"]) for record in step_records]
    fig: Figure
    fig, (loss_axes, lr_axes) = plt.subplots(1, 2, figsize=(14, 4.5), facecolor=SURFACE)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=TEXT_PRIMARY, fontweight="bold")

    loss_axes.plot(
        steps,
        [record["step/loss"] for record in step_records],
        color=TRAIN_COLOR,
        linewidth=1.5,
    )
    _style_axes(loss_axes, "Training loss per logged step", "loss", xlabel="optimizer step")

    lr_axes.plot(
        steps,
        [record["step/learning_rate"] for record in step_records],
        color=TRAIN_COLOR,
        linewidth=LINE_WIDTH,
    )
    _style_axes(lr_axes, "Learning-rate schedule", "learning rate", xlabel="optimizer step")
    lr_axes.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return output_path
