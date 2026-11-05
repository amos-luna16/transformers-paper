from __future__ import annotations

import math
from collections.abc import Callable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from transformer_nmt.config import SchedulerName


def noam_learning_rate(
    step: int,
    d_model: int,
    warmup_steps: int,
    scale: float = 1.0,
) -> float:
    """Learning rate of the schedule in section 5.3 of the paper.

    ``lr = scale * d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5)``: linear warm-up
    followed by inverse-square-root decay, peaking at ``step == warmup_steps``.

    Args:
        step: One-based optimisation step.
        d_model: Model dimension.
        warmup_steps: Number of warm-up steps.
        scale: Constant multiplier on the schedule.

    Returns:
        The learning rate for ``step``.
    """
    step = max(step, 1)
    return scale * d_model**-0.5 * min(step**-0.5, step * warmup_steps**-1.5)


def warmup_cosine_factor(
    step: int,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.05,
) -> float:
    """Multiplier on the peak learning rate for linear warm-up followed by cosine decay.

    Args:
        step: One-based optimisation step.
        warmup_steps: Steps to ramp linearly from 0 to the peak.
        total_steps: Total steps after which the factor reaches ``min_lr_ratio``.
        min_lr_ratio: Floor of the schedule relative to the peak.

    Returns:
        A factor in ``[0, 1]``.
    """
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def create_lr_scheduler(
    optimizer: Optimizer,
    *,
    name: SchedulerName,
    d_model: int,
    warmup_steps: int,
    total_steps: int,
    noam_scale: float = 1.0,
) -> LambdaLR:
    """Create a per-step ``LambdaLR`` scheduler.

    For ``"noam"`` the optimizer's base learning rate must be ``1.0`` because the lambda
    returns the absolute learning rate. For ``"cosine"`` the base learning rate is the peak.

    Args:
        optimizer: Optimizer whose learning rate is scheduled.
        name: ``"noam"`` (paper) or ``"cosine"`` (warm-up + cosine decay).
        d_model: Model dimension used by the Noam schedule.
        warmup_steps: Warm-up steps.
        total_steps: Total training steps (cosine schedule only).
        noam_scale: Multiplier on the Noam schedule.

    Returns:
        The configured scheduler; call ``scheduler.step()`` after every optimizer step.
    """
    schedule: Callable[[int], float]
    if name == "noam":
        schedule = lambda step: noam_learning_rate(step + 1, d_model, warmup_steps, noam_scale)  # noqa: E731
    elif name == "cosine":
        schedule = lambda step: warmup_cosine_factor(step + 1, warmup_steps, total_steps)  # noqa: E731
    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"Unknown scheduler '{name}'")
    return LambdaLR(optimizer, lr_lambda=schedule)
