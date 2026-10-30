import math

import torch

from transformer_nmt.training.scheduler import (
    create_lr_scheduler,
    noam_learning_rate,
    warmup_cosine_factor,
)


def test_noam_peaks_at_warmup():
    warmup, d_model = 100, 64
    values = [noam_learning_rate(step, d_model, warmup) for step in range(1, 301)]
    peak_step = values.index(max(values)) + 1
    assert peak_step == warmup
    assert values[0] < values[50] < values[99] and values[150] > values[299]
    assert math.isclose(values[99], d_model**-0.5 * warmup**-0.5)


def test_cosine_factor_boundaries():
    assert warmup_cosine_factor(5, warmup_steps=10, total_steps=100) == 0.5
    assert math.isclose(warmup_cosine_factor(10, 10, 100), 1.0)
    assert math.isclose(warmup_cosine_factor(100, 10, 100), 0.05)


def test_create_lr_scheduler_drives_optimizer():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = create_lr_scheduler(
        optimizer, name="noam", d_model=64, warmup_steps=10, total_steps=50
    )
    seen = []
    for _ in range(20):
        optimizer.step()
        scheduler.step()
        seen.append(scheduler.get_last_lr()[0])
    assert seen.index(max(seen)) == 8  # lr for step 10 (the warm-up peak) is set after step 9
    assert seen[-1] < seen[8]
