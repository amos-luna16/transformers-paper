from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(
    seed: int,
    *,
    deterministic: bool = False,
) -> None:
    """Seed Python, NumPy and PyTorch (CPU and every CUDA device).

    Args:
        seed: Base seed. In distributed training pass the same seed on every rank so model
            initialisation matches; data order is handled by the sampler.
        deterministic: Force deterministic cuDNN kernels (slower, exact reproducibility).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
