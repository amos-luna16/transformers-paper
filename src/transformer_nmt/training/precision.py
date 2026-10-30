from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass

import torch

from transformer_nmt.config import PrecisionMode

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PrecisionPolicy:
    """Resolved numeric precision for a run."""

    name: str
    autocast_dtype: torch.dtype | None
    use_grad_scaler: bool

    @property
    def enabled(self) -> bool:
        """Whether autocast is active."""
        return self.autocast_dtype is not None


def resolve_precision(requested: PrecisionMode, device: torch.device) -> PrecisionPolicy:
    """Map the requested precision to what the device supports.

    ``auto`` selects bf16 on GPUs with native bfloat16 (Ampere, Ada, Hopper), fp16 with loss
    scaling on older CUDA GPUs, and fp32 on CPU/MPS. bf16 needs no gradient scaler because it
    keeps fp32's exponent range; fp16 does.

    Args:
        requested: ``auto``, ``bf16``, ``fp16`` or ``fp32``.
        device: Target device.

    Returns:
        The resolved :class:`PrecisionPolicy`.
    """
    is_cuda = device.type == "cuda"
    bf16_ok = is_cuda and torch.cuda.is_bf16_supported()

    if requested == "auto":
        requested = "bf16" if bf16_ok else ("fp16" if is_cuda else "fp32")

    if requested == "bf16" and not bf16_ok:
        logger.warning("bf16 requested but unsupported on %s; falling back to fp32", device)
        requested = "fp32"
    if requested == "fp16" and not is_cuda:
        logger.warning("fp16 autocast requires CUDA; falling back to fp32 on %s", device)
        requested = "fp32"

    if requested == "bf16":
        return PrecisionPolicy("bf16", torch.bfloat16, use_grad_scaler=False)
    if requested == "fp16":
        return PrecisionPolicy("fp16", torch.float16, use_grad_scaler=True)
    return PrecisionPolicy("fp32", None, use_grad_scaler=False)


def enable_tensor_core_math() -> None:
    """Allow TensorFloat-32 matmuls on Ampere+ GPUs for faster fp32 math at ~1e-3 precision."""
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")


@contextlib.contextmanager
def autocast_context(policy: PrecisionPolicy, device: torch.device) -> Iterator[None]:
    """Context manager enabling autocast according to ``policy`` (no-op for fp32)."""
    if not policy.enabled:
        yield
        return
    with torch.autocast(device_type=device.type, dtype=policy.autocast_dtype):
        yield


def create_grad_scaler(
    policy: PrecisionPolicy, device: torch.device
) -> torch.amp.GradScaler | None:
    """Create a ``GradScaler`` only when the policy requires loss scaling (fp16)."""
    if not policy.use_grad_scaler:
        return None
    return torch.amp.GradScaler(device.type)
