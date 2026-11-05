from __future__ import annotations

import os
import platform
import socket
import subprocess
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class HardwareInfo:
    """Snapshot of the software and hardware a run executes on."""

    hostname: str
    platform: str
    python_version: str
    torch_version: str
    cuda_version: str | None
    cudnn_version: int | None
    device_type: str
    device_name: str
    device_count: int
    total_memory_gb: float | None
    compute_capability: str | None
    bf16_supported: bool
    cpu_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""
        return asdict(self)


def resolve_device(
    preferred: str | None = None,
) -> torch.device:
    """Pick the best available device unless ``preferred`` is given.

    Args:
        preferred: Explicit device string such as ``cuda:1``, ``mps`` or ``cpu``.

    Returns:
        The selected device.
    """
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collect_hardware_info(
    device: torch.device,
) -> HardwareInfo:
    """Gather hardware/software metadata for experiment tracking.

    Args:
        device: Device the model runs on.

    Returns:
        A populated :class:`HardwareInfo`.
    """
    is_cuda = device.type == "cuda"
    properties = torch.cuda.get_device_properties(device) if is_cuda else None
    if is_cuda and properties is not None:
        device_name = properties.name
        total_memory_gb: float | None = round(properties.total_memory / 1024**3, 2)
        compute_capability: str | None = f"{properties.major}.{properties.minor}"
        bf16_supported = torch.cuda.is_bf16_supported()
    else:
        device_name = platform.processor() or platform.machine()
        total_memory_gb = None
        compute_capability = None
        bf16_supported = False
    return HardwareInfo(
        hostname=socket.gethostname(),
        platform=platform.platform(),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda if is_cuda else None,
        cudnn_version=torch.backends.cudnn.version() if is_cuda else None,
        device_type=device.type,
        device_name=device_name,
        device_count=torch.cuda.device_count() if is_cuda else 1,
        total_memory_gb=total_memory_gb,
        compute_capability=compute_capability,
        bf16_supported=bf16_supported,
        cpu_count=os.cpu_count() or 1,
    )


def peak_memory_gb(
    device: torch.device,
) -> float | None:
    """Return the peak allocated CUDA memory in GiB since the last reset, if applicable."""
    if device.type != "cuda":
        return None
    return round(torch.cuda.max_memory_allocated(device) / 1024**3, 3)


def git_commit_hash() -> str | None:
    """Return the current git commit hash, or ``None`` outside a repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None
