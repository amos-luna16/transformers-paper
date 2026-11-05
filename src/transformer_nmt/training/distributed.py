from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor

from transformer_nmt.utils.hardware import resolve_device

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DistributedContext:
    """Process-group coordinates for the current process."""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_distributed(self) -> bool:
        """Whether more than one process participates."""
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        """Whether this process should log, checkpoint and track metrics."""
        return self.rank == 0


def initialize_distributed(
    preferred_device: str | None = None,
) -> DistributedContext:
    """Initialise ``torch.distributed`` from the environment set by ``torchrun``.

    When ``WORLD_SIZE`` is absent or ``1`` the function returns a single-process context
    without creating a process group, so the same training code runs on a laptop.

    Args:
        preferred_device: Explicit device for single-process runs (``cuda``, ``mps``, ``cpu``).

    Returns:
        The :class:`DistributedContext` of this process.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedContext(
            rank=0, local_rank=0, world_size=1, device=resolve_device(preferred_device)
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    dist.init_process_group(backend=backend, device_id=device if backend == "nccl" else None)
    logger.info("Initialised %s process group: rank %d/%d on %s", backend, rank, world_size, device)
    return DistributedContext(
        rank=rank, local_rank=local_rank, world_size=world_size, device=device
    )


def all_reduce_sum(
    values: Tensor,
    context: DistributedContext,
) -> Tensor:
    """Sum ``values`` across all ranks (identity when not distributed)."""
    if context.is_distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


def barrier(
    context: DistributedContext,
) -> None:
    """Synchronise all ranks (no-op when not distributed)."""
    if context.is_distributed:
        dist.barrier()


def shutdown_distributed(
    context: DistributedContext,
) -> None:
    """Destroy the process group if one was created."""
    if context.is_distributed and dist.is_initialized():
        dist.destroy_process_group()
