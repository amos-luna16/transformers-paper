import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from transformer_nmt.training.distributed import (
    DistributedContext,
    all_reduce_sum,
    barrier,
    shutdown_distributed,
)
from transformer_nmt.training.metrics import TokenLossAccumulator

WORLD_SIZE = 2


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _worker(
    rank: int,
    port: int,
) -> None:
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=WORLD_SIZE
    )
    context = DistributedContext(
        rank=rank, local_rank=rank, world_size=WORLD_SIZE, device=torch.device("cpu")
    )
    try:
        local = TokenLossAccumulator(
            loss_sum=2.0 * (rank + 1), token_count=rank + 1, correct_tokens=rank
        )
        reduced = TokenLossAccumulator.from_tensor(
            all_reduce_sum(local.as_tensor(context.device), context)
        )
        assert reduced.loss_sum == 6.0 and reduced.token_count == 3 and reduced.correct_tokens == 1
        assert abs(reduced.mean_loss - 2.0) < 1e-9
        flag = torch.tensor([1.0 if rank == 0 else 0.0], dtype=torch.float64)
        assert all_reduce_sum(flag, context).item() == 1.0  # rank-0 decisions reach every rank
        barrier(context)
        assert context.is_main_process == (rank == 0)
    finally:
        shutdown_distributed(context)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed not built")
def test_all_reduce_helpers_with_two_gloo_ranks():
    mp.spawn(_worker, args=(_free_port(),), nprocs=WORLD_SIZE, join=True)
