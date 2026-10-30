import torch

from transformer_nmt.training.precision import (
    autocast_context,
    create_grad_scaler,
    resolve_precision,
)


def test_cpu_falls_back_to_fp32():
    cpu = torch.device("cpu")
    for requested in ("auto", "bf16", "fp16", "fp32"):
        policy = resolve_precision(requested, cpu)
        assert policy.name == "fp32" and not policy.enabled and not policy.use_grad_scaler
        assert create_grad_scaler(policy, cpu) is None


def test_autocast_context_is_noop_for_fp32():
    policy = resolve_precision("fp32", torch.device("cpu"))
    with autocast_context(policy, torch.device("cpu")):
        assert (torch.ones(2) @ torch.ones(2)).dtype == torch.float32
