"""Unit tests for Euler, Heun, DDIM, and MSE BC samplers."""

import torch
import torch.nn as nn

from flow_matching_policy.loss import cosine_beta_schedule
from flow_matching_policy.model import DiT1D
from flow_matching_policy.sampler import (
    predict_mse_bc,
    sample_ddim,
    sample_flow_euler,
    sample_flow_heun,
)


class _CountingModel(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.calls = 0

    def forward(
        self, noisy_actions: torch.Tensor, timestep: torch.Tensor, obs_cond: torch.Tensor
    ) -> torch.Tensor:
        self.calls += 1
        return self.inner(noisy_actions, timestep, obs_cond)


def test_sampler_determinism_and_nfe() -> None:
    base = DiT1D(action_dim=2, pred_horizon=16, hidden_dim=32, depth=1, num_heads=2)
    obs_cond = torch.randn(2, 2, 32)
    shape = (2, 16, 2)

    # Euler determinism & NFE
    counter = _CountingModel(base)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    out1 = sample_flow_euler(counter, obs_cond, shape, num_steps=4, generator=g1)
    assert counter.calls == 4
    out2 = sample_flow_euler(counter, obs_cond, shape, num_steps=4, generator=g2)
    torch.testing.assert_close(out1, out2)
    assert out1.min() >= -1.0 and out1.max() <= 1.0

    # Heun NFE = 2 * num_steps
    counter.calls = 0
    out_heun = sample_flow_heun(
        counter, obs_cond, shape, num_steps=4, generator=torch.Generator().manual_seed(42)
    )
    assert counter.calls == 8
    assert out_heun.shape == shape

    # DDIM determinism & NFE
    betas = cosine_beta_schedule(timesteps=100)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    counter.calls = 0
    d1 = sample_ddim(
        counter,
        obs_cond,
        shape,
        alphas_cumprod,
        num_steps=8,
        generator=torch.Generator().manual_seed(7),
    )
    assert counter.calls == 8
    d2 = sample_ddim(
        counter,
        obs_cond,
        shape,
        alphas_cumprod,
        num_steps=8,
        generator=torch.Generator().manual_seed(7),
    )
    torch.testing.assert_close(d1, d2)

    # MSE BC NFE = 1
    counter.calls = 0
    bc = predict_mse_bc(counter, obs_cond, shape)
    assert counter.calls == 1
    assert bc.shape == shape
