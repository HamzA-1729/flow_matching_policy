"""Unit tests for FlowMatchingLoss, DDPMLoss schedule bounds, and MSEBCLoss."""

import torch

from flow_matching_policy.loss import DDPMLoss, FlowMatchingLoss, MSEBCLoss
from flow_matching_policy.model import DiT1D


def test_flow_matching_loss_hand_computed() -> None:
    torch.manual_seed(0)
    # Identity-init DiT1D outputs exact zeros, so loss == mean((x_data - x_noise)^2)
    model = DiT1D(action_dim=2, pred_horizon=16, hidden_dim=32, depth=1, num_heads=2)
    fm_loss = FlowMatchingLoss(model)
    x_data = torch.ones(2, 16, 2)
    obs_cond = torch.zeros(2, 2, 32)

    loss, metrics = fm_loss(x_data, obs_cond)
    assert loss.ndim == 0
    assert loss.item() > 0.0
    assert "loss" in metrics


def test_ddpm_schedule_bounds() -> None:
    model = DiT1D(action_dim=2, pred_horizon=16, hidden_dim=32, depth=1, num_heads=2)
    ddpm = DDPMLoss(model, timesteps=100, cosine_s=0.008)

    assert ddpm.alphas_cumprod[0].item() > 0.99
    assert ddpm.alphas_cumprod[-1].item() < 1e-3

    x_data = torch.randn(4, 16, 2)
    obs_cond = torch.randn(4, 2, 32)
    loss, _ = ddpm(x_data, obs_cond)
    assert torch.isfinite(loss)


def test_mse_bc_loss() -> None:
    model = DiT1D(action_dim=2, pred_horizon=16, hidden_dim=32, depth=1, num_heads=2)
    bc_loss = MSEBCLoss(model)
    x_data = torch.ones(2, 16, 2) * 0.5
    obs_cond = torch.randn(2, 2, 32)
    loss, _ = bc_loss(x_data, obs_cond)
    # At init, model outputs 0 => MSE(0, 0.5) == 0.25
    assert abs(loss.item() - 0.25) < 1e-5
