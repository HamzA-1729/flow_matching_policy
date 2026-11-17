"""1-batch overfit convergence tests for Flow Matching, DDPM, and MSE BC."""

import pytest
import torch

from flow_matching_policy.loss import DDPMLoss, FlowMatchingLoss, MSEBCLoss
from flow_matching_policy.model import DiT1D, StateObsEncoder
from flow_matching_policy.utils import set_seed


@pytest.mark.parametrize(
    ("mode", "min_drop"),
    [
        ("flow", 0.60),
        ("mse_bc", 0.95),
        ("ddpm", 0.75),
    ],
)
def test_one_batch_overfit(mode: str, min_drop: float) -> None:
    set_seed(42)
    obs_enc = StateObsEncoder(obs_dim=6, hidden_dim=64)
    model = DiT1D(action_dim=2, pred_horizon=16, hidden_dim=64, depth=2, num_heads=4, dropout=0.0)

    if mode == "flow":
        criterion = FlowMatchingLoss(model)
    elif mode == "ddpm":
        criterion = DDPMLoss(model, timesteps=100)
    else:
        criterion = MSEBCLoss(model)

    obs = torch.randn(4, 2, 6)
    actions = torch.tanh(torch.randn(4, 16, 2))
    optimizer = torch.optim.AdamW(list(obs_enc.parameters()) + list(model.parameters()), lr=3e-3)

    init_loss = None
    final_loss = 0.0
    for step in range(250):
        optimizer.zero_grad()
        obs_cond = obs_enc(obs)
        loss, _ = criterion(actions, obs_cond)
        if step == 0:
            init_loss = loss.item()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    assert init_loss is not None and init_loss > 0.0
    drop = (init_loss - final_loss) / init_loss
    assert drop >= min_drop, f"{mode} loss drop {drop:.3f} < {min_drop}"
