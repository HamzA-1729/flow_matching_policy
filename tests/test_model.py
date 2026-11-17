"""Unit tests for DiT1D backbone and observation encoders."""

import torch

from flow_matching_policy.model import DiT1D, ImageObsEncoder, StateObsEncoder


def test_dit1d_shapes_and_identity_init() -> None:
    model = DiT1D(action_dim=2, pred_horizon=16, hidden_dim=128, depth=4, num_heads=4)
    obs_enc = StateObsEncoder(obs_dim=6, hidden_dim=128)

    obs = torch.randn(4, 2, 6)
    obs_cond = obs_enc(obs)
    assert obs_cond.shape == (4, 2, 128)

    noisy_actions = torch.randn(4, 16, 2)
    t = torch.rand(4)
    out = model(noisy_actions, t, obs_cond)
    assert out.shape == (4, 16, 2)
    # Zero-initialized final_proj + AdaLN-Zero gates => exact zero output at init
    torch.testing.assert_close(out, torch.zeros_like(out))


def test_image_obs_encoder_and_gradients() -> None:
    model = DiT1D(action_dim=2, pred_horizon=16, hidden_dim=128, depth=2, num_heads=4)
    img_enc = ImageObsEncoder(proprio_dim=2, hidden_dim=128)

    image = torch.rand(3, 2, 3, 64, 64)
    proprio = torch.randn(3, 2, 2)
    obs_cond = img_enc(image, proprio)
    assert obs_cond.shape == (3, 2, 128)

    noisy_actions = torch.randn(3, 16, 2)
    t = torch.rand(3)
    out = model(noisy_actions, t, obs_cond)
    loss = out.pow(2).mean()
    loss.backward()

    assert model.final_proj.weight.grad is not None
