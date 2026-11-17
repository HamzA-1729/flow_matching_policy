"""Unit tests for the 2D Push-Block-to-Goal environment."""

import numpy as np

from flow_matching_policy.env import EnvConfig, PushBlockEnv, classify_rollout_mode


def test_bimodal_start_invariant() -> None:
    env = PushBlockEnv(EnvConfig(obs_mode="state"))
    min_dist = env.cfg.pusher_radius + env.cfg.block_radius + 0.08
    for seed in range(50):
        obs = env.reset(seed=seed)
        assert obs.shape == (6,)
        pusher, block, goal = obs[:2], obs[2:4], obs[4:6]
        rel = pusher - block
        dist = float(np.linalg.norm(rel))
        assert min_dist <= dist <= 0.55
        assert float(np.dot(rel, goal - block)) > 0.15


def test_deterministic_reset() -> None:
    env = PushBlockEnv(EnvConfig(obs_mode="state"))
    obs_a = env.reset(seed=42)
    obs_b = env.reset(seed=42)
    np.testing.assert_allclose(obs_a, obs_b)


def test_quasistatic_contact() -> None:
    env = PushBlockEnv(EnvConfig(obs_mode="state"))
    env.reset(seed=7)
    # Place pusher directly above block and push down
    env.block = np.array([0.0, 0.0], dtype=np.float32)
    env.pusher = np.array([0.0, 0.18], dtype=np.float32)
    _, _, _, _ = env.step(np.array([0.0, -0.06], dtype=np.float32))
    # Distance should be at least pusher_radius + block_radius = 0.16
    dist = float(np.linalg.norm(env.block - env.pusher))
    assert dist >= env.cfg.pusher_radius + env.cfg.block_radius - 1e-5
    assert env.block[1] < 0.0


def test_image_obs_shape() -> None:
    env = PushBlockEnv(EnvConfig(obs_mode="image", image_size=64))
    obs = env.reset(seed=123)
    assert isinstance(obs, dict)
    assert obs["image"].shape == (64, 64, 3)
    assert obs["image"].dtype == np.uint8
    assert obs["proprio"].shape == (2,)
    assert obs["proprio"].dtype == np.float32


def test_classify_rollout_mode() -> None:
    goal = np.array([0.0, 0.55], dtype=np.float32)
    block_traj = np.zeros((4, 2), dtype=np.float32)

    # Left detour: crosses axial < 0 with lateral > 0.10 (since u_hat=(0,1), u_perp=(-1,0), so x < -0.10 is left)
    left_pusher = np.array([[0.0, 0.3], [-0.2, 0.1], [-0.2, -0.1], [0.0, -0.2]], dtype=np.float32)
    assert classify_rollout_mode(left_pusher, block_traj, goal, tau_lat=0.10) == "left"

    # Right detour: x > 0.10
    right_pusher = np.array([[0.0, 0.3], [0.2, 0.1], [0.2, -0.1], [0.0, -0.2]], dtype=np.float32)
    assert classify_rollout_mode(right_pusher, block_traj, goal, tau_lat=0.10) == "right"

    # Never gets behind block
    stuck_pusher = np.array([[0.0, 0.3], [0.0, 0.2], [0.0, 0.17], [0.0, 0.17]], dtype=np.float32)
    assert classify_rollout_mode(stuck_pusher, block_traj, goal, tau_lat=0.10) == "none"
