"""2D Push-Block-to-Goal environment with bimodal start geometry.

Supports both ``state`` (6D vector) and ``image`` (64x64 uint8 RGB + 2D proprioception)
observation modes without any external rendering dependencies.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

ObsMode = Literal["state", "image"]
ModeLabel = Literal["left", "right", "none"]


@dataclass
class EnvConfig:
    """Configuration for the 2D Push-Block-to-Goal environment."""

    obs_mode: ObsMode = "state"
    max_episode_steps: int = 120
    max_step: float = 0.06
    tau_lat: float = 0.10
    image_size: int = 64
    pusher_radius: float = 0.06
    block_radius: float = 0.10
    goal_radius: float = 0.08
    goal_xy: tuple[float, float] = (0.0, 0.55)


class PushBlockEnv:
    """Quasi-static 2D disk-pusher manipulation environment.

    The pusher starts on the goal side of the block:
        ``(pusher - block) · (goal - block) > 0.15``
    so pushing straight drives the block away from the goal. To succeed,
    the policy must detour either left or right around the block.
    """

    def __init__(self, config: EnvConfig | None = None) -> None:
        self.cfg = config or EnvConfig()
        self.goal = np.array(self.cfg.goal_xy, dtype=np.float32)
        self.pusher = np.zeros(2, dtype=np.float32)
        self.block = np.zeros(2, dtype=np.float32)
        self.step_count = 0
        self._rng = np.random.default_rng(0)

        # Precompute normalized pixel coordinate grid in [-1, 1]^2 for fast NumPy rendering
        s = self.cfg.image_size
        xs = np.linspace(-1.0, 1.0, s, dtype=np.float32)
        ys = np.linspace(1.0, -1.0, s, dtype=np.float32)  # top-down: +y is up
        self._grid_x, self._grid_y = np.meshgrid(xs, ys)

    def reset(self, seed: int | None = None) -> np.ndarray | dict[str, np.ndarray]:
        """Reset environment with a bimodal start pose."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.step_count = 0
        self.block = np.array(
            [
                self._rng.uniform(-0.15, 0.15),
                self._rng.uniform(-0.15, 0.10),
            ],
            dtype=np.float32,
        )

        min_dist = self.cfg.pusher_radius + self.cfg.block_radius + 0.08
        max_dist = 0.55
        goal_vec = self.goal - self.block

        for _ in range(1000):
            cand = self._rng.uniform(-0.80, 0.80, size=2).astype(np.float32)
            rel = cand - self.block
            dist = float(np.linalg.norm(rel))
            if min_dist <= dist <= max_dist and float(np.dot(rel, goal_vec)) > 0.15:
                self.pusher = cand
                return self.get_obs()

        raise RuntimeError("Failed to sample valid bimodal start pose within 1000 attempts.")

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray | dict[str, np.ndarray], float, bool, dict]:
        """Advance the environment by one quasi-static delta-position step."""
        act = np.asarray(action, dtype=np.float32)
        act = np.clip(act, -self.cfg.max_step, self.cfg.max_step)

        p_lim = 1.0 - self.cfg.pusher_radius
        self.pusher = np.clip(self.pusher + act, -p_lim, p_lim)

        # Quasi-static disk-disk contact resolution
        diff = self.block - self.pusher
        dist = float(np.linalg.norm(diff))
        contact_dist = self.cfg.pusher_radius + self.cfg.block_radius
        if dist < contact_dist:
            if dist > 1e-8:
                normal = diff / dist
            else:
                normal = np.array([0.0, -1.0], dtype=np.float32)
            self.block = self.pusher + normal * contact_dist
            b_lim = 1.0 - self.cfg.block_radius
            self.block = np.clip(self.block, -b_lim, b_lim)

        self.step_count += 1
        goal_dist = float(np.linalg.norm(self.block - self.goal))
        success = goal_dist <= self.cfg.goal_radius
        done = success or (self.step_count >= self.cfg.max_episode_steps)
        reward = 1.0 if success else -goal_dist

        info = {
            "is_success": success,
            "goal_dist": goal_dist,
            "step_count": self.step_count,
        }
        return self.get_obs(), reward, done, info

    def get_state(self) -> np.ndarray:
        """Return 6D state vector [pusher_x, pusher_y, block_x, block_y, goal_x, goal_y]."""
        return np.concatenate([self.pusher, self.block, self.goal]).astype(np.float32)

    def render(self) -> np.ndarray:
        """Render top-down 64x64x3 uint8 RGB frame purely in NumPy."""
        s = self.cfg.image_size
        img = np.full((s, s, 3), 240, dtype=np.uint8)

        # Goal region (soft green)
        g_mask = (self._grid_x - self.goal[0]) ** 2 + (
            self._grid_y - self.goal[1]
        ) ** 2 <= self.cfg.goal_radius**2
        img[g_mask] = np.array([80, 200, 120], dtype=np.uint8)

        # Block (steel blue)
        b_mask = (self._grid_x - self.block[0]) ** 2 + (
            self._grid_y - self.block[1]
        ) ** 2 <= self.cfg.block_radius**2
        img[b_mask] = np.array([50, 110, 210], dtype=np.uint8)

        # Pusher (crimson)
        p_mask = (self._grid_x - self.pusher[0]) ** 2 + (
            self._grid_y - self.pusher[1]
        ) ** 2 <= self.cfg.pusher_radius**2
        img[p_mask] = np.array([220, 70, 70], dtype=np.uint8)

        return img

    def get_obs(self) -> np.ndarray | dict[str, np.ndarray]:
        """Return observation according to ``cfg.obs_mode``."""
        if self.cfg.obs_mode == "state":
            return self.get_state()
        return {
            "image": self.render(),
            "proprio": self.pusher.copy(),
        }


def classify_rollout_mode(
    pusher_traj: np.ndarray,
    block_traj: np.ndarray,
    goal: np.ndarray,
    tau_lat: float = 0.10,
) -> ModeLabel:
    """Classify a rollout as ``'left'``, ``'right'``, or ``'none'``.

    Uses the initial block-to-goal axis direction ``u_hat`` and measures the
    pusher's relative position against the *current* block position at each step.
    """
    init_vec = goal - block_traj[0]
    norm = float(np.linalg.norm(init_vec))
    if norm < 1e-8:
        return "none"
    u_hat = init_vec / norm
    u_perp = np.array([-u_hat[1], u_hat[0]], dtype=np.float32)

    for p_t, b_t in zip(pusher_traj, block_traj, strict=True):
        rel = p_t - b_t
        axial = float(np.dot(rel, u_hat))
        if axial < 0.0:
            lateral = float(np.dot(rel, u_perp))
            if lateral > tau_lat:
                return "left"
            if lateral < -tau_lat:
                return "right"
            return "none"

    return "none"
