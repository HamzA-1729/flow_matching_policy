"""Action-chunk samplers for Flow Matching (Euler, Heun), DDIM (eta=0), and MSE BC.

``sample_flow_euler`` and ``sample_flow_heun`` are adapted from
``flow_matching_ddpm v0.1.0`` for 3D ``(B, H_pred, A_dim)`` action chunks
with pre-encoded observation conditioning ``obs_cond`` and identical final
``[-1, 1]`` action clamping across all policies.
"""

import torch
import torch.nn as nn


@torch.no_grad()
def sample_flow_euler(
    model: nn.Module,
    obs_cond: torch.Tensor,
    shape: tuple[int, ...],
    num_steps: int = 16,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """1st-order Euler ODE sampler (NFE = ``num_steps``)."""
    model.eval()
    device = obs_cond.device
    x = torch.randn(shape, device=device, generator=generator)
    dt = 1.0 / num_steps

    for step in range(num_steps):
        t = torch.full((shape[0],), step * dt, device=device, dtype=torch.float32)
        v = model(x, t, obs_cond)
        x = x + dt * v

    return torch.clamp(x, -1.0, 1.0)


@torch.no_grad()
def sample_flow_heun(
    model: nn.Module,
    obs_cond: torch.Tensor,
    shape: tuple[int, ...],
    num_steps: int = 8,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """2nd-order Heun predictor-corrector ODE sampler (NFE = ``2 * num_steps``)."""
    model.eval()
    device = obs_cond.device
    x = torch.randn(shape, device=device, generator=generator)
    dt = 1.0 / num_steps

    for step in range(num_steps):
        t_curr = torch.full((shape[0],), step * dt, device=device, dtype=torch.float32)
        t_next = torch.full(
            (shape[0],), min(1.0, (step + 1) * dt), device=device, dtype=torch.float32
        )
        v_curr = model(x, t_curr, obs_cond)
        x_pred = x + dt * v_curr
        v_next = model(x_pred, t_next, obs_cond)
        x = x + 0.5 * dt * (v_curr + v_next)

    return torch.clamp(x, -1.0, 1.0)


@torch.no_grad()
def sample_ddim(
    model: nn.Module,
    obs_cond: torch.Tensor,
    shape: tuple[int, ...],
    alphas_cumprod: torch.Tensor,
    num_steps: int = 16,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Deterministic DDIM (eta=0) sampler with per-step x0_hat clipping (NFE = ``num_steps``)."""
    model.eval()
    device = obs_cond.device
    total_timesteps = len(alphas_cumprod)
    step_indices = torch.linspace(0, total_timesteps - 1, num_steps, dtype=torch.long, device=device)

    x = torch.randn(shape, device=device, generator=generator)

    for i in reversed(range(num_steps)):
        curr_idx = step_indices[i]
        prev_idx = step_indices[i - 1] if i > 0 else None

        ab_curr = alphas_cumprod[curr_idx]
        ab_prev = (
            alphas_cumprod[prev_idx]
            if prev_idx is not None
            else torch.tensor(1.0, device=device, dtype=alphas_cumprod.dtype)
        )

        t_norm = torch.full(
            (shape[0],),
            float(curr_idx.item()) / max(total_timesteps - 1, 1),
            device=device,
            dtype=torch.float32,
        )
        eps_pred = model(x, t_norm, obs_cond)

        # Predict x0_hat and clip to [-1, 1] at every step
        x0_hat = (x - torch.sqrt(1.0 - ab_curr) * eps_pred) / torch.sqrt(ab_curr)
        x0_hat = torch.clamp(x0_hat, -1.0, 1.0)

        x = torch.sqrt(ab_prev) * x0_hat + torch.sqrt(1.0 - ab_prev) * eps_pred

    return torch.clamp(x, -1.0, 1.0)


@torch.no_grad()
def predict_mse_bc(
    model: nn.Module,
    obs_cond: torch.Tensor,
    shape: tuple[int, ...],
) -> torch.Tensor:
    """Single-pass deterministic MSE BC prediction (NFE = 1)."""
    model.eval()
    device = obs_cond.device
    placeholder = torch.zeros(shape, device=device, dtype=torch.float32)
    t_zero = torch.zeros((shape[0],), device=device, dtype=torch.float32)
    pred = model(placeholder, t_zero, obs_cond)
    return torch.clamp(pred, -1.0, 1.0)
