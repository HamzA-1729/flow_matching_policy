"""Loss functions for Flow Matching, DDPM (cosine schedule), and MSE BC.

``FlowMatchingLoss`` is adapted from ``flow_matching_ddpm v0.1.0`` to support
arbitrary tensor shapes ``(B, H_pred, A_dim)`` via generic time broadcasting
and observation conditioning ``obs_cond``.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(timesteps: int = 100, s: float = 0.008) -> torch.Tensor:
    """Cosine variance schedule from Nichol & Dhariwal (2021).

    Guarantees ``alpha_bar[-1] < 1e-3`` for ``timesteps=100`` by clamping
    ``beta_t <= 0.999``.
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos(((t + s) / (1.0 + s)) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0, 0.999).float()


class FlowMatchingLoss(nn.Module):
    """Continuous-Time Optimal Transport Flow Matching Loss for action chunks."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, x_data: torch.Tensor, obs_cond: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        b = x_data.shape[0]
        t = torch.rand(b, device=x_data.device, dtype=x_data.dtype)
        x_noise = torch.randn_like(x_data)

        t_b = t.view(b, *([1] * (x_data.dim() - 1)))
        x_t = (1.0 - t_b) * x_noise + t_b * x_data
        target_velocity = x_data - x_noise

        pred_velocity = self.model(x_t, t, obs_cond)
        loss = F.mse_loss(pred_velocity, target_velocity)
        return loss, {"loss": float(loss.item())}


class DDPMLoss(nn.Module):
    """Discrete-time DDPM noise prediction loss with cosine beta schedule."""

    def __init__(self, model: nn.Module, timesteps: int = 100, cosine_s: float = 0.008) -> None:
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        betas = cosine_beta_schedule(timesteps=timesteps, s=cosine_s)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def forward(
        self, x_data: torch.Tensor, obs_cond: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        b = x_data.shape[0]
        t_idx = torch.randint(0, self.timesteps, (b,), device=x_data.device, dtype=torch.long)
        eps = torch.randn_like(x_data)

        view_shape = (b, *([1] * (x_data.dim() - 1)))
        sqrt_ab = self.sqrt_alphas_cumprod[t_idx].view(*view_shape)
        sqrt_omb = self.sqrt_one_minus_alphas_cumprod[t_idx].view(*view_shape)

        x_t = sqrt_ab * x_data + sqrt_omb * eps
        t_norm = t_idx.float() / max(self.timesteps - 1, 1)
        pred_eps = self.model(x_t, t_norm, obs_cond)
        loss = F.mse_loss(pred_eps, eps)
        return loss, {"loss": float(loss.item())}


class MSEBCLoss(nn.Module):
    """Capacity-matched deterministic MSE Behavioral Cloning loss."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, x_data: torch.Tensor, obs_cond: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        b = x_data.shape[0]
        placeholder = torch.zeros_like(x_data)
        t_zero = torch.zeros(b, device=x_data.device, dtype=x_data.dtype)
        pred_actions = self.model(placeholder, t_zero, obs_cond)
        loss = F.mse_loss(pred_actions, x_data)
        return loss, {"loss": float(loss.item())}
