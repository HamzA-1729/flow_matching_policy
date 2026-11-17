"""1D Action Diffusion Transformer (DiT-1D) and observation encoders.

Adapted from ``flow_matching_ddpm v0.1.0`` AdaLN-Zero DiT design for 3D
action chunks ``(B, H_pred, A_dim)`` conditioned on observation tokens
``(B, H_obs, D)`` via cross-attention.
"""

import math
from typing import Protocol

import torch
import torch.nn as nn


class Backbone(Protocol):
    """Unified interface for action-chunking policy backbones."""

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        obs_cond: torch.Tensor,
    ) -> torch.Tensor: ...


def get_1d_sincos_pos_embed(seq_len: int, dim: int, device: torch.device) -> torch.Tensor:
    """Create 1D sinusoidal positional embeddings of shape ``(1, seq_len, dim)``."""
    pos = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=device) / max(half, 1)
    )
    args = pos * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(seq_len, 1, device=device)], dim=-1)
    return emb.unsqueeze(0)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for continuous ``t in [0, 1]``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / max(half, 1)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0) * 1000.0
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(t.unsqueeze(-1))], dim=-1)
        return self.mlp(emb)


class StateObsEncoder(nn.Module):
    """Encode state observation history ``(B, H_obs, obs_dim)`` to ``(B, H_obs, D)``."""

    def __init__(self, obs_dim: int = 6, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ImageObsEncoder(nn.Module):
    """Encode ``(B, H_obs, 3, H, W)`` RGB + ``(B, H_obs, 2)`` proprio to ``(B, H_obs, D)``."""

    def __init__(self, proprio_dim: int = 2, hidden_dim: int = 128) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            nn.Linear(64 + proprio_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        b, h_obs, c, h, w = image.shape
        feat = self.cnn(image.view(b * h_obs, c, h, w)).view(b, h_obs, 64)
        return self.proj(torch.cat([feat, proprio], dim=-1))


class DiT1DBlock(nn.Module):
    """AdaLN-Zero self-attention + observation cross-attention + AdaLN-Zero MLP."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_cross = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )
        # 6 modulation parameters: (gamma1, beta1, alpha1, gamma2, beta2, alpha2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor, obs_cond: torch.Tensor
    ) -> torch.Tensor:
        g1, b1, a1, g2, b2, a2 = self.adaLN_modulation(t_emb).unsqueeze(1).chunk(6, dim=-1)

        # 1. AdaLN-Zero Self-Attention
        h = self.norm1(x) * (1.0 + g1) + b1
        attn_out, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + a1 * attn_out

        # 2. Cross-Attention to observation tokens
        q = self.norm_cross(x)
        cross_out, _ = self.cross_attn(q, obs_cond, obs_cond, need_weights=False)
        x = x + cross_out

        # 3. AdaLN-Zero MLP
        h2 = self.norm2(x) * (1.0 + g2) + b2
        x = x + a2 * self.mlp(h2)
        return x


class DiT1D(nn.Module):
    """1D Action Diffusion Transformer for ``(B, H_pred, A_dim)`` action chunks."""

    def __init__(
        self,
        action_dim: int = 2,
        pred_horizon: int = 16,
        hidden_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        self.hidden_dim = hidden_dim

        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.time_embed = SinusoidalTimeEmbedding(hidden_dim)
        self.blocks = nn.ModuleList(
            [
                DiT1DBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.final_proj = nn.Linear(hidden_dim, action_dim)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        obs_cond: torch.Tensor,
    ) -> torch.Tensor:
        b, h, _ = noisy_actions.shape
        pos_emb = get_1d_sincos_pos_embed(h, self.hidden_dim, noisy_actions.device)
        x = self.action_proj(noisy_actions) + pos_emb
        t_emb = self.time_embed(timestep)

        for block in self.blocks:
            x = block(x, t_emb, obs_cond)

        return self.final_proj(self.final_norm(x))
