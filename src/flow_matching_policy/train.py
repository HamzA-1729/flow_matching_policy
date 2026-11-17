"""Unified training pipeline for Flow Matching, DDPM, and MSE BC policies."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from flow_matching_policy.dataset import (
    TRAIN_SEED_START,
    VAL_LOSS_SEED_START,
    ActionNormalizer,
    PolicyDataset,
    collect_expert_episodes,
)
from flow_matching_policy.env import EnvConfig
from flow_matching_policy.loss import DDPMLoss, FlowMatchingLoss, MSEBCLoss
from flow_matching_policy.model import DiT1D, ImageObsEncoder, StateObsEncoder
from flow_matching_policy.utils import EMA, get_cosine_schedule_with_warmup, set_seed


def _encode_batch(
    batch: dict[str, torch.Tensor],
    obs_enc: nn.Module,
    obs_mode: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    actions = batch["actions"].to(device)
    if obs_mode == "state":
        obs_cond = obs_enc(batch["obs"].to(device))
    else:
        obs_cond = obs_enc(batch["image"].to(device), batch["proprio"].to(device))
    return obs_cond, actions


@torch.no_grad()
def evaluate_val_loss(
    val_loader: DataLoader,
    obs_enc: nn.Module,
    criterion: nn.Module,
    obs_mode: str,
    device: torch.device,
) -> float:
    obs_enc.eval()
    criterion.eval()
    losses = []
    for batch in val_loader:
        obs_cond, actions = _encode_batch(batch, obs_enc, obs_mode, device)
        loss, _ = criterion(actions, obs_cond)
        losses.append(float(loss.item()))
    obs_enc.train()
    criterion.train()
    return float(np.mean(losses)) if losses else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Flow Matching, DDPM, or MSE BC policy.")
    parser.add_argument("--mode", choices=["flow", "ddpm", "mse_bc"], required=True)
    parser.add_argument("--config", type=str, default="configs/state_policy.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_mode = cfg["obs_mode"]

    env_cfg = EnvConfig(
        obs_mode=obs_mode,
        max_episode_steps=cfg["env"]["max_episode_steps"],
        max_step=cfg["env"]["max_step"],
        tau_lat=cfg["env"]["tau_lat"],
        image_size=cfg["env"]["image_size"],
    )

    train_eps = collect_expert_episodes(
        cfg["data"]["train_episodes"],
        TRAIN_SEED_START,
        env_cfg,
        action_noise_std=cfg["data"]["action_noise_std"],
    )
    val_eps = collect_expert_episodes(
        cfg["data"]["val_episodes"],
        VAL_LOSS_SEED_START,
        env_cfg,
        action_noise_std=cfg["data"]["action_noise_std"],
    )

    normalizer = ActionNormalizer()
    normalizer.fit(np.concatenate([e.actions for e in train_eps], axis=0))

    train_ds = PolicyDataset(
        train_eps,
        normalizer,
        obs_mode=obs_mode,
        obs_horizon=cfg["data"]["obs_horizon"],
        pred_horizon=cfg["data"]["pred_horizon"],
        random_shift_px=cfg["data"].get("random_shift_px", 0),
    )
    val_ds = PolicyDataset(
        val_eps,
        normalizer,
        obs_mode=obs_mode,
        obs_horizon=cfg["data"]["obs_horizon"],
        pred_horizon=cfg["data"]["pred_horizon"],
        random_shift_px=0,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    m_cfg = cfg["model"]
    obs_enc: nn.Module = (
        StateObsEncoder(obs_dim=6, hidden_dim=m_cfg["hidden_dim"])
        if obs_mode == "state"
        else ImageObsEncoder(proprio_dim=2, hidden_dim=m_cfg["hidden_dim"])
    ).to(device)

    model = DiT1D(
        action_dim=2,
        pred_horizon=cfg["data"]["pred_horizon"],
        hidden_dim=m_cfg["hidden_dim"],
        depth=m_cfg["depth"],
        num_heads=m_cfg["num_heads"],
        mlp_ratio=m_cfg["mlp_ratio"],
        dropout=m_cfg["dropout"],
    ).to(device)

    if args.mode == "flow":
        criterion: nn.Module = FlowMatchingLoss(model).to(device)
    elif args.mode == "ddpm":
        criterion = DDPMLoss(
            model, timesteps=cfg["ddpm"]["timesteps"], cosine_s=cfg["ddpm"]["cosine_s"]
        ).to(device)
    else:
        criterion = MSEBCLoss(model).to(device)

    params = list(obs_enc.parameters()) + list(model.parameters())
    optimizer = torch.optim.AdamW(
        params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    total_steps = cfg["train"]["total_steps"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps=cfg["train"]["warmup_steps"], total_steps=total_steps
    )
    ema_model = EMA(model, decay=cfg["train"]["ema_decay"])
    ema_enc = EMA(obs_enc, decay=cfg["train"]["ema_decay"])

    train_iter = iter(train_loader)
    pbar = tqdm(range(1, total_steps + 1), desc=f"{obs_mode}/{args.mode}/s{args.seed}")
    for step in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        optimizer.zero_grad()
        obs_cond, actions = _encode_batch(batch, obs_enc, obs_mode, device)
        loss, _ = criterion(actions, obs_cond)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg["train"]["grad_clip"])
        optimizer.step()
        scheduler.step()
        ema_model.update(model)
        ema_enc.update(obs_enc)

        if step % 500 == 0 or step == total_steps:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    val_loss = evaluate_val_loss(val_loader, obs_enc, criterion, obs_mode, device)
    ema_model.copy_to(model)
    ema_enc.copy_to(obs_enc)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{obs_mode}_{args.mode}_seed{args.seed}.pt"
    torch.save(
        {
            "mode": args.mode,
            "obs_mode": obs_mode,
            "seed": args.seed,
            "model_state_dict": model.state_dict(),
            "obs_enc_state_dict": obs_enc.state_dict(),
            "normalizer": normalizer.state_dict(),
            "config": cfg,
            "val_loss": val_loss,
        },
        ckpt_path,
    )
    print(f"Saved EMA checkpoint to {ckpt_path} (val_loss={val_loss:.4f})")


if __name__ == "__main__":
    main()
