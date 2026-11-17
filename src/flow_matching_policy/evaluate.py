"""Closed-loop rollout evaluation, Wilson CIs, repeat-state entropy, and NFE sweeps."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from flow_matching_policy.dataset import (
    TEST_ROLLOUT_SEED_START,
    VAL_ROLLOUT_SEED_START,
    ActionNormalizer,
)
from flow_matching_policy.env import EnvConfig, ModeLabel, PushBlockEnv, classify_rollout_mode
from flow_matching_policy.loss import cosine_beta_schedule
from flow_matching_policy.model import DiT1D, ImageObsEncoder, StateObsEncoder
from flow_matching_policy.sampler import predict_mse_bc, sample_ddim, sample_flow_euler


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score confidence interval for a binomial proportion."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def binary_entropy_bits(left_count: int, right_count: int) -> float:
    """Plug-in binary entropy H(p) in bits."""
    total = left_count + right_count
    if total == 0:
        return 0.0
    p = left_count / total
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p))


@torch.no_grad()
def run_episode(
    env: PushBlockEnv,
    obs_enc: nn.Module,
    model: nn.Module,
    normalizer: ActionNormalizer,
    mode: str,
    obs_mode: str,
    obs_horizon: int,
    pred_horizon: int,
    action_horizon: int,
    nfe: int,
    ep_seed: int,
    sample_seed: int,
    alphas_cumprod: torch.Tensor,
    device: torch.device,
) -> tuple[bool, ModeLabel]:
    """Run one closed-loop receding-horizon episode."""
    obs = env.reset(seed=ep_seed)
    obs_hist: list[np.ndarray | dict[str, np.ndarray]] = [obs] * obs_horizon
    pusher_traj = [env.pusher.copy()]
    block_traj = [env.block.copy()]
    gen = torch.Generator(device=device).manual_seed(sample_seed)

    done = False
    info: dict = {"is_success": False}
    while not done:
        if obs_mode == "state":
            obs_t = torch.from_numpy(np.stack(obs_hist, axis=0)).unsqueeze(0).to(device)
            obs_cond = obs_enc(obs_t)
        else:
            imgs = np.stack([o["image"] for o in obs_hist], axis=0).astype(np.float32) / 255.0
            imgs = np.transpose(imgs, (0, 3, 1, 2))
            proprio = np.stack([o["proprio"] for o in obs_hist], axis=0)
            img_t = torch.from_numpy(imgs).unsqueeze(0).to(device)
            prop_t = torch.from_numpy(proprio).unsqueeze(0).to(device)
            obs_cond = obs_enc(img_t, prop_t)

        shape = (1, pred_horizon, 2)
        if mode == "flow":
            norm_chunk = sample_flow_euler(model, obs_cond, shape, num_steps=nfe, generator=gen)
        elif mode == "ddpm":
            norm_chunk = sample_ddim(
                model, obs_cond, shape, alphas_cumprod, num_steps=nfe, generator=gen
            )
        else:
            norm_chunk = predict_mse_bc(model, obs_cond, shape)

        raw_chunk = normalizer.denormalize(norm_chunk.squeeze(0).cpu().numpy())
        for act in raw_chunk[:action_horizon]:
            obs, _, done, info = env.step(act)
            obs_hist = obs_hist[1:] + [obs]
            pusher_traj.append(env.pusher.copy())
            block_traj.append(env.block.copy())
            if done:
                break

    mode_label = classify_rollout_mode(
        np.stack(pusher_traj, axis=0),
        np.stack(block_traj, axis=0),
        env.goal,
        tau_lat=env.cfg.tau_lat,
    )
    return bool(info["is_success"]), mode_label


def evaluate_checkpoint(
    ckpt_path: Path,
    seed_start: int,
    num_episodes: int,
    nfe: int,
    action_horizon: int,
    entropy_states: int,
    entropy_rollouts: int,
    device: torch.device,
) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    obs_mode = ckpt["obs_mode"]
    mode = ckpt["mode"]

    env_cfg = EnvConfig(
        obs_mode=obs_mode,
        max_episode_steps=cfg["env"]["max_episode_steps"],
        max_step=cfg["env"]["max_step"],
        tau_lat=cfg["env"]["tau_lat"],
        image_size=cfg["env"]["image_size"],
    )
    env = PushBlockEnv(env_cfg)
    normalizer = ActionNormalizer()
    normalizer.load_state_dict(ckpt["normalizer"])

    m_cfg = cfg["model"]
    obs_enc: nn.Module = (
        StateObsEncoder(obs_dim=6, hidden_dim=m_cfg["hidden_dim"])
        if obs_mode == "state"
        else ImageObsEncoder(proprio_dim=2, hidden_dim=m_cfg["hidden_dim"])
    ).to(device)
    obs_enc.load_state_dict(ckpt["obs_enc_state_dict"])
    obs_enc.eval()

    model = DiT1D(
        action_dim=2,
        pred_horizon=cfg["data"]["pred_horizon"],
        hidden_dim=m_cfg["hidden_dim"],
        depth=m_cfg["depth"],
        num_heads=m_cfg["num_heads"],
        mlp_ratio=m_cfg["mlp_ratio"],
        dropout=m_cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    betas = cosine_beta_schedule(
        timesteps=cfg["ddpm"]["timesteps"], s=cfg["ddpm"]["cosine_s"]
    ).to(device)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

    successes = 0
    none_count = 0
    for i in range(num_episodes):
        succ, m_label = run_episode(
            env,
            obs_enc,
            model,
            normalizer,
            mode,
            obs_mode,
            cfg["data"]["obs_horizon"],
            cfg["data"]["pred_horizon"],
            action_horizon,
            nfe,
            ep_seed=seed_start + i,
            sample_seed=50_000 + i,
            alphas_cumprod=alphas_cumprod,
            device=device,
        )
        successes += int(succ)
        none_count += int(m_label == "none")

    ci_low, ci_high = wilson_ci(successes, num_episodes)

    # Repeat-state mode entropy
    entropies = []
    for k in range(entropy_states):
        left_c, right_c = 0, 0
        for r in range(entropy_rollouts):
            _, m_label = run_episode(
                env,
                obs_enc,
                model,
                normalizer,
                mode,
                obs_mode,
                cfg["data"]["obs_horizon"],
                cfg["data"]["pred_horizon"],
                action_horizon,
                nfe,
                ep_seed=seed_start + k,
                sample_seed=90_000 + k * 100 + r,
                alphas_cumprod=alphas_cumprod,
                device=device,
            )
            if m_label == "left":
                left_c += 1
            elif m_label == "right":
                right_c += 1
        entropies.append(binary_entropy_bits(left_c, right_c))

    return {
        "seed": ckpt["seed"],
        "mode": mode,
        "nfe": nfe,
        "action_horizon": action_horizon,
        "success_rate": successes / num_episodes,
        "wilson_ci": [ci_low, ci_high],
        "none_rate": none_count / num_episodes,
        "mean_entropy_bits": float(np.mean(entropies)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained policies.")
    parser.add_argument("--config", type=str, default="configs/state_policy.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2024])
    parser.add_argument("--modes", nargs="+", default=["mse_bc", "flow", "ddpm"])
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--out-dir", type=str, default="outputs")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_mode = cfg["obs_mode"]
    seed_start = VAL_ROLLOUT_SEED_START if args.split == "val" else TEST_ROLLOUT_SEED_START
    num_eps = cfg["eval"]["val_episodes"] if args.split == "val" else cfg["eval"]["test_episodes"]

    results = []
    for mode in args.modes:
        nfes = [1] if mode == "mse_bc" else cfg["eval"]["nfe_sweep"]
        for nfe in nfes:
            seed_metrics = []
            for seed in args.seeds:
                ckpt_path = Path(args.ckpt_dir) / f"{obs_mode}_{mode}_seed{seed}.pt"
                if not ckpt_path.exists():
                    continue
                res = evaluate_checkpoint(
                    ckpt_path,
                    seed_start=seed_start,
                    num_episodes=num_eps,
                    nfe=nfe,
                    action_horizon=cfg["data"]["action_horizon"],
                    entropy_states=cfg["eval"]["entropy_states"],
                    entropy_rollouts=cfg["eval"]["entropy_rollouts_per_state"],
                    device=device,
                )
                seed_metrics.append(res)
                results.append(res)

            if seed_metrics:
                succs = [m["success_rate"] for m in seed_metrics]
                ents = [m["mean_entropy_bits"] for m in seed_metrics]
                nones = [m["none_rate"] for m in seed_metrics]
                print(
                    f"[{args.split}] {mode:6s} NFE={nfe:2d} | "
                    f"success={np.mean(succs):.3f}±{np.std(succs):.3f} | "
                    f"entropy={np.mean(ents):.3f}±{np.std(ents):.3f} bits | "
                    f"none={np.mean(nones):.3f}"
                )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"eval_{obs_mode}_{args.split}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation results to {out_file}")


if __name__ == "__main__":
    main()
