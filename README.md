# Flow Matching Policy: Continuous-Control & Visuomotor Action Generation

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.14+](https://img.shields.io/badge/PyTorch-2.14%2B-EE4C2C.svg)](https://pytorch.org/)
[![Package Manager: uv](https://img.shields.io/badge/managed%20by-uv-DE5FE9.svg)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*A clean, first-principles PyTorch implementation and comparative benchmark of **Optimal Transport Flow Matching Policy** against **DDPM/DDIM Diffusion Policy** and **Deterministic MSE Behavioral Cloning (BC)** on a bimodal 2D Push-T-style manipulation task (`obs_mode: state | image`) using a shared **1D Action Diffusion Transformer (`DiT1D`)** backbone.*

---

## 🌟 Core Capabilities & Key Highlights

* **Shared `DiT1D` Action-Chunking Backbone:** 1D sinusoidal action-step positional embeddings, AdaLN-Zero time conditioning, and cross-attention over pre-encoded observation history tokens (`obs_horizon=2`, `pred_horizon=16`, `action_horizon=8`), adapted from `flow_matching_ddpm v0.1.0`.
* **Deliberately Bimodal 2D Push-Block-to-Goal Environment:** Pusher starts on the goal side of the block (`(pusher - block) · (goal - block) > 0.15`), requiring a left or right detour around the block. Supports both `state` (6D) and `image` (64×64 NumPy RGB + 2D proprioception) modes.
* **Rigorous 3-Way Comparison:** Identical `DiT1D` capacity, optimizer, EMA weights (`decay=0.995`), and final `[-1, 1]` action clamping across **Flow Matching** (Euler/Heun), **DDPM/DDIM** (cosine schedule, $\bar{\alpha}_{T-1} < 10^{-3}$, $\eta=0$, per-step $\hat{x}_0$ clipping), and **MSE BC**.
* **Repeat-State Mode Entropy ($\bar{H}$):** Evaluates genuine multimodal sampling across $30$ fixed initial states $\times 10$ sampling seeds, distinguishing true stochastic mode coverage from deterministic drift.
* **Disjoint Validation & Test Protocol:** 100 validation rollout seeds for the smoke gate and tuning; 300 held-out test seeds across 3 training seeds (`{42, 1337, 2024}`) with per-seed 95% Wilson confidence intervals.

---

## 📐 Architecture: `DiT1D` Action-Chunking Policy

```text
Observation History (B, H_obs=2, O_dim)  [or 64x64 RGB + 2D Proprio]
       │
       ▼  Encoded ONCE per rollout step
[ StateObsEncoder / ImageObsEncoder ] ──► obs_cond: (B, 2, D)
                                                 │
Noisy Action Chunk (B, H_pred=16, 2)             │
       │                                         │
       ▼                                         ▼
┌────────────────────────────────────────────────────────┐
│ DiT1DBlock (Repeated 4x, hidden_dim=128, heads=4)      │
│   1. AdaLN-Zero Self-Attention (16 action tokens)      │
│   2. Cross-Attention (Q: actions, K/V: obs_cond)       │
│   3. AdaLN-Zero MLP                                    │
└────────────────────────────────────────────────────────┘
       │
       ▼  Final LayerNorm + Zero-Init Linear
Predicted Velocity / Noise / Action: (B, 16, 2)
```

---

## 📚 Mathematical Formulations

1. **Optimal Transport Flow Matching:**
   $$x_t = (1 - t) x_0 + t x_1, \quad u_t = x_1 - x_0, \quad \mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}\left[\|v_\theta(x_t, t, \text{obs}) - (x_1 - x_0)\|^2\right]$$
   *Note on NFE=1:* At $t=0$, $x_0 \perp x_1$, so 1 Euler step lands on $\mathbb{E}[x_1 \mid \text{obs}]$ (conditional mean collapse); multimodality recovers at $\text{NFE} \ge 2\text{--}4$.
2. **DDPM / DDIM ($\eta=0$):** Cosine $\bar{\alpha}_t$ schedule ($T=100$, $\bar{\alpha}_{T-1} \approx 2.4 \times 10^{-7} < 10^{-3}$) with per-step $\hat{x}_0 \in [-1, 1]$ clipping. (At $\text{NFE}=1$, $\hat{x}_0$ recovery amplifies $\epsilon$-error by $1/\sqrt{\bar{\alpha}_{T-1}} \approx 2000\times$.)
3. **Repeat-State Mode Entropy:** For $K=30$ states $\times$ $M=10$ seeds with mode counts $(N_{\text{left}, k}, N_{\text{right}, k})$:
   $$H(p_k) = -p_k \log_2 p_k - (1 - p_k) \log_2 (1 - p_k) \quad (\text{bits})$$

---

## 📊 Empirical Results & Diagnostics (`obs_mode: state`, 300 Held-Out Test Episodes × 3 Seeds `{42, 1337, 2024}`)

| Method | NFE | Success Rate (Mean ± Std) | Mode Entropy $\bar{H}$ (bits) | `none` Rate |
| :--- | :---: | :---: | :---: | :---: |
| **MSE BC (Deterministic)** | 1 | **`0.628 ± 0.044`** | `0.000 ± 0.000` | `0.000` |
| **Flow Matching (Euler)** | 1 | `0.381 ± 0.068` | `0.491 ± 0.065` | `0.003` |
| **Flow Matching (Euler)** | 2 | **`0.483 ± 0.034`** | `0.796 ± 0.009` | `0.000` |
| **Flow Matching (Euler)** | 4 | `0.417 ± 0.021` | `0.836 ± 0.010` | `0.000` |
| **Flow Matching (Euler)** | 8 | `0.429 ± 0.057` | `0.851 ± 0.002` | `0.000` |
| **Flow Matching (Euler)** | 16 | `0.416 ± 0.013` | **`0.852 ± 0.003`** | `0.000` |
| **DDPM / DDIM ($\eta=0$)** | 1 | `0.001 ± 0.002` | `0.473 ± 0.016` | `0.653` |
| **DDPM / DDIM ($\eta=0$)** | 2 | `0.002 ± 0.003` | `0.403 ± 0.111` | `0.188` |
| **DDPM / DDIM ($\eta=0$)** | 4 | `0.397 ± 0.023` | `0.766 ± 0.030` | `0.000` |
| **DDPM / DDIM ($\eta=0$)** | 8 | `0.362 ± 0.039` | `0.825 ± 0.006` | `0.000` |
| **DDPM / DDIM ($\eta=0$)** | 16 | `0.346 ± 0.052` | `0.851 ± 0.011` | `0.000` |

### Key Findings
* **Multimodal Coverage ($\bar{H}$):** MSE BC is strictly deterministic (`0.000` bits), whereas Flow Matching (`0.852` bits at NFE=16) and DDIM (`0.851` bits at NFE=16) capture both left and right detour modes from identical initial states.
* **Low-NFE Stability (Flow vs. DDIM):** At NFE=1–2, DDIM suffers severe $\hat{x}_0$ error amplification ($1/\sqrt{\bar{\alpha}_{T-1}} \approx 2000\times$), collapsing to `0.001–0.002` success (`none = 0.653 / 0.188`). Flow Matching remains stable at NFE=2 (`0.483` success, `0.796` bits, `0.000` `none` rate).
* **Receding-Horizon Symmetry Breaking:** Because `action_horizon=8` replans mid-episode, MSE BC breaks symmetry via slight initial drift and avoids compounding stochasticity, achieving higher raw success (`0.628`) at the cost of zero mode diversity (`0.000` bits).

---

## 📁 Repository Structure

```text
flow_matching_policy/
├── README.md
├── pyproject.toml
├── LICENSE
├── CONTRIBUTING.md
├── CITATION.cff
├── .pre-commit-config.yaml
├── configs/
│   ├── state_policy.yaml
│   └── image_policy.yaml
├── src/flow_matching_policy/
│   ├── __init__.py
│   ├── env.py                 # 2D PushBlockEnv (state + 64x64 NumPy RGB)
│   ├── dataset.py             # ScriptedExpert, ActionNormalizer, PolicyDataset
│   ├── model.py               # DiT1D, StateObsEncoder, ImageObsEncoder
│   ├── loss.py                # FlowMatchingLoss, DDPMLoss (cosine), MSEBCLoss
│   ├── sampler.py             # Euler, Heun, DDIM (eta=0), MSE BC
│   ├── utils.py               # set_seed, EMA, cosine warmup scheduler
│   ├── train.py               # Unified training pipeline
│   └── evaluate.py            # Rollout eval, Wilson CIs, repeat-state entropy, NFE sweep
└── tests/
    ├── test_env.py
    ├── test_dataset.py
    ├── test_model.py
    ├── test_loss.py
    ├── test_sampler.py
    └── test_overfit.py
```

---

## 🚀 Quickstart & Reproducibility

```bash
git clone https://github.com/HamzA-1729/flow_matching_policy.git
cd flow_matching_policy
uv sync

# 1. Run fast unit tests & 300-episode expert gate
uv run pytest -v -m "not slow"
uv run pytest -v -m slow

# 2. Train all 3 policies across 3 seeds (state mode)
for seed in 42 1337 2024; do
  uv run python -m flow_matching_policy.train --mode mse_bc --config configs/state_policy.yaml --seed $seed
  uv run python -m flow_matching_policy.train --mode flow --config configs/state_policy.yaml --seed $seed
  uv run python -m flow_matching_policy.train --mode ddpm --config configs/state_policy.yaml --seed $seed
done

# 3. Evaluate on validation (100 eps) and held-out test (300 eps + 30x10 entropy)
uv run python -m flow_matching_policy.evaluate --config configs/state_policy.yaml --split val --seeds 42 1337 2024
uv run python -m flow_matching_policy.evaluate --config configs/state_policy.yaml --split test --seeds 42 1337 2024
```

---

## 📚 References
* Lipman et al. (2023). *Flow Matching for Generative Modeling*. ICLR.
* Chi et al. (2023). *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*. RSS.
* Peebles & Xie (2023). *Scalable Diffusion Models with Transformers (DiT)*. ICCV.
* Song et al. (2021). *Denoising Diffusion Implicit Models (DDIM)*. ICLR.
* Nichol & Dhariwal (2021). *Improved Denoising Diffusion Probabilistic Models*. ICML.
