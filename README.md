# Autonomous Red-Team Agent via Deep Reinforcement Learning

> **Portfolio project** — Trains a PPO agent to discover multi-stage attack paths
> through a simulated enterprise network using Gymnasium + Stable-Baselines3.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────┐
│              RL Agent (PPO Policy)               │
│          2×256 MLP Actor + Critic heads          │
└────────────────────────┬─────────────────────────┘
                         │  Action (integer 0–35)
                         ▼
┌──────────────────────────────────────────────────────────────┐
│               RedTeamEnv (Gymnasium)                         │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐   │
│  │  NetworkX DiGraph — 9 nodes, 3 subnets, 2 firewalls   │   │
│  │  Each node: OS, CVEs, exploit/detect probabilities    │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                              │
│  Observation (36 floats):                                    │
│    [discovered | access_level | alerted | position_onehot]  │
│                                                              │
│  Action Space (Discrete 36):                                 │
│    SCAN(0–8) | EXPLOIT(9–17) | PRIV_ESC(18–26) | MOVE(27–35)│
│                                                              │
│  Reward:  Rt = R_access + R_stealth − C_step                 │
└──────────────────────────────────────────────────────────────┘
                         │  (obs, reward, done)
                         ▼
               Stable-Baselines3 PPO
               TensorBoard logging
```

---

## Network Topology

```
Internet
   │
  [0] dmz-webserver   ──── CVE-2021-41773 (Apache path traversal, 80%)
  [1] dmz-dns         ──── CVE-2020-8617  (BIND assertion, 40%)
   │
  [2] fw-dmz-corp     ──── CVE-2020-3452  (Cisco HTTPS, 35%)       ← Firewall 1
   │
  [3] corp-ws-1       ──── CVE-2017-0144  (EternalBlue SMB, 75%)
  [4] corp-ws-2       ──── CVE-2017-0144  (EternalBlue SMB, 65%)
  [5] corp-webapp     ──── CVE-2020-1938  (Ghostcat Tomcat, 55%)
   │
  [6] fw-corp-dc      ──── CVE-2020-2021  (PAN-OS auth bypass, 30%) ← Firewall 2
   │
  [7] dc-database     ──── CVE-2020-0618  (SQL Server RCE, 45%)
  [8] dc-active-dir   ──── CVE-2021-42278 (NoPac AD privesc, 55%)  ← TARGET ★
```

---

## MDP Specification

| Component | Details |
|---|---|
| **State space** | 36-dim float32 vector (4 arrays × 9 nodes) |
| **Action space** | Discrete(36) — 4 types × 9 targets |
| **Reward** | `Rt = R_access + R_stealth − C_step` |
| **R_capture_target** | +100 (ROOT on Active Directory) |
| **R_root_access** | +50 (ROOT on any node) |
| **R_user_access** | +10 (USER shell gained) |
| **R_new_discovery** | +5 (first scan of host) |
| **R_ids_alert** | −25 (IDS detection) |
| **C_step** | −0.1 (per-step time penalty) |
| **Episode length** | max 500 steps |
| **Win condition** | ROOT access on node 8 (AD) |

---

## Project Structure

```
RedTeamRL/
├── env/
│   ├── network_topology.py   # NetworkX graph + CVE metadata
│   └── network_env.py        # Gymnasium environment (MDP core)
├── agent/
│   └── trainer.py            # PPO training + custom callbacks
├── visualization/
│   └── plot_attack_path.py   # Network topology + attack path charts
├── logs/                     # TensorBoard + Monitor CSV output
├── models/                   # Saved model weights
│   └── checkpoints/          # Intermediate checkpoints
├── main.py                   # CLI entry point
├── requirements.txt
└── setup.ps1                 # Windows PowerShell setup script
```

---

## Quick Start (Windows / PowerShell)

### 1 — Setup

```powershell
cd "D:\Gand Fad Project\RedTeamRL"
.\setup.ps1
```

Or manually:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# CPU-only PyTorch first (avoids downloading 2 GB CUDA build)
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

### 2 — Sanity check (random policy, ~5 seconds)

```powershell
python main.py --mode demo
```

Expected output — environment steps correctly, prints per-step state.

### 3 — Train

```powershell
# Default: 500 k timesteps, 4 parallel envs
python main.py --mode train

# Shorter smoke test (to verify training loop runs)
python main.py --mode train --timesteps 50000 --envs 2
```

Training time on 8 GB RAM / CPU: ~30–60 min for 500 k steps.

### 4 — Monitor live (new terminal)

```powershell
.\.venv\Scripts\Activate.ps1
tensorboard --logdir logs
# → http://localhost:6006
```

### 5 — Evaluate trained model

```powershell
python main.py --mode evaluate --episodes 30
```

### 6 — Visualise attack path

```powershell
# Two-panel topology + attack path graphic
python main.py --mode visualize

# Training reward/length curves
python main.py --mode visualize --curves
```

---

## Key Concepts

### Why PPO?

- **Clipped objective** prevents destructively large policy updates —
  crucial when rewards are sparse and the agent takes thousands of
  random steps before finding any exploit path.
- **Entropy bonus** (`ent_coef=0.01`) encourages exploration of
  unfamiliar network segments rather than converging prematurely
  on the first exploit path discovered.

### Bellman Optimality

The agent's Q-function is trained toward:

```
Q*(s,a) = R(s,a) + γ · max_a' Q*(s', a')
```

With `γ=0.99` the agent values long-horizon plans —
essential for a 4-hop attack chain that may take 100+ steps.

### Action Masking (upgrade path)

`RedTeamEnv.action_masks()` returns a boolean mask of valid actions.
Swap `PPO` for `MaskablePPO` from `sb3-contrib` to eliminate
the invalid-action penalty and dramatically improve sample efficiency:

```python
from sb3_contrib import MaskablePPO
model = MaskablePPO("MlpPolicy", env, ...)
```

---

## Expected Training Behaviour

| Timesteps | Typical behaviour |
|---|---|
| 0–50 k | Random exploration, mostly invalid actions, negative rewards |
| 50–150 k | Learns SCAN → EXPLOIT chain for DMZ nodes |
| 150–300 k | Discovers Firewall-1 bypass, reaches Corporate subnet |
| 300–500 k | Full chain: DMZ → Corporate → Firewall-2 → AD root |

Compromise rate should reach **40–70 %** after 500 k steps on CPU.

---

## TensorBoard Metrics

| Metric | Meaning |
|---|---|
| `rollout/ep_rew_mean` | Mean episode reward (primary signal) |
| `rollout/ep_len_mean` | Mean steps to termination |
| `redteam/compromise_rate` | Rolling fraction of episodes that rooted AD |
| `redteam/nodes_discovered` | Avg recon coverage per episode |
| `redteam/nodes_owned` | Avg footprint size per episode |
| `redteam/alerts_triggered` | Avg IDS noise per episode |
| `train/entropy_loss` | Exploration entropy (should stay > 0) |
| `train/value_loss` | Critic accuracy |

---

## Skills Demonstrated (for CV / portfolio)

- Custom Gymnasium environment implementing a Markov Decision Process
- PPO deep reinforcement learning with Stable-Baselines3
- Graph-based attack simulation (NetworkX DiGraph)
- Stochastic exploit / IDS mechanics from real CVE data
- TensorBoard integration for training observability
- Action masking interface for improved sample efficiency
- Matplotlib dark-mode network topology visualisation
