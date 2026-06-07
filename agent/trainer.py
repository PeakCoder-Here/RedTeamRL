"""
trainer.py
────────────────────────────────────────────────────────────────
PPO training pipeline for the autonomous red-team agent.

Algorithm choice:  PPO (Proximal Policy Optimization)
  - Handles discrete action spaces cleanly
  - More stable than vanilla policy gradient (clipped objective)
  - Works well on CPU for moderately sized environments

Architecture:  2 × 256 MLP (Actor + Critic heads share base)
Hyperparameters tuned for:
  - Sparse rewards with long episode horizons (~500 steps)
  - Exploration via entropy bonus (ent_coef=0.01)
  - Long credit assignment via high gamma (0.99)

TensorBoard:
  tensorboard --logdir logs/
"""

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sb3_contrib import MaskablePPO as PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from env.network_env import RedTeamEnv, AccessLevel
from env.network_topology import TARGET_NODE, NODE_REGISTRY

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent.parent
LOG_DIR   = BASE_DIR / "logs"
MODEL_DIR = BASE_DIR / "models"
CKPT_DIR  = MODEL_DIR / "checkpoints"

for d in [LOG_DIR, MODEL_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# Custom TensorBoard Callback
# ──────────────────────────────────────────────────────────────
class RedTeamMetricsCallback(BaseCallback):
    """
    Logs cybersecurity-specific episode metrics to TensorBoard:

      redteam/nodes_discovered   — recon coverage
      redteam/nodes_owned        — footprint size
      redteam/alerts_triggered   — IDS noise generated
      redteam/target_pwned       — 1 if AD was rooted, else 0
      redteam/compromise_rate    — rolling success rate across all episodes
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._total_episodes  = 0
        self._pwned_episodes  = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._total_episodes += 1
                pwned = int(info.get("target_access", 0) == AccessLevel.ROOT)
                self._pwned_episodes += pwned

                self.logger.record("redteam/nodes_discovered",
                                   info.get("nodes_discovered", 0))
                self.logger.record("redteam/nodes_owned",
                                   info.get("nodes_owned", 0))
                self.logger.record("redteam/alerts_triggered",
                                   info.get("alerts_triggered", 0))
                self.logger.record("redteam/target_pwned",   pwned)
                self.logger.record("redteam/compromise_rate",
                                   self._pwned_episodes / self._total_episodes)

        return True  # Never stop training early from this callback


# ──────────────────────────────────────────────────────────────
# Environment factory
# ──────────────────────────────────────────────────────────────
def _make_env(rank: int, seed: int):
    """Returns a factory closure for make_vec_env."""
    def _init():
        env = RedTeamEnv()
        env = Monitor(env, str(LOG_DIR / f"monitor_{rank}"))
        env.reset(seed=seed + rank)
        return env
    return _init


# ──────────────────────────────────────────────────────────────
# Training entry-point
# ──────────────────────────────────────────────────────────────
def train_ppo(
    total_timesteps: int = 500_000,
    n_envs:          int = 4,
    seed:            int = 42,
) -> PPO:
    """
    Train a PPO agent to compromise the simulated enterprise network.

    Args:
        total_timesteps: Environment steps to train for (default 500 k)
        n_envs:          Parallel environments (default 4)
        seed:            RNG seed for reproducibility

    Returns:
        Trained PPO model (also saved to models/redteam_ppo_final.zip)
    """
    print("=" * 65)
    print("  Autonomous Red-Team RL Agent  —  PPO Training")
    print("─" * 65)
    print(f"  Total timesteps : {total_timesteps:,}")
    print(f"  Parallel envs   : {n_envs}")
    print(f"  Random seed     : {seed}")
    print(f"  Target node     : [{TARGET_NODE}] {NODE_REGISTRY[TARGET_NODE].name}")
    print(f"  Win condition   : ROOT access on Active Directory")
    print("=" * 65)

    # ── Vectorized training environments ──────────────────────
    env = DummyVecEnv([_make_env(i, seed) for i in range(n_envs)])

    # ── Single eval environment ───────────────────────────────
    eval_env = Monitor(RedTeamEnv(), str(LOG_DIR / "eval_monitor"))

    # ── Callbacks ─────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(MODEL_DIR),
        log_path=str(LOG_DIR),
        eval_freq=max(10_000 // n_envs, 1),
        n_eval_episodes=20,
        deterministic=True,
        render=False,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1),
        save_path=str(CKPT_DIR),
        name_prefix="redteam_ppo",
        verbose=0,
    )

    metrics_callback = RedTeamMetricsCallback(verbose=0)

    # ── PPO Hyperparameters ────────────────────────────────────
    # n_steps × n_envs = steps collected before each policy update
    # For sparse rewards, larger batches → more stable gradient signal
    model = PPO(
        policy="MlpPolicy",
        env=env,

        # Core PPO
        learning_rate=3e-4,
        n_steps=2048,           # Rollout length per env
        batch_size=256,         # Mini-batch for SGD update
        n_epochs=10,            # PPO update iterations per rollout

        # Credit assignment
        gamma=0.99,             # High discount → values long-horizon plans
        gae_lambda=0.95,        # Generalised Advantage Estimation

        # PPO clip
        clip_range=0.15,
        clip_range_vf=None,

        # Entropy bonus — critical for exploring sparse-reward environments
        ent_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5,

        # Policy network: 2 hidden layers × 256 neurons each (Actor + Critic)
        # SB3 ≥ v1.8 uses dict directly (not wrapped in a list)
        policy_kwargs={
            "net_arch": dict(pi=[256, 256], vf=[256, 256]),
            "activation_fn": torch.nn.ReLU,
        },

        tensorboard_log=str(LOG_DIR),
        seed=seed,
        verbose=1,
    )

    # Print model summary
    obs_dim = model.observation_space.shape[0]
    print(f"\n  Obs dim  : {obs_dim}")
    print(f"  Actions  : {model.action_space.n}")
    print(f"  Device   : {model.device}")
    print(f"  Network  : MLP 2×256 (Actor) | 2×256 (Critic)\n")

    # ── Train ──────────────────────────────────────────────────
    t0 = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback, metrics_callback],
        tb_log_name="PPO_RedTeam",
        progress_bar=True,
        reset_num_timesteps=True,
    )
    elapsed = time.time() - t0

    print(f"\n✓ Training complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # ── Persist final model ────────────────────────────────────
    final_path = str(MODEL_DIR / "redteam_ppo_final")
    model.save(final_path)
    print(f"✓ Final model → {final_path}.zip")
    print(f"✓ Best model  → {MODEL_DIR / 'best_model.zip'}")
    print(f"\n  Launch TensorBoard:  tensorboard --logdir {LOG_DIR}")

    env.close()
    eval_env.close()
    return model


# ──────────────────────────────────────────────────────────────
# Quick evaluation helper (used by main.py --mode evaluate)
# ──────────────────────────────────────────────────────────────
def evaluate_model(
    model: PPO,
    n_episodes: int = 20,
    deterministic: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Run n_episodes with the given model and return aggregate statistics.
    """

    env = RedTeamEnv(render_mode="human" if verbose else None)
    rewards, steps, compromised = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done   = False

        while not done:
            action_masks = env.action_masks()
            action, _ = model.predict(obs, deterministic=deterministic,
                                      action_masks=action_masks)
            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated

        pwned = env.access_level[TARGET_NODE] == AccessLevel.ROOT
        rewards.append(env.total_reward)
        steps.append(env.step_count)
        compromised.append(int(pwned))

        if verbose:
            print(
                f"  Ep {ep+1:>3} │ reward={env.total_reward:>8.1f} │ "
                f"steps={env.step_count:>4} │ "
                f"pwned={'✓ DOMAIN ADMIN' if pwned else '✗'}"
            )

    env.close()

    stats = {
        "mean_reward":      float(np.mean(rewards)),
        "std_reward":       float(np.std(rewards)),
        "mean_steps":       float(np.mean(steps)),
        "compromise_rate":  float(np.mean(compromised)),
        "n_episodes":       n_episodes,
    }

    if verbose:
        print("\n" + "─" * 55)
        print(f"  Mean reward      : {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"  Mean steps       : {stats['mean_steps']:.1f}")
        print(f"  Compromise rate  : {stats['compromise_rate']*100:.1f}%")

    return stats
