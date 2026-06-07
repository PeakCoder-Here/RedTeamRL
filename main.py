"""
main.py
────────────────────────────────────────────────────────────────
CLI entry point for the Autonomous Red-Team RL Agent.

Modes
─────
  demo        Quick 3-episode sanity check with random policy
  train       PPO training run (default: 500 k timesteps)
  evaluate    Load saved model and print per-episode stats
  visualize   Render attack-path graph or training curves

Examples
────────
  python main.py --mode demo
  python main.py --mode train --timesteps 500000 --envs 4
  python main.py --mode evaluate --model models/best_model --episodes 20
  python main.py --mode visualize --model models/best_model
  python main.py --mode visualize --curves
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent


# ──────────────────────────────────────────────────────────────
# Mode handlers
# ──────────────────────────────────────────────────────────────

def mode_demo(_args):
    """
    Run 3 episodes with a completely random policy.
    Verifies the environment resets, steps, and terminates correctly
    before wasting GPU/CPU time on training.
    """
    from env.network_env import RedTeamEnv, AccessLevel
    from env.network_topology import TARGET_NODE

    print("─" * 60)
    print("  DEMO — Random Policy (environment sanity check)")
    print("─" * 60)

    env = RedTeamEnv(render_mode="human")

    for ep in range(3):
        obs, info = env.reset()
        done = False
        print(f"\n▶ Episode {ep + 1}")

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        pwned = env.access_level[TARGET_NODE] == AccessLevel.ROOT
        print(
            f"\n  Final — steps={env.step_count} | "
            f"reward={env.total_reward:.1f} | "
            f"owned={int((env.access_level > 0).sum())}/9 | "
            f"target={'✓ PWNED' if pwned else '✗'}"
        )

    env.close()
    print("\n✓ Environment OK.  Run:  python main.py --mode train")


def mode_train(args):
    """Launch the PPO training pipeline."""
    from agent.trainer import train_ppo

    train_ppo(
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        seed=args.seed,
    )


def mode_evaluate(args):
    """Load a saved model and evaluate over N episodes."""
    from sb3_contrib import MaskablePPO
    from agent.trainer import evaluate_model

    model_path = args.model or str(BASE_DIR / "models" / "best_model")
    print(f"Loading: {model_path}")

    model = MaskablePPO.load(model_path)
    evaluate_model(model, n_episodes=args.episodes, deterministic=True)


def mode_visualize(args):
    """Render training curves or single-episode attack-path graph."""
    from visualization.plot_attack_path import (
        plot_training_curves,
        run_episode_with_model,
        plot_network_and_attack_path,
    )

    if args.curves:
        plot_training_curves()
        return

    from sb3_contrib import MaskablePPO

    model_path = args.model or str(BASE_DIR / "models" / "best_model")
    print(f"Loading: {model_path}")
    model = MaskablePPO.load(model_path)

    print("Running evaluation episode …")
    result = run_episode_with_model(model, deterministic=True)

    save_path = str(BASE_DIR / "logs" / "attack_path.png")
    plot_network_and_attack_path(result, save_path=save_path)


# ──────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="RedTeamRL",
        description="Autonomous Red-Team Reinforcement Learning Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--mode",
        choices=["demo", "train", "evaluate", "visualize"],
        default="demo",
        help="Execution mode (default: demo)",
    )

    # Training args
    p.add_argument("--timesteps", type=int, default=500_000,
                   help="Total PPO training timesteps (default: 500000)")
    p.add_argument("--envs",      type=int, default=4,
                   help="Parallel environments for training (default: 4)")
    p.add_argument("--seed",      type=int, default=42,
                   help="RNG seed (default: 42)")

    # Evaluation args
    p.add_argument("--model",    type=str, default=None,
                   help="Path to saved model (no .zip extension)")
    p.add_argument("--episodes", type=int, default=20,
                   help="Number of evaluation episodes (default: 20)")

    # Visualisation args
    p.add_argument("--curves", action="store_true",
                   help="Plot training curves instead of attack path")

    return p


# ──────────────────────────────────────────────────────────────
# Entry
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "demo":      mode_demo,
        "train":     mode_train,
        "evaluate":  mode_evaluate,
        "visualize": mode_visualize,
    }
    dispatch[args.mode](args)
