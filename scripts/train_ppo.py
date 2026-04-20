#!/usr/bin/env python3
"""PPO training script for GraspEnv using Stable-Baselines3.

Usage:
    python scripts/train_ppo.py
    python scripts/train_ppo.py --total-steps 1000000 --n-envs 4 --run-name ppo-run-1
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor


def _make_env(seed: int, force_mag: float, survival_bonus: float = 0.0):
    """Factory for a single GraspEnv instance (must be importable for subprocess)."""
    def _init():
        from scripts.env import GraspEnv
        env = GraspEnv(force_mag=force_mag, survival_bonus=survival_bonus)
        env.reset(seed=seed)
        return env
    return _init


def make_vec_env(n_envs: int, seed: int = 0, force_mag: float = 5.0, survival_bonus: float = 0.0) -> SubprocVecEnv:
    import platform
    start_method = "fork" if platform.system() != "Windows" else "spawn"
    fns = [_make_env(seed + i, force_mag, survival_bonus) for i in range(n_envs)]
    return SubprocVecEnv(fns, start_method=start_method)


class ProgressCallback(BaseCallback):
    """Print one-line summary every log_interval steps."""

    def __init__(self, log_interval: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self._last_log = 0
        self._start_time = time.time()

    def _on_step(self) -> bool:
        ts = self.num_timesteps
        if ts - self._last_log >= self.log_interval:
            self._last_log = ts
            ep_info = self.model.ep_info_buffer
            if ep_info:
                mean_len = float(np.mean([e["l"] for e in ep_info]))
                mean_rew = float(np.mean([e["r"] for e in ep_info]))
            else:
                mean_len = mean_rew = float("nan")
            elapsed = time.time() - self._start_time
            fps = int(ts / elapsed) if elapsed > 0 else 0
            print(
                f"[{ts:>8d}] ep_len={mean_len:6.1f} ep_rew={mean_rew:+8.2f} fps={fps:5d}",
                flush=True,
            )
        return True


def parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description="PPO training for GraspEnv")
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--run-name", type=str, default=f"ppo-grasp-{ts}")
    parser.add_argument("--log-dir", type=str, default="runs/")
    parser.add_argument("--force-mag", type=float, default=5.0,
                        help="Disturbance force magnitude (0=disable)")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="PPO rollout length per env before update")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--eval-freq", type=int, default=25_000,
                        help="Total env steps between eval runs")
    parser.add_argument("--load-model", type=str, default=None,
                        help="Path to a .zip model to warm-start from (env is replaced)")
    parser.add_argument("--survival-bonus", type=float, default=0.0,
                        help="Per-step reward bonus for staying alive (0=disabled)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.log_dir) / args.run_name
    tb_dir = run_dir / "tb"
    ckpt_dir = run_dir / "ckpts"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run: {args.run_name}")
    print(f"Log dir: {run_dir}")
    print(f"Total steps: {args.total_steps:,}  n_envs: {args.n_envs}  force_mag: {args.force_mag}  survival_bonus: {args.survival_bonus}")

    # Training envs
    train_env = VecMonitor(make_vec_env(args.n_envs, seed=0, force_mag=args.force_mag, survival_bonus=args.survival_bonus))

    # Single eval env
    eval_env = VecMonitor(make_vec_env(1, seed=9999, force_mag=args.force_mag, survival_bonus=args.survival_bonus))

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=str(ckpt_dir),
        name_prefix="ppo_grasp",
        verbose=1,
    )
    eval_cb = EvalCallback(
        eval_env,
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        log_path=str(run_dir),
        best_model_save_path=str(run_dir / "best"),
        deterministic=True,
        verbose=1,
    )
    progress_cb = ProgressCallback(log_interval=10_000)

    if args.load_model:
        print(f"Loading model from {args.load_model}")
        model = PPO.load(
            args.load_model,
            env=train_env,
            tensorboard_log=str(tb_dir),
            device="cpu",
        )
        # Override hyperparams that may differ from the loaded model
        model.n_steps = args.n_steps
        model.batch_size = args.batch_size
        model.learning_rate = args.learning_rate
    else:
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            tensorboard_log=str(tb_dir),
            device="cpu",
            verbose=0,
        )

    start = time.time()
    model.learn(
        total_timesteps=args.total_steps,
        callback=[checkpoint_cb, eval_cb, progress_cb],
        tb_log_name=args.run_name,
        reset_num_timesteps=True,
    )
    elapsed = time.time() - start

    final_path = run_dir / "final.zip"
    model.save(str(final_path))
    print(f"\nFinal model saved to {final_path}")
    print(f"Total wall time: {elapsed:.1f}s")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
