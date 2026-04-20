#!/usr/bin/env python3
"""Extract and print a summary of a PPO training run from evaluations.npz."""

import sys
import numpy as np
from pathlib import Path


def summarize(run_dir: str) -> None:
    p = Path(run_dir)
    npz_path = p / "evaluations.npz"
    if not npz_path.exists():
        print(f"No evaluations.npz found in {run_dir}")
        return

    data = np.load(str(npz_path))
    timesteps = data["timesteps"]
    ep_lengths = data["ep_lengths"]  # shape: (n_evals, n_episodes)
    results = data["results"]        # shape: (n_evals, n_episodes)

    print(f"\n=== {p.name} ===")
    print(f"{'Timestep':>10}  {'mean_ep_len':>11}  {'std_ep_len':>10}  {'mean_rew':>9}  {'ep_lengths'}")
    print("-" * 80)
    for ts, lens, rews in zip(timesteps, ep_lengths, results):
        print(
            f"{int(ts):>10}  {np.mean(lens):>11.2f}  {np.std(lens):>10.2f}  "
            f"{np.mean(rews):>+9.2f}  {list(lens.astype(int))}"
        )

    best_idx = np.argmax(ep_lengths.mean(axis=1))
    print(f"\nBest eval: timestep={int(timesteps[best_idx])}, "
          f"mean_ep_len={ep_lengths[best_idx].mean():.2f}, "
          f"ep_lengths={list(ep_lengths[best_idx].astype(int))}")
    print(f"Final eval: timestep={int(timesteps[-1])}, "
          f"mean_ep_len={ep_lengths[-1].mean():.2f}, "
          f"ep_lengths={list(ep_lengths[-1].astype(int))}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/summarize_run.py runs/<run-name>")
        sys.exit(1)
    summarize(sys.argv[1])
