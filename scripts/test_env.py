#!/usr/bin/env python3
"""Smoke test for GraspEnv: reset + 10 random steps + close."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.env import GraspEnv


def main() -> None:
    env = GraspEnv()
    obs, info = env.reset()
    assert obs.shape == (6,), f"bad obs shape: {obs.shape}"
    assert np.all(np.isfinite(obs)), f"non-finite obs: {obs}"
    print(f"reset OK: obs={obs} slip_mag={info['slip_mag']:.4f}")

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        z = float(env.data.xpos[env._obj_body, 2])
        print(
            f"step {i:2d}: r={reward:+.1f} term={terminated} "
            f"trunc={truncated} z={z:.3f} slip={info['slip_mag']:.4f}"
        )
        if terminated or truncated:
            print("episode ended early")
            break

    env.close()
    print("OK")


if __name__ == "__main__":
    main()
