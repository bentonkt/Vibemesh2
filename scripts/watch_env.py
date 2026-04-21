#!/usr/bin/env python3
"""Watch GraspEnv episodes in the MuJoCo viewer.

Usage (macOS requires mjpython):
    mjpython scripts/watch_env.py                        # hold ctrl (no noise)
    mjpython scripts/watch_env.py --explore               # smooth OU noise around hold
    mjpython scripts/watch_env.py --explore --sigma 0.5   # stronger exploration
    mjpython scripts/watch_env.py --random                # pure random (jerky)
    mjpython scripts/watch_env.py --force 20              # custom disturbance force
    mjpython scripts/watch_env.py --model runs/exp5-survival-bonus/final.zip
                                                          # trained PPO policy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.env import GraspEnv


class OUNoise:
    """Ornstein-Uhlenbeck process for smooth correlated exploration noise."""

    def __init__(
        self,
        size: int,
        mu: np.ndarray | float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.mu = np.full(size, mu) if isinstance(mu, (int, float)) else mu.copy()
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.rng = rng or np.random.default_rng()
        self.state = self.mu.copy()

    def reset(self) -> None:
        self.state = self.mu.copy()

    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + self.sigma * self.rng.standard_normal(self.size)
        self.state += dx
        return self.state.copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch GraspEnv in viewer")
    parser.add_argument("--random", action="store_true", help="Pure random actions (jerky)")
    parser.add_argument("--explore", action="store_true", help="Smooth OU noise around hold ctrl")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to a trained SB3 PPO .zip — use policy for actions")
    parser.add_argument("--deterministic", action="store_true",
                        help="Deterministic policy rollout (ignored without --model)")
    parser.add_argument("--force", type=float, default=5.0, help="Disturbance force magnitude (default: 5)")
    parser.add_argument("--show-forces", action="store_true",
                        help="Render an arrow on the object each step showing xfrc_applied")
    parser.add_argument("--arrow-scale", type=float, default=0.02,
                        help="Length multiplier on the force arrow (m per N; default 0.02)")
    parser.add_argument("--randomize", action="store_true",
                        help="Fresh random seed each reset — new disturbance trajectory per episode")
    parser.add_argument("--sigma", type=float, default=0.2, help="OU noise sigma (default: 0.2)")
    parser.add_argument("--theta", type=float, default=0.15, help="OU noise mean reversion rate (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    env = GraspEnv(force_mag=args.force)
    obs, info = env.reset(seed=args.seed)
    hold_ctrl = env.data.ctrl.copy().astype(np.float32)

    model = None
    if args.model:
        from stable_baselines3 import PPO
        model = PPO.load(args.model, device="cpu")

    ou_noise = OUNoise(
        size=env.model.nu,
        theta=args.theta,
        sigma=args.sigma,
        rng=np.random.default_rng(args.seed),
    )

    if model is not None:
        mode = f"policy ({args.model}, {'deterministic' if args.deterministic else 'stochastic'})"
    elif args.random:
        mode = "random actions (jerky)"
    elif args.explore:
        mode = f"OU exploration (sigma={args.sigma}, theta={args.theta})"
    else:
        mode = "hold ctrl"
    print(f"Watching: {mode}, force={args.force}N, seed={args.seed}")
    print("Close the viewer window to exit.")

    rng = np.random.default_rng(args.seed)

    with mujoco.viewer.launch_passive(
        model=env.model, data=env.data,
        show_left_ui=False, show_right_ui=False,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(env.model, viewer.cam)
        rate = RateLimiter(frequency=200.0, warn=False)
        step = 0
        episode = 0

        while viewer.is_running():
            if model is not None:
                action, _ = model.predict(obs, deterministic=args.deterministic)
            elif args.random:
                action = env.action_space.sample()
            elif args.explore:
                noise = ou_noise.sample().astype(np.float32)
                action = np.clip(hold_ctrl + noise, env.action_space.low, env.action_space.high)
            else:
                action = hold_ctrl

            obs, reward, terminated, truncated, info = env.step(action)
            step += 1

            if step % 50 == 0 or terminated:
                z = float(env.data.xpos[env._obj_body, 2])
                print(
                    f"ep {episode} step {step:4d}: r={reward:+.4f} slip={info['slip_mag']:.4f} "
                    f"ret={info['retention']:.4f} z={z:.3f}"
                )

            if terminated or truncated:
                outcome = "dropped" if terminated else "timeout (survived full ep)"
                print(f"Episode {episode} ended (step {step}, {outcome}). Resetting...")
                if args.randomize:
                    reset_seed = int(rng.integers(0, 2**31 - 1))
                else:
                    reset_seed = args.seed + episode + 1
                obs, info = env.reset(seed=reset_seed)
                hold_ctrl = env.data.ctrl.copy().astype(np.float32)
                ou_noise.reset()
                step = 0
                episode += 1

            # Draw a force arrow using user_scn (mjVIS_PERTFORCE only draws mouse
            # perturbations, not arbitrary xfrc_applied — so we render our own).
            if args.show_forces and hasattr(env, "last_applied_force"):
                viewer.user_scn.ngeom = 0
                f = env.last_applied_force
                mag = float(np.linalg.norm(f))
                if mag > 1e-6:
                    # xipos = body inertial frame (COM). For YCB meshes with
                    # inertiafromgeom, this is the can's visual center; xpos
                    # is the frame origin which can be several cm off the mesh.
                    start = env.data.xipos[env._obj_body].astype(np.float64)
                    end = start + f.astype(np.float64) * args.arrow_scale
                    geom = viewer.user_scn.geoms[0]
                    geom.rgba[:] = [1.0, 0.2, 0.2, 0.9]
                    mujoco.mjv_connector(
                        geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.004,
                        start, end,
                    )
                    viewer.user_scn.ngeom = 1

            viewer.sync()
            rate.sleep()

    env.close()


if __name__ == "__main__":
    main()
