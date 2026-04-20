#!/usr/bin/env python3
"""MJX smoke test: random actions, n_envs parallel, print shapes and throughput.

Usage:
    python scripts/smoke_jax.py
    python scripts/smoke_jax.py --n-envs 64 --n-steps 200
    python scripts/smoke_jax.py --n-envs 4 --n-steps 50  # quick local test

Expected output (RTX 3090, n_envs=64):
    obs shape:     (64, 45)
    reward shape:  (64,)
    done shape:    (64,)
    throughput:    ~20000–60000 env-steps/sec
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def run_smoke_test(n_envs: int = 4, n_steps: int = 100, object_id: str = "005_tomato_soup_can"):
    import jax
    import jax.numpy as jnp

    print(f"JAX devices: {jax.devices()}")
    print(f"n_envs={n_envs}, n_steps={n_steps}, object={object_id}")

    # Phase 1: build init snapshot (CPU, ~30s with replay)
    print("\n[1/4] Building init snapshot (keyframe replay on CPU)...")
    t0 = time.time()
    from scripts.env_jax import GraspEnvMJX, build_init_snapshot
    snapshot = build_init_snapshot(object_id=object_id)
    print(f"      Done in {time.time() - t0:.1f}s")

    # Phase 2: build MJX env (converts model to GPU)
    print("[2/4] Building MJX env (uploading model to GPU)...")
    t0 = time.time()
    env = GraspEnvMJX(snapshot, object_id=object_id)
    print(f"      Done in {time.time() - t0:.1f}s")
    print(f"      obs_dim={env.obs_dim}, act_dim={env.act_dim}")

    # Phase 3: JIT-warm-up reset
    print("[3/4] JIT warm-up (reset + 1 step)...")
    t0 = time.time()
    rng = jax.random.PRNGKey(0)
    rng, reset_rng = jax.random.split(rng)
    batch_data, batch_obs = env.reset(n_envs, reset_rng)
    batch_obs.block_until_ready()

    ctrl_low = jnp.array(snapshot.ctrl_range[:, 0], dtype=jnp.float32)
    ctrl_high = jnp.array(snapshot.ctrl_range[:, 1], dtype=jnp.float32)

    rng, step_rng, action_rng = jax.random.split(rng, 3)
    rand_action = jax.random.uniform(
        action_rng, shape=(n_envs, env.act_dim),
        minval=ctrl_low, maxval=ctrl_high,
    )
    next_data, obs, reward, done, info = env.step(batch_data, rand_action, step_rng)
    obs.block_until_ready()
    print(f"      Done in {time.time() - t0:.1f}s")

    # Print shapes to verify correctness
    print("\n--- Shape check ---")
    print(f"obs shape:     {obs.shape}   (expected ({n_envs}, 45))")
    print(f"reward shape:  {reward.shape}  (expected ({n_envs},))")
    print(f"done shape:    {done.shape}   (expected ({n_envs},))")
    print(f"displacement:  min={float(jnp.min(info['displacement'])):.4f}  max={float(jnp.max(info['displacement'])):.4f}")
    print(f"slip_mag:      min={float(jnp.min(info['slip_mag'])):.4f}  max={float(jnp.max(info['slip_mag'])):.4f}")
    assert obs.shape == (n_envs, 45), f"obs shape mismatch: {obs.shape}"
    assert reward.shape == (n_envs,)
    assert done.shape == (n_envs,)
    print("All shape assertions passed ✓")

    # Phase 4: Throughput benchmark
    print(f"\n[4/4] Throughput benchmark ({n_steps} steps × {n_envs} envs)...")
    batch_data, _ = env.reset(n_envs, jax.random.PRNGKey(42))
    jax.block_until_ready(batch_data)  # warm start

    t_start = time.time()
    for i in range(n_steps):
        rng, step_rng, action_rng = jax.random.split(rng, 3)
        rand_action = jax.random.uniform(
            action_rng, shape=(n_envs, env.act_dim),
            minval=ctrl_low, maxval=ctrl_high,
        )
        batch_data, obs, reward, done, info = env.step(batch_data, rand_action, step_rng)

    # Block to ensure all GPU work is complete before timing
    obs.block_until_ready()
    t_elapsed = time.time() - t_start

    total_steps = n_steps * n_envs
    steps_per_sec = total_steps / t_elapsed
    print(f"\n--- Throughput ---")
    print(f"Total env-steps:   {total_steps:,}")
    print(f"Wall time:         {t_elapsed:.2f}s")
    print(f"Env-steps/sec:     {steps_per_sec:,.0f}")
    print(f"Target (pass):     >5,000 env-steps/sec")
    print(f"Status:            {'PASS ✓' if steps_per_sec > 5000 else 'FAIL ✗ — check GPU/CUDA setup'}")

    return steps_per_sec


def main():
    parser = argparse.ArgumentParser(description="MJX smoke test: random actions, throughput")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments (default: 4 for local test; use 64 on 3090)")
    parser.add_argument("--n-steps", type=int, default=100,
                        help="Number of steps per benchmark (default: 100)")
    parser.add_argument("--object", type=str, default="005_tomato_soup_can")
    args = parser.parse_args()

    steps_per_sec = run_smoke_test(n_envs=args.n_envs, n_steps=args.n_steps, object_id=args.object)
    print(f"\nFinal: {steps_per_sec:,.0f} env-steps/sec with n_envs={args.n_envs}")


if __name__ == "__main__":
    main()
