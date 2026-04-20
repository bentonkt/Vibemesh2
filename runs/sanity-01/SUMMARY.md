# Sanity Run: sanity-01

**Date:** 2026-04-19
**Command:** `python scripts/train_ppo.py --total-steps 50000 --n-envs 4 --run-name sanity-01`

## Numbers

| Metric | Value |
|---|---|
| total_timesteps | 50,000 |
| n_envs | 4 |
| force_mag | 5.0 N |
| mean_ep_length (last 100 eps, training buffer @ 50k) | 23.2 |
| mean_ep_reward (last 100 eps, training buffer @ 50k) | -14.57 |
| eval ep_length @ 50k (n=5, deterministic) | 19.80 ± 1.17 |
| eval ep_reward @ 50k (n=5, deterministic) | -13.23 ± 0.31 |
| eval ep_length @ 25k (n=5, deterministic) | 29.00 ± 4.56 |
| eval ep_reward @ 25k (n=5, deterministic) | -15.53 ± 1.83 |
| total wall time | 2557.3 s (42.6 min) |
| avg fps | ~23 |

## Eval ep_lengths per episode

| Checkpoint | Lengths |
|---|---|
| 25k | [32, 23, 32, 34, 24] |
| 50k | [21, 19, 18, 20, 21] |

## Finding: ep_len did NOT reach >100

**Diagnosis:**
- 50k steps with n_steps=2048, n_envs=4 yields only **6 PPO gradient updates** (50k / 8192 per batch ≈ 6)
- Each episode terminates in ~23 steps — random arm actions move the gripper far enough to drop the object within 0.12 s of sim time
- 5N disturbance force compounds this; even without external force, random joint control causes drop in ~22 steps
- The policy at 25k eval showed improvement (ep_len=29 vs baseline ~24) but degraded by 50k — classic underfitting/collapse with too few updates
- Reset bottleneck: 5-keyframe replay × 80 interp steps = 2100 mj_step/reset; with ep_len=23 and 4 envs, ~70% of wall time is in resets

**Training infrastructure is sound:**
- SubprocVecEnv(n_envs=4): works, no crashes
- TensorBoard logging: `runs/sanity-01/tb/`
- CheckpointCallback @ 50k: `runs/sanity-01/ckpts/`
- EvalCallback @ 25k + 50k: `runs/sanity-01/evaluations.npz`
- Final model: `runs/sanity-01/final.zip`

## Recommended next steps

1. **More steps**: 500k–1M steps are needed to see clear ep_len improvement against 5N forces
2. **Lower force_mag for curriculum**: start at 0.5N and ramp up as policy improves
3. **Reduce n_steps to 512**: 4× more gradient updates per 50k steps (24 vs 6), faster learning signal
4. **Reduce reset cost further**: drop INTERP_STEPS_PER_KF to 30–50; current z≈0.19 m is sufficient
