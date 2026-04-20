# Experiments Log

Research goal: train PPO on GraspEnv (tomato soup can, 5 N disturbance) to hold object ≥ 400/500 steps on deterministic eval.

Baseline: sanity-01 — eval ep_length = 19.8 ± 1.2 @ 50k steps.

## exp1-n512-300k

**Start:** 2026-04-19
**Hypothesis:** C1 — PPO can learn given many more gradient updates.
**Design changes vs sanity-01:** n_steps 2048 → 512 (4× more updates/step), total 50k → 300k (6×). Everything else identical.
**Command:**
```
python3 scripts/train_ppo.py \
  --total-steps 300000 \
  --n-envs 4 \
  --run-name exp1-n512-300k \
  --force-mag 5.0 \
  --n-steps 512 \
  --batch-size 256 \
  --learning-rate 3e-4
```
**Budget:** ~3.6 hr wall time at 23 fps. Expected PPO updates: ~146 (vs sanity-01's 6).
**Success criterion:** eval ep_length ≥ 50 (≥2.5× baseline) → C1 gains moderate evidence.
**Retry criterion:** eval ep_length in 25–49 → promising, schedule a longer follow-up.
**Fail criterion:** eval ep_length < 25 → PPO not learning; pivot to curriculum or reward redesign.
