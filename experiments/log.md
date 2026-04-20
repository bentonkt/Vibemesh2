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
**Result:** RETRY — eval ep_len = 24.80 ± 0.40 @ 300k. Training ep_len = 30.8 still rising. +25% over baseline. See runs/exp1-n512-300k/ANALYSIS.md.

## exp2-n1024-300k

**Start:** 2026-04-20
**Hypothesis:** C3 — n_steps=1024 may achieve better gradient estimates than 512 while still giving 2× more updates than sanity-01's 2048. Head-to-head comparison vs exp1.
**Design changes vs exp1:** n_steps 512 → 1024. Everything else identical.
**Command:**
```
python3 scripts/train_ppo.py \
  --total-steps 300000 \
  --n-envs 4 \
  --run-name exp2-n1024-300k \
  --force-mag 5.0 \
  --n-steps 1024 \
  --batch-size 256 \
  --learning-rate 3e-4
```
**Budget:** ~3.6 hr wall time at 27 fps. Expected PPO updates: ~73 (vs exp1's ~146).
**Success criterion:** eval ep_length ≥ 50 → C3 moderate evidence for 1024.
**Retry criterion:** eval ep_length in 25–49 → inconclusive vs exp1.
**Fail criterion:** eval ep_length < 25 → 512 better, C3 favors smaller n_steps.
**Result:** FAIL — eval ep_len = 18.2 ± 0.40 @ 300k. Worse than baseline AND exp1. Peaked 28.6 @ 25k then collapsed. C3 resolved: n_steps=512 wins. See runs/exp2-n1024-300k/ANALYSIS.md.

## exp3a-curr-1n (Curriculum Phase A)

**Start:** 2026-04-20
**Hypothesis:** C2 — training at low force first (1N) lets policy learn stable hold behavior; warm-start for 5N phase.
**Design changes vs exp1:** force_mag 5.0 → 1.0; total_steps 300k → 150k (phase A only). n_steps=512 (best from C3).
**Command:**
```
python3 scripts/train_ppo.py \
  --total-steps 150000 \
  --n-envs 4 \
  --run-name exp3a-curr-1n \
  --force-mag 1.0 \
  --n-steps 512 \
  --batch-size 256 \
  --learning-rate 3e-4
```
**Budget:** ~1.8 hr wall time. Expected PPO updates: ~73.
**Note:** Eval in this phase is at 1N. Phase B (exp3b) loads this model and trains/evals at 5N.

## exp3b-curr-5n (Curriculum Phase B)

**Start:** 2026-04-20 (after exp3a completes)
**Hypothesis:** C2 continued — warm-started policy from 1N transfers to 5N better than cold-start.
**Design changes vs exp1:** warm-start from exp3a; same 150k steps remaining.
**Command:**
```
python3 scripts/train_ppo.py \
  --total-steps 150000 \
  --n-envs 4 \
  --run-name exp3b-curr-5n \
  --force-mag 5.0 \
  --n-steps 512 \
  --batch-size 256 \
  --learning-rate 3e-4 \
  --load-model runs/exp3a-curr-1n/final.zip
```
**Budget:** ~1.8 hr wall time.
**Success criterion:** eval ep_length ≥ 50 → C2 positive evidence.
**Retry criterion:** eval ep_length 25–49 → modest curriculum benefit; consider longer phase B.
**Fail criterion:** eval ep_length < 25 → curriculum doesn't transfer; fixed-5N is equivalent or better.
