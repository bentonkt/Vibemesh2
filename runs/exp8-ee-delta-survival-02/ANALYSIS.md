# exp8-ee-delta-survival-02: ANALYSIS (Supplemental)

**Host**: 3090 (Windows, CPU)  
**Config**: EE-delta + survival_bonus=0.2, **n_steps=2048 (DEFAULT — unintentional)**, n_envs=4, 300k steps  
**Hypothesis**: survival_bonus=0.2 > 0.1 — stronger signal may help vs exp7

## Result: SLOW GRIND — eval ep_len = 30.2 @ 300k

| Eval Step | Eval ep_len | Eval ep_rew |
|-----------|-------------|-------------|
| 25k | 22.2 | -9.80 |
| 50k | 20.2 | -9.64 |
| 75k | 23.0 | -9.35 |
| 100k | 23.4 | -8.85 |
| 125k | 24.2 | -8.89 |
| 150k | 23.6 | -9.06 |
| 175k | 25.8 | -8.66 |
| 200k | 25.4 | -8.74 |
| 225k | 24.4 | -8.89 |
| 250k | 25.0 | -8.68 |
| 275k | 27.8 | -8.60 |
| **300k** | **30.2** | **-7.59** |

## Key Finding: n_steps=2048 is the dominant failure mode

Both my exp7 (sb=0.1) and this run (sb=0.2) stalled:
- exp7 (sb=0.1, n_steps=2048): eval 25 @ 300k
- exp8 (sb=0.2, n_steps=2048): eval 30 @ 300k
- **prior exp7 (sb=0.1, n_steps=512): eval 135 @ 300k (BREAKTHROUGH)**
- **prior exp8 (sb=0.5, n_steps=512): eval 416 @ 200k (GOAL)**

survival_bonus=0.2 is marginally (+5 ep_len) better than 0.1 with n_steps=2048, but neither can escape the negative-reward basin without frequent gradient updates.

## Verdict: FAIL — n_steps=2048 suppresses learning for this short-episode task

With 22-step mean episodes and n_steps=2048 per env, each rollout contains ~93 complete episodes 
(4 envs × 2048 steps / 22 ep_len). The value estimates are noisy/stale. n_steps=512 reduces this 
to ~23 episodes per rollout, enabling much tighter credit assignment. Always use n_steps=512.
