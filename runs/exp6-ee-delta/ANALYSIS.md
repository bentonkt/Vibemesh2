# exp6-ee-delta — ANALYSIS (both hosts)

**Design:** 22D EE-delta action space (per PDF item 6) + 3-term spec reward (no survival bonus). Same hyperparameters on both hosts. Two independent seeds — one on shitter (WSL, fork), one on 3090 (Windows, spawn).

## Numbers

| timesteps | shitter eval | 3090 eval |
|---:|---:|---:|
| 25k   | 22.4 ± 1.6 | 18.8 ± 0.7 |
| 50k   | 22.0 ± 2.4 | 20.0 ± 1.1 |
| 100k  | 21.6 ± 0.5 | 24.0 ± 0.9 |
| 150k  | 22.6 ± 1.2 | 23.8 ± 0.7 |
| 200k  | 25.2 ± 1.5 | 25.4 ± 0.5 |
| 250k  | 29.0 ± 0.6 | 29.0 ± 1.1 |
| 300k  | **37.2 ± 1.2** | **35.4 ± 1.5** |

Training ep_len @ 300k: 59.8 (shitter) / 46.8 (3090). Still rising on both.

Wall time: shitter 9250 s (2h34m), 3090 10945 s (3h02m). 3090 slower because Windows `spawn` start_method has more overhead than Linux `fork`. CPU-only on both — see separate JAX port branch for GPU path.

## Verdict

**RETRY** — eval more than doubled vs. pre-exp5 baseline (~18 → ~36) but **well short of exp5's 100.2** with the survival bonus on the old 23D action space. The spec-aligned action space helps (stochastic policy can't jerk arm joints anymore) but removing the survival bonus hurt more than the action-space fix gained.

Consistent curves across two independent seeds: both flat at ~22 until ~175k, then a steady climb. Same shape as exp1, but higher ceiling. Unlike exp5, no sharp breakthrough — just gradual improvement, still rising at 300k.

## Claim updates

- **C1** (PPO can learn): still positive (exp5 already confirmed). Both exp6 runs also show positive slope.
- **C5** (survival bonus): still strongly positive from exp5. Removing it in exp6 confirmed its importance.
- **New C6** (EE-delta action space helps): **weak-positive.** Baseline reward signal was good enough for +50% vs. exp1, but the survival bonus was doing more of the heavy lifting.

## Recommended next experiment

**exp7-ee-delta-survival:** keep EE-delta action space, re-enable `REWARD_ALIVE_BONUS=0.1`. Everything else identical to exp6. Hypothesis: action-space fix + reward shaping compose → eval > 100 within 300k, or even faster breakthrough than exp5 (which took 200k to break out).
