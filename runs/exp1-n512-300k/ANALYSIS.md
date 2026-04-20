# exp1-n512-300k Analysis

**Date:** 2026-04-20
**Verdict:** RETRY — learning confirmed but slow; not at success threshold

## Numbers

| Checkpoint | eval ep_len (mean ± std) | eval ep_reward | training ep_len |
|---|---|---|---|
| 25k  | 19.40 ± 0.80 | -13.22 | ~22 |
| 75k  | 16.00 ± 0.00 | -11.96 | ~19 |
| 150k | 17.60 ± 0.49 | -11.81 | ~21 |
| 225k | 20.80 ± 0.40 | -11.92 | ~25 |
| 275k | 22.00 ± 0.00 | -11.55 | ~27 |
| 300k | **24.80 ± 0.40** | -11.66 | **30.8** |

Baseline (sanity-01 @ 50k): eval ep_len = 19.8

## Claims

- **C1 (PPO can learn):** Weak-positive. ep_len improved 19.8 → 24.8 (+25%). Still rising at 300k (training ep_len = 30.8). Needs more steps.
- **C3 (n_steps=512 beats 2048):** Weak-positive. 24.8 vs 19.8 at same 300k budget. Better than sanity-01's collapse to 19.8 at 50k.

## Observations

1. Training ep_len (30.8) >> eval ep_len (24.8): significant train/eval gap, possibly due to deterministic eval policy being overly conservative or fixed seed regularization.
2. Policy initially dropped (75k = 16.0) before recovering — transient exploration collapse typical of PPO.
3. Monotonic improvement from 200k→300k with no sign of plateau at termination → run needs more steps.
4. Wall time: 10638s (~2.96h) at 27 fps. Budget accurate.

## Implications for exp2

Policy is still improving at end of 300k. Two options:
1. Extend same config to 600k (C1 test — does more training help?)
2. Test n_steps=1024 at 300k (C3 head-to-head)
3. Try curriculum force_mag (C2 — start 1N→5N)

**Decision:** Test n_steps=1024 (exp2) to close out C3, then curriculum (exp3) since 5N may be the core barrier.
