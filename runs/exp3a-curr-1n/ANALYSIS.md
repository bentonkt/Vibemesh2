# exp3a-curr-1n Analysis (Curriculum Phase A)

**Date:** 2026-04-20
**Verdict:** Informative — 1N force behaves identically to 5N; drop bottleneck is not force-dominated

## Numbers

| Checkpoint | eval ep_len (mean ± std) | eval ep_reward |
|---|---|---|
| 25k  | 20.20 ± 0.40 | -12.88 |
| 75k  | 16.00 ± 0.00 | -11.94 |
| 150k | **17.60 ± 0.49** | -11.71 |

Training ep_len @ 150k: 21.0

## Key Finding

1N disturbance force produces the same ~17-20 ep_len as 5N. The object is being dropped due to random/learned joint actions moving the hand off the grasp posture — NOT primarily due to external disturbance. Force magnitude is not the core bottleneck.

## Implication for C2

Phase A warm-start model has ep_len ~17.6 at 1N eval — not obviously better than cold-start. The value of curriculum must come from the phase B adaptation, if any.
