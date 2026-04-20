# exp3b-curr-5n Analysis (Curriculum Phase B — final eval)

**Date:** 2026-04-20
**Verdict:** RETRY — eval ep_len 25.80 > exp1 (24.80), modest curriculum benefit; training still rising

## Numbers

| Checkpoint (phase B steps) | eval ep_len (mean ± std) | eval ep_reward | training ep_len |
|---|---|---|---|
| 25k  | 19.00 ± 0.00 | -12.03 | ~21 |
| 75k  | 20.80 ± 0.40 | -11.99 | ~23 |
| 125k | 22.00 ± 0.00 | -11.84 | ~28 |
| 150k | **25.80 ± 0.40** | -11.97 | **34.1** |

Total steps: 150k (1N) + 150k (5N) = 300k. Compare to exp1 (300k at 5N direct): 24.80.

## Claims

- **C2 (curriculum beats fixed-force):** Weak-positive. 25.80 vs exp1's 24.80 at same total budget (+1 step, +4%). Not dramatic. Phase A (1N) performed identically to 5N, so the benefit is modest.

## Observations

1. Eval cluster very tight: [25, 26, 26, 26, 26] — deterministic policy hits consistent barrier around step 25-26.
2. Training ep_len (34.1) >> eval ep_len (25.8) — same large train/eval gap as exp1.
3. Training curve still rising at termination (34.1 at 150k, up from 28.8 at 130k).
4. The ~25-step barrier is structural: the deterministic policy outputs a fixed ctrl trajectory that exhausts grasp stability at step ~25. Likely joint positions drifting into an unstable configuration over ~25 steps.

## Implications for exp4

The primary bottleneck is NOT force magnitude. The policy needs either:
1. A stronger gradient signal to hold (survival bonus reward) — give explicit positive reward for each step alive
2. Force_mag=0 diagnostic to confirm whether any force component contributes to failure
Decision: exp4 = force_mag=0 diagnostic (300k, n_steps=512). Exp5 = reward modification (survival bonus).
