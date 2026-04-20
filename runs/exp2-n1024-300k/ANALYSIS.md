# exp2-n1024-300k Analysis

**Date:** 2026-04-20
**Verdict:** FAIL — eval ep_len 18.2 < baseline 19.8; worse than exp1 (24.8)

## Numbers

| Checkpoint | eval ep_len (mean ± std) | eval ep_reward | training ep_len |
|---|---|---|---|
| 25k  | 28.60 ± 5.46 | -15.50 | ~24 |
| 75k  | 18.20 ± 0.40 | -12.37 | ~21 |
| 125k | 16.00 ± 0.00 | -12.24 | ~20 |
| 200k | 18.00 ± 0.00 | -12.10 | ~20 |
| 275k | 18.80 ± 0.40 | -11.67 | ~21 |
| 300k | **18.20 ± 0.40** | -11.65 | **21.2** |

Baseline: 19.8. Exp1 (n_steps=512): 24.8.

## Claims

- **C3 (n_steps=512 or 1024 beats 2048):** Positive for 512, negative for 1024.
  - n_steps=1024 WORSE than n_steps=512 (18.2 vs 24.8) — **C3 resolved: n_steps=512 is best among tested values.**
  - n_steps=1024 also worse than baseline sanity-01 (18.2 vs 19.8) at final eval.

## Observations

1. Peaked early at 28.6 @ 25k (high-variance exploration) then collapsed — hallmark of gradient noise with large rollout batches.
2. Training ep_len (21.2) ≈ eval ep_len (18.2) — smaller train/eval gap than exp1, suggesting policy is less overfit but also less learned.
3. With n_steps=1024, n_envs=4: ~73 PPO updates in 300k steps vs exp1's ~146. Fewer updates = slower learning here.
4. Wall time: 11337s (~3.15h) at 26 fps.

## Implications

- **C3 closed:** use n_steps=512 going forward.
- The core bottleneck is the 5N fixed force. Curriculum (C2) is the highest-priority next experiment.
- exp3 plan: phase A 150k at 1N (n_steps=512) → phase B load + 150k at 5N (n_steps=512).
