# exp4-no-force-300k — ANALYSIS

**Status:** Partial (150k / 300k steps). Training process was terminated when the overseer's tmux window was killed; SIGHUP propagated to the background subprocess. No data beyond 150k.

## Design
Same as `exp1-n512-300k` but `force_mag=0.0` — eliminate external disturbance entirely to isolate whether the drop mechanism comes from the policy's own joint actions or from the applied force.

## Numbers (eval, deterministic, 1 eval ep per checkpoint)

| timesteps | mean ep_length | mean ep_reward |
|---:|---:|---:|
| 25,000  | 20 | -12.71 |
| 50,000  | 16 | -12.12 |
| 75,000  | 16 | -12.18 |
| 100,000 | 16 | -11.62 |
| 125,000 | 17 | -11.96 |
| 150,000 | 16 | -11.58 |

Training rollout ep_length tracked 19–20 steadily over 160k steps; reward improved mildly (-12.7 → -11.58) but ep_length did not.

## Verdict

**FAIL** (below baseline 19.8) but highly informative.

Even with zero external force, the policy can't extend grasp duration beyond ~16 eval steps. This confirms the `key_finding` from exp3: drops are driven by the policy's own joint actions, not the disturbance force. The current reward structure does not give a strong enough signal to *stay still* — the retention penalty scales with displacement (which is bounded by the drop threshold at ~0.05 m), producing only ~-0.5 per step in worst case, while the smoothness regularizer actively pushes the policy to make small actions. Together these reward a neutral "do nothing" policy only weakly.

## Implication for C2

C2 (force curriculum helps) is effectively moot: since 1 N and 5 N produce the same ceiling, and 0 N also produces ~20-step episodes, the policy's failure mode is not force-dependent.

## Recommended next experiment (exp5)

Add a **survival bonus** to the reward:
```
r_survive = +0.1 if not dropped else 0.0
```
At 0.1/step over a 500-step episode, max survival reward = +50, dominating the -10 drop penalty by 5×. This reshapes the reward landscape: the policy now has a clear incentive to *not act* when the grasp is stable, instead of only being penalized for displacement.

If exp5 shows clear learning (eval ep_len > 100), the survival bonus is validated and future work tunes the coefficient. If not, investigate policy architecture (action residuals, initialization).
