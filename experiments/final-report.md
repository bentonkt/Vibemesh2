# Overseer Experiment Arc — Final Report

**Window:** 2026-04-19 23:34 → 2026-04-20 14:06 ET (~14.5 hours)
**Goal:** PPO policy on `GraspEnv` (tomato soup can, 5 N disturbance) that holds the object for ≥ 400 / 500 eval steps.
**Starting baseline:** sanity-01, eval ep_length = 19.8 ± 1.2.
**Final best:** exp5-survival-bonus, eval ep_length = 100.2 ± 24.6 @ 275k; training rollout 127.7 still rising at 300k.

**Verdict:** Primary bottleneck identified and partially solved. Goal not yet reached (100 vs 400) but the trajectory is clearly upward and the root-cause change landed.

## Experiments

| # | ID | Steps | Key variable | Eval ep_len | Notes |
|--:|---|--:|---|--:|---|
| 0 | sanity-01           | 50k  | n_steps=2048, force=5N | 19.8 | baseline |
| 1 | exp1-n512-300k      | 300k | n_steps=512            | 24.8 | first signal of learning |
| 2 | exp2-n1024-300k     | 300k | n_steps=1024           | 18.2 | **C3 resolved:** 512 > 1024 > 2048 |
| 3a | exp3a-curr-1n      | 150k | force curriculum phase A (1 N) | 17.6 | 1 N ≈ 5 N — force isn't the bottleneck |
| 3b | exp3b-curr-5n      | 150k | force curriculum phase B (5 N, warm-start) | 25.8 | +1 over exp1 at same budget, marginal |
| 4 | exp4-no-force-300k  | 150k (partial) | force = 0 | 16-20 | **Key finding:** drop is policy-induced, not force-induced |
| 5 | exp5-survival-bonus | 300k | +0.1/step survival bonus | **100.2** | **Breakthrough at ~200k steps** |

## Claims

| Claim | Statement | Final evidence | Basis |
|---|---|---|---|
| C1 | PPO + 45D obs can learn the task | **positive** | exp5 eval 100.2 (5× baseline); training still rising |
| C2 | Force curriculum (low→high) helps | weak-positive, moot | exp3 showed +4% over fixed force. Irrelevant once reward shaping is fixed |
| C3 | n_steps=512 beats 2048 at fixed budget | **positive** | exp1 (24.8) > sanity-01 (19.8) > exp2 (18.2). Use 512 |
| C4 | Lower INTERP_STEPS_PER_KF increases fps w/o hurting stability | moderate-positive | all runs hit 26-42 fps with stable resets |
| C5 | Survival bonus unlocks learning | **positive** | eval ep_len breakthrough from ~25 to 100 in exp5 |

## Key findings

1. **Reward shaping was the primary bottleneck.** Prior to exp5 the reward structure (retention displacement + smoothness regularizer + terminal drop penalty) produced a local minimum where "don't move, accept the drop" had similar expected value to "try to hold, fail, drop faster." Adding a +0.1/step survival bonus shifted the expected return of holding from "slightly negative" to "strongly positive," which PPO could exploit once its value function caught up at ~200k steps.

2. **External force is not the bottleneck.** exp4 (force=0) produced the same eval ceiling as 1 N and 5 N runs. Drops originate in the policy's own joint actions destabilizing the grasp, not in the disturbance. This invalidates the assumption baked into C2 — a force curriculum gives nothing if the policy can't even hold a zero-force object.

3. **n_steps=512 strongly preferred over 2048 within 300k budget.** More frequent gradient updates (~146 vs 24) give PPO enough passes over the data to escape early plateaus. Don't use SB3's default n_steps=2048 for this kind of short-episode task.

4. **Reset speed is not a limiting factor.** All runs hit ~26-42 fps. The fps boost in exp5 (26 → 42) comes from longer episodes amortizing the keyframe-replay reset cost, not from changes to reset logic.

5. **Post-breakthrough instability.** exp5 eval variance spiked at 275k (std=24.6) — some episodes hit the full 500-step timeout, others drop at ~50. This is expected in the "sometimes learned" regime and will smooth out with more training, or by tuning retention weight down so long episodes aren't penalized into net-negative reward territory.

## What's broken / open

1. **Retention weight is too high at long horizons.** In exp5, eval reward *decreases* as ep_len increases, because retention penalty scales linearly with displacement × steps. The policy is holding the object but drifting slightly on every step. Current net reward at 100-step eps is roughly `+0.1 × 100 − 10 × 0.03 × 100 = −20`, making long episodes look worse than short successful grasps.

2. **Eval uses only 5 deterministic episodes.** Noise dominates the signal at the breakthrough point. Bump to 20 for runs that care about precise eval curves.

3. **Only one object.** All experiments used `005_tomato_soup_can`. Policy generalization across YCB/HOPE objects is untested.

4. **Final report was written by hand, not by the overseer agent.** The overseer died twice mid-loop (the first time from a bad prompt that treated "I'll check later" as end-of-turn; the second time after launching exp4, likely due to Sonnet's implicit max-turns limit on a 10-hour chained tool-call session). Training continued until my `kill-worker.sh` accidentally SIGHUP'd the exp4 subprocess. Fixed in this run by launching exp5 with `nohup setsid` so it was immune to any window kill.

## Recommended next steps

1. **Continue exp5 from `final.zip`** for another 300-500k steps with the same reward shape. Most direct path to eval ep_len ≥ 200.
2. **Tune `REWARD_RETENTION_SCALE` down** from 10.0 to 1.0-3.0 so long successful episodes stop accumulating into net-negative reward. Alternative: cap the displacement shaping with `max(disp - 0.01, 0)`.
3. **Bump eval to 20 deterministic episodes** to get honest mean ± std in the breakthrough regime.
4. **After the policy holds reliably at 5N**, add a second object (mustard bottle or mug) and evaluate whether the same policy generalizes. If not, retrain with randomized object sampling.
5. **Move experiments to a shared lab server** for the final 1M+ step training runs. Current host (WSL2 on shitter) is fine for short iterations but slow for sustained training; the 3090 would add 2-3× throughput primarily via more parallel envs.

## Artifacts

All commits are on `origin/main`:

- `72906ac` — experiments/ bootstrap + tunable PPO args
- `2909295` — exp1-n512-300k (eval 24.8)
- `0c199a5` — exp2-n1024-300k (eval 18.2, C3 resolved)
- `476c84e` — exp3-curriculum (eval 25.8)
- `5220fa8` — exp4-no-force partial + survival_bonus plumbing
- `<THIS COMMIT>` — exp5-survival-bonus (eval 100.2) + this final report

Training artifacts under `runs/<exp-id>/`: `stdout.log`, `ANALYSIS.md`, `evaluations.npz`, `final.zip`, `tb/`, `ckpts/`, `best/`. Checkpoints and TB events are git-ignored; the summary + analysis files are committed.
