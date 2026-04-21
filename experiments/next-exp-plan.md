# Next Experiment Plans (Arc-2, prepared 2026-04-21)

## Current Status
- exp7 (shitter): survival_bonus=0.1, stalled eval ~25, still training ~120k/300k
- exp8 (3090): survival_bonus=0.5, BREAKTHROUGH — eval 50 @ 100k, training ep_len 74.7 @ 120k

## Key Insight
survival_bonus=0.5 (not 0.1) is the unlock. 5× stronger survival signal drives positive reward
by 70k steps → strong learning gradient → exponential ep_len growth.

## exp9 — shitter (launch immediately when exp7 finishes ~25min)
**Hypothesis**: survival_bonus=0.5 replicates exp8 breakthrough on Linux/shitter (CPU)
**Command**:
```
cd /tmp/claude-worktrees/vibemesh-overseer-arc2 && \
nohup setsid python scripts/train_ppo.py \
  --total-steps 300000 \
  --n-envs 4 \
  --run-name exp9-sb05-shitter \
  --force-mag 5.0 \
  --survival-bonus 0.5 \
  --retention-scale 10.0 \
  > runs/exp9-sb05-shitter/stdout.log 2>&1 &
```
mkdir first: `mkdir -p runs/exp9-sb05-shitter`

## exp10 — 3090 (launch immediately when exp8 finishes ~25min)
**Hypothesis**: Warm-starting from exp8 best checkpoint pushes eval ep_len toward 400+
**Command** (schtask on 3090):
```
python scripts/train_ppo.py \
  --total-steps 500000 \
  --n-envs 4 \
  --run-name exp10-warm-sb05 \
  --force-mag 5.0 \
  --survival-bonus 0.5 \
  --retention-scale 10.0 \
  --load-model runs/exp8-ee-delta-sb05/best/best_model.zip \
  > runs/exp10-warm-sb05/stdout.log 2>&1
```

## Fallback: if exp8 does NOT reach eval >= 200 by 300k
exp10 runs same warm-start regardless — warm-start always beneficial if exp8 > exp5.

## Post-exp10: exp11 (shitter, warm-start from exp10 best)
If still below 400, extend training on shitter using exp10's best checkpoint.
