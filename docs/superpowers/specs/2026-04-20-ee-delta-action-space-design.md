# EE-Delta Action Space Design

**Date:** 2026-04-20
**Author:** vibemesh-ee-delta worker agent
**PDF Ref:** VibeMesh 2.0 Mujoco Setup (2).pdf, items 6 & 7

---

## Problem

**exp4 finding:** With `force_mag=0`, eval `ep_len` stayed at ~20 — the object dropped almost immediately despite no external disturbance. The only destabilizing influence was the policy itself. The 23D raw joint action space lets PPO's stochastic policy wiggle all 7 arm joints independently each step, randomly repositioning the wrist and destabilizing the grasp before any meaningful signal is available.

**PDF spec mismatch:** Item 6 specifies a 22D action space: `[ee_pos_delta(3), ee_rot_delta(3), hand_ctrl(16)]`. The arm portion should be expressed as small Cartesian deltas, not raw joint targets. This constrains arm motion to small, physically coherent EE movements while freeing the hand (LEAP joints) to reconfigure for slip recovery.

---

## New Action Space: 22D

| Slice     | Description            | Bounds          | Unit  |
|-----------|------------------------|-----------------|-------|
| `[0:3]`   | EE position delta      | ±0.01 per step  | m     |
| `[3:6]`   | EE rotation delta      | ±0.05 per step  | rad   |
| `[6:22]`  | LEAP hand joint targets| model ctrlrange | rad   |

The hand portion remains absolute (not deltas) — the hand needs to reconfigure fully for slip recovery.

---

## Mink IK Integration

Each `step()` call:

```python
# 1. Decode action
ee_pos_delta = action[:3]           # meters, world frame
ee_rot_delta = action[3:6]          # axis-angle, rad
hand_ctrl    = action[6:22]         # absolute LEAP joint targets

# 2. Read current EE pose from MuJoCo site
ee_pos  = data.site_xpos[ee_site_id]           # (3,)
ee_xmat = data.site_xmat[ee_site_id].reshape(3, 3)  # rotation matrix

# 3. Compute target pose
target_pos = ee_pos + ee_pos_delta
target_so3 = mink.SO3.from_matrix(ee_xmat) @ mink.SO3.exp(ee_rot_delta)
target_se3 = mink.SE3.from_rotation_and_translation(target_so3, target_pos)

# 4. Solve differential IK (one step)
mink_config.update(data.qpos)
ee_task.set_target(target_se3)
vel = mink.solve_ik(mink_config, [ee_task, posture_task], dt, "daqp",
                    damping=1e-3, limits=[ConfigurationLimit(model)])
mink_config.integrate_inplace(vel, dt)

# 5. Write controls
data.ctrl[:ARM_NU]             = mink_config.q[:ARM_NU]  # arm joints
data.ctrl[ARM_NU:ARM_NU+HAND_NU] = hand_ctrl              # LEAP hand
```

`dt = n_substeps × model.opt.timestep` (e.g., 5 × 0.001 = 0.005 s).

Posture task (`cost=5e-2`) is set once per episode from the post-keyframe configuration and regularizes the arm toward the grasp pose.

---

## Changes in env.py

- Add module constants `ARM_NU = 7`, `HAND_NU = 16`.
- `__init__`: Initialize `mink.Configuration`, `mink.FrameTask("attachment_site", "site")`, `mink.PostureTask`, `mink.ConfigurationLimit`. Set `action_space` to `Box(shape=(22,))` with bounds above. Change `_prev_action` to 22D.
- `reset()`: After keyframe replay, call `mink_config.update(data.qpos)` and `posture_task.set_target_from_configuration(mink_config)`. Initialize `_prev_action` to `[zeros(6), data.ctrl[7:23]]`.
- `step()`: Replace `self.data.ctrl[:] = action` with the IK pipeline above. Smoothness penalty continues to use `action - prev_action` (both 22D now).

## Changes in train_ppo.py

No structural change needed. The 22D action space flows through SB3 automatically. `--survival-bonus` CLI arg and `survival_bonus` env arg remain (default 0.0).

---

## Reward: Revert to 3-Term Form

Spec (item 7): retention + drop + smoothness + (later: slip magnitude).

**Remove the survival bonus from the effective reward.** exp5 introduced `r_alive = survival_bonus` as a workaround for the arm-jitter bug. With the EE-delta action space, the arm can only make small coherent movements per step, eliminating the pathological destabilization. The survival bonus is no longer needed and diverges from the advisor's spec.

Implementation: `survival_bonus` constructor arg remains (default 0.0), so passing `--survival-bonus 0.0` (or omitting it) gives the 3-term reward. No code removal needed; the existing formula already defaults to 3-term.

---

## Verification Checklist

Before training:

- [ ] `env.action_space.shape == (22,)` — confirmed by test_env.py assert
- [ ] `env.action_space.low[:3]` ≈ -0.01, `env.action_space.high[:3]` ≈ 0.01
- [ ] `env.action_space.low[3:6]` ≈ -0.05, `env.action_space.high[3:6]` ≈ 0.05
- [ ] 10 random-action steps complete without exception
- [ ] `obs.shape == (45,)` and all finite
- [ ] `info['retention']` is finite and typically < 0.1 after reset
- [ ] `data.ctrl` shape unchanged (23D) — only the *input* mapping changes
- [ ] mink IK converges within the first step (no `MinkError`)
