# Robot State Observation Expansion — Design Spec

**Date:** 2026-04-19
**PDF item:** 6 (Uksang Phase 4 TODO)
**Status:** Approved, ready for implementation

## Goal

Expand `GraspEnv`'s observation from 6D `[slip(3), ee_pos(3)]` to 45D by adding full robot joint state: arm positions, hand positions, and hand velocities. This gives the policy proprioceptive awareness of its own configuration.

## New Observation Layout

| Field | Dim | Source |
|---|---|---|
| `slip` | 3 | `_compute_slip()` — object COM vel in palm frame |
| `ee_pos` | 3 | `data.site_xpos[ee_site]` |
| `arm_qpos` | 7 | xArm joint positions (joint1–joint7) |
| `hand_qpos` | 16 | LEAP hand joint positions (leap_right/0–15) |
| `hand_qvel` | 16 | LEAP hand joint velocities (leap_right/0–15) |
| **Total** | **45** | |

Slip and ee_pos remain first — primary signal stays up front, matching v1 spec ordering.

## Index Caching

At `__init__` time, map joint names to qpos/qvel slots using `mj_name2id` + `jnt_qposadr` / `jnt_dofadr`:

```python
arm_joint_names = [f"joint{i}" for i in range(1, 8)]
hand_joint_names = [f"leap_right/{i}" for i in range(16)]

self._arm_qpos_idx = np.array([
    model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
    for n in arm_joint_names
])
self._hand_qpos_idx = np.array([
    model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
    for n in hand_joint_names
])
self._hand_qvel_idx = np.array([
    model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
    for n in hand_joint_names
])
```

Cached once — zero overhead per step.

## Changes to env.py

1. **`__init__`**: cache `_arm_qpos_idx`, `_hand_qpos_idx`, `_hand_qvel_idx`. Change `observation_space` shape `(6,)` → `(45,)`.
2. **`_obs()`**: concatenate `[slip, ee_pos, arm_qpos, hand_qpos, hand_qvel]` as float32.

## Changes to test_env.py

- Assert `obs.shape == (45,)` instead of `(6,)`.

## Not changed

- Action space (23D joint-space)
- Reward function
- Termination logic
- Slip computation
- Disturbance forces

## Verification

Smoke test should show:
- `obs.shape == (45,)` with finite values
- `arm_qpos` close to home pose `[0, -0.247, 0, 0.909, 0, 1.156, 0]` after reset
- `hand_qvel` near zero after reset (object held at rest)
