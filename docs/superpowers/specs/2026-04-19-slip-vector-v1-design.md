# Slip Vector v1 — Design Spec

**Date:** 2026-04-19
**PDF item:** 5 (Uksang Phase 4 TODO)
**Status:** Approved, ready for implementation

## Goal

Add a 3D slip vector to `GraspEnv`'s observation. The slip vector measures how the object is moving relative to the LEAP hand. Magnitude 0 = firm hold. This is the oracle sim signal that will be replaced by a learned audio model on hardware (A-SLIP extension).

## Computation

Object COM translational velocity transformed into the `palm_lower` body's local frame:

```python
palm_xmat = data.xmat[palm_body_id].reshape(3, 3)
obj_vel_world = data.cvel[obj_body_id][3:]  # cvel is [angular(3), linear(3)]
slip = palm_xmat.T @ obj_vel_world
```

- **Stateless** — no previous-timestep storage needed.
- **Frame:** `palm_lower` body. Fixed relative to the fingers. Can be swapped later without changing obs shape.
- **Why COM velocity:** matches the real audio sensor's aggregate nature (detects overall relative motion, not per-contact detail). Simplest possible computation; item 4's per-contact approach is deferred to v2.

### Why Approach A over alternatives

| Approach | Pros | Cons |
|---|---|---|
| **A. COM velocity in palm frame** (chosen) | Stateless, 3 lines, matches audio sensor | Includes rotational contribution at COM |
| B. Position displacement in palm frame | Measures drift directly | Requires prev-step state, noisy |
| C. Contact-point-averaged velocity | Most physically accurate | Noisy contact set, complex, closer to item 4 |

## Observation space

**Before:** `Box(3,)` = EE position only.
**After:** `Box(6,)` = `[slip(3), ee_pos(3)]`.

Slip first in the vector — primary signal before state. Matches PDF item 6 ordering: "slip vector (3D), robot state."

EE position remains 3D (not full 6D pose). Adding rotation is a separate later step.

## Changes to env.py

1. **`__init__`**: cache `palm_body_id` via `mj_name2id("leap_right/palm_lower", BODY)`. Change `observation_space` shape `(3,)` → `(6,)`.
2. **`_compute_slip() -> np.ndarray`**: new method. 3 lines: extract `cvel[obj_body_id][3:]`, rotate by `palm_xmat.T`, return 3D.
3. **`_obs() -> np.ndarray`**: `np.concatenate([slip, ee_pos]).astype(float32)`.
4. **`step()`**: add `info["slip_mag"] = float(np.linalg.norm(slip))`.
5. **`reset()`**: add `info["slip_mag"] = 0.0`.

## Changes to test_env.py

- Assert `obs.shape == (6,)` instead of `(3,)`.
- Print `info.get("slip_mag")` each step.

## Not changed

- Action space (23D joint-space)
- Reward (drop-only)
- Termination logic
- Keyframe replay
- Physics substeps

## Verification

Smoke test should show:
- `obs.shape == (6,)` with finite values
- `slip_mag ≈ 0` after reset (object held firmly)
- `slip_mag > 0` during random-action steps (object disturbed)
- Object z behavior unchanged from before
