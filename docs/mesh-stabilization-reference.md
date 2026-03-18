# Mesh & Interaction Stabilization — Lessons from the mink Repo

How we stabilized YCB object meshes and robotic arm/hand interactions in the [mink fork](https://github.com/bentonkt/mink), across the **Grasp** and **main** branches.

---

## 1. Dual-Geom Pattern (Visual vs. Collision)

Every YCB object gets two geoms on its body:

- **Visual geom** (`contype=0, conaffinity=0`) — full-resolution mesh for rendering, zero physics participation
- **Collision geom** (`condim=6`) — either a decimated convex-hull mesh or a **box primitive override** for physics

```python
# Visual — render only
vgeom.contype = 0
vgeom.conaffinity = 0

# Collision — physics only, invisible
cgeom.rgba[:] = [0.0, 0.0, 0.0, 0.0]
cgeom.condim = 6
```

### Box Primitive Override for Rectangular Objects

The most important mesh stabilization insight: **3D-scanned convex hulls of box-shaped objects have slightly curved "flat" faces** due to scan noise. This causes the object to balance on a corner edge — a false equilibrium that no amount of contact parameter tuning can fix.

The solution: replace the convex-hull collision mesh with a **box primitive** (`mjGEOM_BOX`) for rectangular objects:

```python
# From _add_ycb_object() in arm_hand_xarm_leap_ycb.py:
if col_box_halfextents is not None:
    # Box primitive has mathematically flat faces and correct
    # contact normals, eliminating the false-equilibrium problem.
    cgeom.type = mujoco.mjtGeom.mjGEOM_BOX
    cgeom.size[:3] = col_box_halfextents
else:
    # Mesh convex-hull for curved objects (default)
    cgeom.type = mujoco.mjtGeom.mjGEOM_MESH
    cgeom.meshname = f"{obj_name}_col"
```

This is called out explicitly in the code comments:
> "Scan-mesh convex hulls of rectangular objects have slightly non-flat faces (scan noise) that let the box balance on a corner. A primitive has perfectly flat faces and eliminates this problem permanently."

---

## 2. Per-Object Contact Parameter Tuning

The `library.py` YCB manipulation library defines a `ContactParams` dataclass with defaults and per-object overrides:

### Default Contact Parameters
```python
ContactParams(
    friction=(sanitized from physics.json, clamped to [0.3, 2.5]),
    condim=6,
    solref=(0.02, 1.0),
    solimp=(0.9, 0.95, 0.001, 0.5, 2.0),
)
```

### Overrides for Problematic Objects
Thin/concave objects get **softer contacts and lower friction** to prevent jitter:

```python
_CONTACT_OVERRIDES = {
    "072-a_toy_airplane": ContactParams(
        friction=(0.8, 0.02, 0.001),
        solref=(0.03, 1.0),       # softer (longer time constant)
        solimp=(0.88, 0.95, 0.002, 0.5, 2.0),
    ),
    "031_spoon": ContactParams(friction=(0.7, 0.01, 0.001), solref=(0.03, 1.0), ...),
    "049_small_clamp": ContactParams(friction=(0.8, 0.02, 0.001), solref=(0.03, 1.0), ...),
}
```

### Friction Sanitization
Raw friction values from the physics JSON are clamped to sane ranges:
- Sliding: `[0.3, 2.5]`
- Torsional: `[0.001, 0.05]`
- Rolling: `[0.00005, 0.01]`

---

## 3. Full 6-Component Inertia Tensors

Instead of using just the diagonal inertia `(Ixx, Iyy, Izz)`, the code reads the **full 6-component tensor** from the YCB XML files (`fullinertia` attribute), preserving off-diagonal terms that couple rotational axes:

```python
full_inertia = _read_full_inertia(xml_path)  # [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
body.fullinertia[:] = full_inertia
body.explicitinertial = True
```

This matters for asymmetric objects (mugs, bottles, tools) where the off-diagonal terms affect how the object tumbles and settles.

---

## 4. Physics Validation & Audit System

The `library.py` includes an `_audit_spec()` function that flags suspicious physical parameters before they cause simulation instability:

- Non-positive or very low mass (`< 0.001 kg`)
- Very high mass (`> 6.0 kg`)
- Non-positive diagonal inertia
- **Inertia triangle inequality violations** (`Ixx + Iyy < Izz`, etc.) — physically impossible
- Friction out of expected range
- Box density too low (`< 20 kg/m³`) or too high (`> 40,000 kg/m³`)
- Non-finite inertia values

These warnings are stored per-object in `audit_warnings` and surfaced in the benchmark reports.

---

## 5. Geometry-Based Grasp Profile Selection

Instead of one-size-fits-all grasp parameters, the library selects grasp profiles based on object geometry:

```python
def _pick_grasp_profile(object_id, extents):
    if object_id in _GRASP_OVERRIDES:  # manual override for known objects
        return _GRASP_PROFILES[_GRASP_OVERRIDES[object_id]]

    # Heuristic selection from bounding box
    if ez > 0.16 and max(ex, ey) < 0.11:  # tall narrow → "tall" profile
    if ez < 0.04:                           # flat → "flat" profile
    if max_dim < 0.06:                      # small → "small" profile
    return "default"
```

Each profile tunes: XY offset, hover/approach/lift Z offsets, grasp fraction, and step counts per phase. This is what enabled **95/95 YCB objects passing** the benchmark.

---

## 6. Grasp Lock Assist (Kinematic Attachment)

Pure friction grasping is unreliable with convex-hull collision meshes. The solution: after hand-object contact is detected, **rigidly lock the object to the end-effector** by maintaining a fixed transform offset:

```python
# On contact detection:
lock_offset = object_pos - eef_pos
lock_engaged = True

# Every physics step while locked:
eef_pos = data.site_xpos[attachment_site_id]
data.qpos[object_qadr:object_qadr+3] = eef_pos + lock_offset
data.qvel[object_dadr:object_dadr+6] = 0.0
mujoco.mj_forward(model, data)
```

This is enabled by default and can be disabled (`--disable-grasp-lock-assist`) for testing pure physics grasping.

---

## 7. Hand Strength Tuning

The hardcoded grasp script boosts LEAP hand actuator gains at runtime to improve grasp force:

```python
def _configure_hand_strength(model, hand_act_ids, kp_scale=3.0, joint_force_limit=3.0):
    for aid in hand_act_ids:
        kp0 = model.actuator_gainprm[aid, 0]
        model.actuator_gainprm[aid, 0] = kp0 * kp_scale          # 3x position gain
        model.actuator_biasprm[aid, 1] *= kp_scale                # matching bias
        model.actuator_biasprm[aid, 2] *= np.sqrt(kp_scale)       # damping scales with √kp
    # Also raise joint force limits
    model.jnt_actfrcrange[jid] = [-3.0, 3.0]
```

---

## 8. Settle Phase Before Grasping

Objects are spawned slightly above the table and allowed to **settle under gravity for ~4 seconds** (800 steps at 200 Hz) before any grasp waypoints are computed. The grasp target position is read from the object's actual settled pose, not its spawn pose:

```python
if phase == Phase.SETTLE and step_count >= SETTLE_STEPS:
    settled_pos = _object_pos(model, data, GRASP_TARGET)
    # Compute waypoints from actual resting position
    hover_pos = settled_pos + [0, 0, HOVER_Z_OFFSET]
    approach_pos = settled_pos + [0, 0, APPROACH_Z_OFFSET]
```

This handles objects that shift or rotate during settling (common with round/asymmetric meshes).

---

## 9. Decoupled Arm IK + Hand Direct Control

The arm is controlled via mink's QP-based IK solver, but **hand joints are controlled directly via actuator `ctrl`** and excluded from IK:

```python
# Solve IK for arm only
vel = mink.solve_ik(configuration, tasks, rate.dt, solver, ...)
vel[hand_dof_ids] = 0.0  # zero out hand DOFs in IK solution
configuration.integrate_inplace(vel, rate.dt)

# Hand controlled independently via interpolated open/close targets
_set_hand_ctrl(model, data, hand_act_ids, close_frac=hand_frac)
```

This prevents the IK solver from fighting the hand actuators and keeps finger motion smooth during grasps.

---

## 10. Full-Dataset Benchmark (95/95 Pass)

The `benchmark.py` validates every YCB object across multiple grasp/release cycles, checking:
- Gravity settle stability (no explosion, bounded velocity)
- Hand-object contact during grasp
- Object lift delta above threshold
- Successful release and re-settle
- Bounded contact penetration depth

All 95 YCB objects pass with the combination of: per-object contact overrides, geometry-based grasp profiles, grasp lock assist, and the box primitive substitution for rectangular objects.

---

## Key Files (in mink repo, Grasp branch)

| File | Role |
|------|------|
| `examples/arm_hand_xarm_leap_ycb.py` | YCB scene with 3 objects, state-machine grasp sequence |
| `examples/arm_hand_xarm_leap_hardcoded_grasp.py` | Single-object grasp with strength tuning, lock assist, validation |
| `examples/ycb_manipulation/library.py` | YCB object spec library: contact params, grasp profiles, friction sanitization, audit |
| `examples/ycb_manipulation/benchmark.py` | Headless benchmark across all 95 YCB objects |
| `examples/ycb_manipulation/reports/ycb_validation_report.md` | 95/95 pass results |
| `examples/inspect_xarm_leap_mujoco.py` | Diagnostic: kinematic chains, actuators, contact setup, stability checks |
