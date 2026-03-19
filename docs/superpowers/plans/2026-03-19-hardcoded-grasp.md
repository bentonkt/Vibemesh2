# Hardcoded Tomato Can Grasp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a script that autonomously grasps and lifts the tomato soup can using a hardcoded waypoint sequence.

**Architecture:** A state machine drives the xArm end-effector through 4 phases (pre-grasp, descend, close, lift) by programmatically setting mocap targets. The existing IK solver from `test_scene.py` converts these to joint commands. During the CLOSE phase, finger joints are driven directly via `data.ctrl` instead of IK.

**Tech Stack:** MuJoCo, mink (IK), numpy

**Spec:** `docs/superpowers/specs/2026-03-19-hardcoded-grasp-design.md`

---

## File Structure

- **Create:** `scripts/hardcoded_grasp.py` — the grasp script with state machine, waypoints, and sim loop
- **Reuse (import from):** `scripts/test_scene.py` — `build_scene()`, `build_robot_spec()`, constants

---

### Task 1: Make `test_scene.py` importable

Extract the shared functions so `hardcoded_grasp.py` can import them without triggering `main()`.

**Files:**
- Modify: `scripts/test_scene.py`

- [ ] **Step 1: Verify current `__main__` guard**

Read `scripts/test_scene.py` and confirm `main()` is only called inside `if __name__ == "__main__"`. It already is (lines 312-313), so this should be a no-op. Verify by running:

```bash
python -c "from scripts.test_scene import build_scene; print('import OK')"
```

If this fails due to module path issues, add an empty `scripts/__init__.py` or use relative imports in the new script instead.

- [ ] **Step 2: Commit if any changes were needed**

---

### Task 2: Scaffold `hardcoded_grasp.py` with scene setup and IK initialization

Create the new script with scene building, IK setup, and the sim loop skeleton — but no state machine yet. It should launch the viewer with the hand at home pose and the can on the table, identical to `test_scene.py` but without interactive mocap dragging.

**Files:**
- Create: `scripts/hardcoded_grasp.py`

- [ ] **Step 1: Create the script**

```python
#!/usr/bin/env python3
"""Hardcoded grasp: xArm + LEAP hand picks up a tomato soup can.

Usage:
    python scripts/hardcoded_grasp.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

# Import scene building from test_scene (same directory)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_scene import (
    build_scene, FINGERS, OBJECT_SPAWN_Z,
)

# Can geometry
CAN_ID = "005_tomato_soup_can"
CAN_RADIUS = 0.033  # ~6.6cm diameter
CAN_HEIGHT = 0.10   # ~10cm tall
CAN_SPAWN = np.array([0.4, 0.0, OBJECT_SPAWN_Z])


def main():
    print(f"Building scene with {CAN_ID}...")
    model, data, temp_dir = build_scene(CAN_ID)
    print(f"Scene loaded: {model.nbody} bodies, {model.ngeom} geoms")

    robot_nu = model.nu
    assert robot_nu == 23, f"Expected 23 actuators (7 arm + 16 hand), got {robot_nu}"

    # IK configuration
    configuration = mink.Configuration(model)
    ee_task = mink.FrameTask(
        frame_name="attachment_site", frame_type="site",
        position_cost=1.0, orientation_cost=1.0, lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=5e-2)
    finger_tasks = []
    for finger in FINGERS:
        finger_tasks.append(mink.RelativeFrameTask(
            frame_name=f"leap_right/{finger}", frame_type="site",
            root_name="leap_right/palm_lower", root_type="body",
            position_cost=1.0, orientation_cost=0.0, lm_damping=1e-3,
        ))
    tasks = [ee_task, posture_task, *finger_tasks]
    limits = [mink.ConfigurationLimit(model=model)]

    try:
        with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)

            # Initialize from home pose
            configuration.update(data.qpos)
            posture_task.set_target_from_configuration(configuration)
            mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")
            for finger in FINGERS:
                mink.move_mocap_to_frame(
                    model, data, f"{finger}_target", f"leap_right/{finger}", "site"
                )

            T_palm_prev = configuration.get_transform_frame_to_world(
                "leap_right/palm_lower", "body"
            )
            data.ctrl[:robot_nu] = data.qpos[:robot_nu]

            rate = RateLimiter(frequency=200.0, warn=False)
            while viewer.is_running():
                configuration.update(data.qpos)

                # EE task from mocap
                T_wt = mink.SE3.from_mocap_name(model, data, "target")
                ee_task.set_target(T_wt)

                # Track palm transform unconditionally (prevents stale delta)
                T_palm = configuration.get_transform_frame_to_world(
                    "leap_right/palm_lower", "body"
                )
                T_delta = T_palm @ T_palm_prev.inverse()
                T_palm_prev = T_palm.copy()

                # Palm-following + finger IK only during open-finger phases
                use_finger_ik = phase in (Phase.PRE_GRASP, Phase.DESCEND)
                if use_finger_ik:
                    for finger in FINGERS:
                        mocap_id = model.body(f"{finger}_target").mocapid[0]
                        T_w_mocap = mink.SE3.from_mocap_id(data, mocap_id)
                        T_new = T_delta @ T_w_mocap
                        data.mocap_pos[mocap_id] = T_new.translation()
                        data.mocap_quat[mocap_id] = T_new.rotation().wxyz

                    world_to_palm = T_palm.inverse()
                    for finger, task in zip(FINGERS, finger_tasks):
                        T_world_target = mink.SE3.from_mocap_name(
                            model, data, f"{finger}_target"
                        )
                        task.set_target(world_to_palm @ T_world_target)

                # Solve IK — use full task list when fingers active, arm-only otherwise
                active_tasks = tasks if use_finger_ik else [ee_task, posture_task]
                vel = mink.solve_ik(
                    configuration, active_tasks, rate.dt, "daqp", damping=1e-3, limits=limits
                )
                configuration.integrate_inplace(vel, rate.dt)

                if use_finger_ik:
                    data.ctrl[:robot_nu] = configuration.q[:robot_nu]
                else:
                    data.ctrl[:ARM_NU] = configuration.q[:ARM_NU]

                mujoco.mj_step(model, data)
                viewer.sync()
                rate.sleep()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run to verify it launches the viewer with arm + can at home pose**

```bash
python scripts/hardcoded_grasp.py
```

Expected: viewer opens, arm at home pose, can on table. Nothing moves autonomously (same as test_scene but no interactive dragging).

- [ ] **Step 3: Commit**

```bash
git add scripts/hardcoded_grasp.py
git commit -m "Scaffold hardcoded_grasp.py with scene + IK setup"
```

---

### Task 3: Add the state machine and EE waypoints

Add the 4-phase state machine that drives the EE mocap target through pre-grasp → descend → close → lift.

**Files:**
- Modify: `scripts/hardcoded_grasp.py`

- [ ] **Step 1: Add state enum and waypoint definitions**

Add above `main()`:

```python
from enum import Enum, auto

class Phase(Enum):
    PRE_GRASP = auto()
    DESCEND = auto()
    CLOSE = auto()
    LIFT = auto()
    DONE = auto()

# EE waypoints: (position, description)
# Positions are world-frame, derived from CAN_SPAWN
PRE_GRASP_HEIGHT = 0.15  # above can top
GRASP_HEIGHT_OFFSET = 0.02  # above can center (tune this)
LIFT_HEIGHT = 0.15  # how high to lift

PHASE_TIMEOUT_STEPS = 1000  # 5s at 200Hz
POS_THRESHOLD = 0.01  # 1cm
SETTLE_THRESHOLD = 0.05  # rad/s for finger joints
SETTLE_STEPS = 100  # 0.5s at 200Hz
```

- [ ] **Step 2: Add state machine logic inside the sim loop**

Before the `while viewer.is_running()` loop, add state initialization:

```python
phase = Phase.PRE_GRASP
phase_step = 0
settle_count = 0

# Read palm-down orientation from home pose
T_home_ee = mink.SE3.from_mocap_name(model, data, "target")
palm_down_quat = T_home_ee.rotation().wxyz  # starting orientation

# Compute waypoints from can position
pre_grasp_pos = CAN_SPAWN.copy()
pre_grasp_pos[2] = CAN_SPAWN[2] + CAN_HEIGHT / 2 + PRE_GRASP_HEIGHT

descend_pos = CAN_SPAWN.copy()
descend_pos[2] = CAN_SPAWN[2] + GRASP_HEIGHT_OFFSET

lift_pos = descend_pos.copy()
lift_pos[2] = descend_pos[2] + LIFT_HEIGHT

print(f"Waypoints: pre_grasp={pre_grasp_pos}, descend={descend_pos}, lift={lift_pos}")
```

Inside the loop, after `configuration.update(data.qpos)`, add the state machine:

```python
phase_step += 1

# Get current EE position
ee_pos = data.site("attachment_site").xpos.copy()

if phase == Phase.PRE_GRASP:
    data.mocap_pos[model.body("target").mocapid[0]] = pre_grasp_pos
    data.mocap_quat[model.body("target").mocapid[0]] = palm_down_quat
    pos_err = np.linalg.norm(ee_pos - pre_grasp_pos)
    if pos_err < POS_THRESHOLD or phase_step > PHASE_TIMEOUT_STEPS:
        if phase_step > PHASE_TIMEOUT_STEPS:
            print(f"PRE_GRASP timed out (err={pos_err:.4f})")
        else:
            print(f"PRE_GRASP reached (err={pos_err:.4f})")
        phase = Phase.DESCEND
        phase_step = 0

elif phase == Phase.DESCEND:
    data.mocap_pos[model.body("target").mocapid[0]] = descend_pos
    data.mocap_quat[model.body("target").mocapid[0]] = palm_down_quat
    pos_err = np.linalg.norm(ee_pos - descend_pos)
    if pos_err < POS_THRESHOLD or phase_step > PHASE_TIMEOUT_STEPS:
        if phase_step > PHASE_TIMEOUT_STEPS:
            print(f"DESCEND timed out (err={pos_err:.4f})")
        else:
            print(f"DESCEND reached (err={pos_err:.4f})")
        phase = Phase.CLOSE
        phase_step = 0
        settle_count = 0

elif phase == Phase.CLOSE:
    # Hold EE position
    data.mocap_pos[model.body("target").mocapid[0]] = descend_pos
    data.mocap_quat[model.body("target").mocapid[0]] = palm_down_quat
    # Finger control handled in Task 4
    pass

elif phase == Phase.LIFT:
    data.mocap_pos[model.body("target").mocapid[0]] = lift_pos
    data.mocap_quat[model.body("target").mocapid[0]] = palm_down_quat
    pos_err = np.linalg.norm(ee_pos - lift_pos)
    if pos_err < POS_THRESHOLD or phase_step > PHASE_TIMEOUT_STEPS:
        print(f"LIFT complete (err={pos_err:.4f})")
        phase = Phase.DONE
        phase_step = 0

elif phase == Phase.DONE:
    # Hold position
    data.mocap_pos[model.body("target").mocapid[0]] = lift_pos
    data.mocap_quat[model.body("target").mocapid[0]] = palm_down_quat
```

- [ ] **Step 3: Run and verify the arm moves through PRE_GRASP → DESCEND**

```bash
python scripts/hardcoded_grasp.py
```

Expected: arm moves above can, then lowers. CLOSE phase begins but fingers don't close yet (that's Task 4). Console prints phase transitions.

- [ ] **Step 4: Commit**

```bash
git add scripts/hardcoded_grasp.py
git commit -m "Add grasp state machine with EE waypoints"
```

---

### Task 4: Add finger close and CLOSE→LIFT transition

Drive finger joints directly during CLOSE phase, bypassing finger IK. Add the settle detection to transition to LIFT.

**Files:**
- Modify: `scripts/hardcoded_grasp.py`

- [ ] **Step 1: Define finger close joint targets**

Add constants near the top (after `CAN_SPAWN`):

```python
# Number of arm actuators (xArm7 = 7)
ARM_NU = 7

# Finger close targets: actuator indices 7-22 (16 hand joints)
# Order matches actuator definition in right_hand.xml with "leap_right/" prefix.
# Each finger has 4 joints. Values in radians — will need tuning.
# fmt: off
FINGER_OPEN = np.array([
    0.0, 0.0, 0.0, 0.0,   # finger 1 (index)
    0.0, 0.0, 0.0, 0.0,   # finger 2 (middle)
    0.0, 0.0, 0.0, 0.0,   # finger 3 (ring)
    0.0, 0.0, 0.0, 0.0,   # thumb
])
FINGER_CLOSED = np.array([
    1.2, 0.0, 1.0, 1.0,   # finger 1: MCP-flex, spread, DIP, tip
    1.2, 0.0, 1.0, 1.0,   # finger 2: MCP-flex, spread, DIP, tip
    1.2, 0.0, 1.0, 1.0,   # finger 3: MCP-flex, spread, DIP, tip
    1.0, 1.2, 0.5, 0.8,   # thumb: PIP_4, PIP, DIP, tip
])
# fmt: on
```

- [ ] **Step 2: Modify the sim loop to handle CLOSE phase finger control**

In the CLOSE branch of the state machine, add finger driving and settle detection:

```python
elif phase == Phase.CLOSE:
    data.mocap_pos[model.body("target").mocapid[0]] = descend_pos
    data.mocap_quat[model.body("target").mocapid[0]] = palm_down_quat

    # Drive fingers directly
    data.ctrl[ARM_NU:] = FINGER_CLOSED

    # Check if fingers have settled
    hand_qvel = data.qvel[ARM_NU:ARM_NU + 16]
    if np.max(np.abs(hand_qvel)) < SETTLE_THRESHOLD:
        settle_count += 1
    else:
        settle_count = 0

    if settle_count >= SETTLE_STEPS or phase_step > PHASE_TIMEOUT_STEPS:
        if phase_step > PHASE_TIMEOUT_STEPS:
            print(f"CLOSE timed out (max_qvel={np.max(np.abs(hand_qvel)):.4f})")
        else:
            print(f"CLOSE settled (max_qvel={np.max(np.abs(hand_qvel)):.4f})")
        phase = Phase.LIFT
        phase_step = 0
```

The `use_finger_ik` gating, `active_tasks` selection, and arm-only ctrl write are already in the scaffold from Task 2. The CLOSE branch just needs to write `data.ctrl[ARM_NU:] = FINGER_CLOSED` — the rest is handled by the existing `else` branch in the IK section.

- [ ] **Step 3: Run and verify the full grasp sequence**

```bash
python scripts/hardcoded_grasp.py
```

Expected: arm moves to can → lowers → fingers close around can → arm lifts → can comes up with it. Console prints all phase transitions.

- [ ] **Step 4: Tune finger joint targets if needed**

If the can slips or fingers don't wrap properly:
- Adjust `FINGER_CLOSED` values
- Adjust `GRASP_HEIGHT_OFFSET` (how low the hand goes)
- Watch the viewer to see where contact is happening

- [ ] **Step 5: Commit**

```bash
git add scripts/hardcoded_grasp.py
git commit -m "Add finger close control and full grasp sequence"
```

---

### Task 5: Polish and verify

Clean up print output, verify the grasp is stable, ensure the script is self-contained and easy to run.

**Files:**
- Modify: `scripts/hardcoded_grasp.py`

- [ ] **Step 1: Add phase status display in the viewer title or console**

Add a print at each phase change showing elapsed time:

```python
elapsed = phase_step / 200.0  # seconds
print(f"[{elapsed:.1f}s] Phase: {phase.name}")
```

- [ ] **Step 2: Run full sequence 2-3 times to verify stability**

```bash
python scripts/hardcoded_grasp.py
```

Verify: can is grasped and held consistently. No jitter, no slipping, no explosions.

- [ ] **Step 3: Final commit**

```bash
git add scripts/hardcoded_grasp.py
git commit -m "Polish hardcoded grasp script"
```
