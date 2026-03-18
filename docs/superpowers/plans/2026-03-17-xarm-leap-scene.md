# xArm + LEAP Hand Scene Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a Python script that assembles and displays a MuJoCo scene with an xArm7 + LEAP hand robot on a table with three YCB objects.

**Architecture:** Single Python script using `MjSpec` API to programmatically load the xArm7 and LEAP hand from the mink submodule, attach them, add a table and YCB objects, then launch the MuJoCo viewer with physics stepping.

**Tech Stack:** MuJoCo (`mujoco` Python bindings, `MjSpec` API), `loop_rate_limiters`

**Spec:** `docs/superpowers/specs/2026-03-17-xarm-leap-scene-design.md`

---

### Task 1: Add `loop-rate-limiters` dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add dependency**

Add `loop-rate-limiters` to the project dependencies in `pyproject.toml`:

```toml
dependencies = [
    "fast-simplification>=0.1.13",
    "loop-rate-limiters>=0.1.0",
    "mink>=1.1.0",
    "mujoco>=3.6.0",
    "trimesh>=4.11.3",
]
```

- [ ] **Step 2: Install**

Run: `uv sync`
Expected: installs `loop-rate-limiters` successfully

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Add loop-rate-limiters dependency"
```

---

### Task 2: Create scene script with robot assembly

**Files:**
- Create: `scenes/xarm_leap_scene.py`

- [ ] **Step 1: Write the script with robot assembly and viewer**

This creates the xArm7 + LEAP hand assembly following mink's `arm_hand_xarm_leap.py` pattern, plus a floor and viewer loop.

```python
"""xArm7 + LEAP Hand scene with YCB objects on a table."""

from pathlib import Path

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MINK_EXAMPLES = _PROJECT_ROOT / "mink" / "examples"
_ARM_XML = _MINK_EXAMPLES / "ufactory_xarm7" / "scene.xml"
_HAND_XML = _MINK_EXAMPLES / "leap_hand" / "right_hand.xml"
_YCB_DIR = _PROJECT_ROOT / "assets" / "ycb"

# 7 arm joints + 16 LEAP hand joints (from mink's arm_hand_xarm_leap.py)
HOME_QPOS = [
    # xarm
    0, -0.247, 0, 0.909, 0, 1.15644, 0,
    # leap (all zeros = open hand)
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
]

# YCB object definitions: (name, mass, diaginertia, inertial_pos, friction, body_pos)
YCB_OBJECTS = [
    {
        "name": "003_cracker_box",
        "mass": 0.411,
        "diaginertia": (0.002481, 0.001736, 0.001098),
        "inertial_pos": (-0.0149, -0.0142, 0.1022),
        "friction": (1.0, 0.005, 0.0001),
        "pos": (0.25, -0.15, 0.55),
    },
    {
        "name": "004_sugar_box",
        "mass": 0.514,
        "diaginertia": (0.001707, 0.001432, 0.000485),
        "inertial_pos": (-0.0077, -0.0171, 0.0860),
        "friction": (1.0, 0.005, 0.0001),
        "pos": (0.25, 0.15, 0.50),
    },
    {
        "name": "005_tomato_soup_can",
        "mass": 0.305,
        "diaginertia": (0.000352, 0.000352, 0.000175),
        "inertial_pos": (-0.0093, 0.0842, 0.0500),
        "friction": (0.8, 0.005, 0.0001),
        "pos": (0.45, 0, 0.47),
    },
]


def construct_model() -> mujoco.MjModel:
    """Build the xArm7 + LEAP hand + table + YCB objects scene."""
    arm = mujoco.MjSpec.from_file(_ARM_XML.as_posix())
    hand = mujoco.MjSpec.from_file(_HAND_XML.as_posix())

    # Attach LEAP hand to xArm's attachment_site.
    palm = hand.body("palm_lower")
    palm.pos[:] = (0.065, -0.04, 0)
    palm.quat[:] = (0, 1, 0, 0)
    site = arm.site("attachment_site")
    arm.attach(hand, prefix="leap_right/", site=site)

    # Replace home keyframe with 23-DOF qpos.
    home_key = arm.key("home")
    arm.delete(home_key)
    arm.add_key(name="home", qpos=HOME_QPOS)

    # Set physics parameters (matching ycb_resting.xml).
    arm.option.timestep = 0.002
    arm.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    arm.option.gravity[:] = (0, 0, -9.81)

    # Add table: static box in front of the arm, surface at z=0.4.
    table = arm.worldbody.add_body(name="table", pos=(0.4, 0, 0.2))
    table.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.4, 0.3, 0.2),
        rgba=(0.8, 0.7, 0.55, 1.0),
        friction=(1.0, 0.005, 0.0001),
        solref=(0.02, 1),
        solimp=(0.9, 0.95, 0.001, 0.5, 2),
    )

    # Add YCB objects on the table.
    for obj in YCB_OBJECTS:
        mesh_dir = _YCB_DIR / obj["name"] / "google_16k"
        visual_mesh = mesh_dir / "nontextured_clean.stl"
        collision_mesh = mesh_dir / "nontextured_collision.stl"

        # Add mesh assets with absolute paths.
        arm.add_mesh(
            name=f"{obj['name']}_visual",
            file=visual_mesh.as_posix(),
        )
        arm.add_mesh(
            name=f"{obj['name']}_collision",
            file=collision_mesh.as_posix(),
        )

        # Add body with freejoint.
        body = arm.worldbody.add_body(name=obj["name"], pos=obj["pos"])
        body.add_freejoint()
        body.mass = obj["mass"]
        body.ipos[:] = obj["inertial_pos"]
        body.inertia[:] = obj["diaginertia"]
        body.explicitinertial = 1

        # Visual geom (render only, no collision).
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=f"{obj['name']}_visual",
            contype=0,
            conaffinity=0,
        )

        # Collision geom.
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=f"{obj['name']}_collision",
            contype=1,
            conaffinity=1,
            friction=obj["friction"],
            solref=(0.02, 1),
            solimp=(0.9, 0.95, 0.001, 0.5, 2),
        )

    return arm.compile()


def main():
    model = construct_model()
    data = mujoco.MjData(model)

    # Reset to home keyframe.
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        rate = RateLimiter(frequency=500.0, warn=False)
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            rate.sleep()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script to verify the scene loads**

Run: `cd /Users/bentontameling/Dev/Vibemesh2 && uv run python scenes/xarm_leap_scene.py`

Expected: MuJoCo viewer opens showing:
- xArm7 arm in home pose on the floor
- LEAP right hand attached at the end effector
- A table (tan box) in front of the arm
- Three YCB objects on/above the table settling under gravity
- No crashes, no contact explosions

If mesh loading fails, check that absolute paths to YCB STL files are correct. If `MjSpec` API calls fail (e.g., `add_mesh`, `add_inertial`, `meshname`), consult MuJoCo docs for the correct parameter names — the API is new and parameter names may differ from XML attribute names.

- [ ] **Step 3: Adjust if needed**

If objects are clipping through the table or flying off, adjust body positions. If the arm is too far from the table, adjust table position. The goal is a stable scene where objects rest on the table within the arm's workspace.

- [ ] **Step 4: Commit**

```bash
git add scenes/xarm_leap_scene.py
git commit -m "Add xArm + LEAP hand scene with table and YCB objects"
```
