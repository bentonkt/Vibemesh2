# Hardcoded Tomato Can Grasp

## Goal

Create a script that drives the xArm + LEAP hand through a predefined grasp sequence to pick up and hold the tomato soup can (`005_tomato_soup_can`), validating that the simulation's contact/friction parameters support grasping.

## New file

`scripts/hardcoded_grasp.py` — imports `build_scene()` and `build_robot_spec()` from `test_scene.py`.

## Grasp sequence (state machine)

Four phases, each defined by an EE mocap target and finger configuration:

| Phase | EE Mocap Target | Fingers | Transition |
|-------|----------------|---------|------------|
| **PRE_GRASP** | Above can (~15cm), palm-down | Open (joints ~0) | EE pos error < 1cm AND orientation error < 0.1 rad |
| **DESCEND** | Lowered to grasp height (can center) | Open | EE pos error < 1cm AND orientation error < 0.1 rad |
| **CLOSE** | Hold position | Close around can | Finger qvel < 0.05 rad/s for 0.5s (see below) |
| **LIFT** | Raise ~15cm | Hold closed | EE pos error < 1cm, then hold indefinitely |

### Phase timeout

Each phase has a maximum duration (e.g., 5s). If the IK doesn't converge within that window, print a warning and advance anyway. This prevents infinite loops on unreachable targets.

## Finger close strategy

The LEAP hand has 4 fingers total (3 standard + thumb):
- `tip_1` (index), `tip_2` (middle), `tip_3` (ring) curl inward from one side
- `th_tip` (thumb) opposes from the other side

### Close approach: direct joint angle targets

During the CLOSE phase, **bypass finger IK** and drive the 16 LEAP joint actuators directly via `data.ctrl`. This avoids conflicts between the finger mocap palm-following logic and the IK solver fighting contact forces.

Starting joint angle targets for a cylindrical wrap (will need tuning):
- **Index/Middle/Ring** (joints 0-11, 4 per finger): MCP spread ~0, MCP flex ~1.2 rad, DIP ~1.0 rad, fingertip ~1.0 rad
- **Thumb** (joints 12-15): PIP_4 ~1.0 rad, thumb PIP ~1.2 rad, thumb DIP ~0.5 rad, thumb tip ~0.8 rad

These are initial estimates — the implementation will print joint values and allow easy tuning.

### CLOSE transition criterion

Monitor `data.qvel` for the LEAP hand joints (indices 7-22 in qvel, corresponding to the 16 hand DOFs after the 7 arm joints). Transition when `max(abs(qvel[7:23])) < 0.05` has held for at least 0.5 seconds (100 timesteps at 200Hz).

## Palm-down orientation

Read the `attachment_site` orientation from the home pose at startup (`mink.SE3.from_frame_name(...)`) and use that as the base orientation. Then apply a rotation to point palm downward. The exact quaternion will be read from the sim at runtime rather than hardcoded, since it depends on the hand attachment transform.

## Control flow

Same sim loop structure as `test_scene.py`:
1. State machine checks phase transition conditions
2. State machine updates EE mocap target position/orientation
3. During PRE_GRASP/DESCEND/LIFT: finger mocap targets follow palm (same as test_scene.py), IK solves for all joints
4. During CLOSE: **skip finger IK and palm-following**; write finger joint targets directly to `data.ctrl[7:23]`; IK still solves for arm joints only (to hold EE position)
5. `mj_step` advances physics
6. Viewer syncs

## What gets reused

- `build_scene()`, `build_robot_spec()` from `test_scene.py`
- IK setup: mink Configuration, FrameTask, RelativeFrameTask for fingers (fingers only active in non-CLOSE phases)
- Contact parameters, proxy geoms, hand-object pair settings
- Temp directory cleanup pattern (`finally: shutil.rmtree(temp_dir)`)

## What's new

- State machine with 4 phases and transition logic
- Direct finger joint angle targets for CLOSE phase
- Phase timeout safety
- EE waypoint positions derived from can spawn position

## Object details

- Tomato soup can: ~6.6cm diameter, ~10cm tall, 349g
- Spawns at `[0.4, 0.0, 0.12]`
- Collision geom: `005_tomato_soup_can_collision_geom`

## Success criteria

- Hand moves to can, closes fingers, lifts can off the ground
- Can stays in hand without slipping or jittering
- Viewer shows the full sequence smoothly
