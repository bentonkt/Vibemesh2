# Hardcoded Tomato Can Grasp

## Goal

Create a script that drives the xArm + LEAP hand through a predefined grasp sequence to pick up and hold the tomato soup can (`005_tomato_soup_can`), validating that the simulation's contact/friction parameters support grasping.

## New file

`scripts/hardcoded_grasp.py` — imports `build_scene()` and `build_robot_spec()` from `test_scene.py`.

## Grasp sequence (state machine)

Four phases, each defined by an EE mocap target and finger configuration:

| Phase | EE Mocap Target | Fingers | Transition |
|-------|----------------|---------|------------|
| **PRE_GRASP** | Above can (~15cm), palm-down | Open (joints ~0) | EE position error < 1cm |
| **DESCEND** | Lowered to grasp height (can center) | Open | EE position error < 1cm |
| **CLOSE** | Hold position | Close around can | Finger joints settle (low velocity) |
| **LIFT** | Raise ~15cm | Hold closed | EE position error < 1cm, then hold indefinitely |

## Finger close strategy

The LEAP hand has 4 fingers total (3 standard + thumb):
- `tip_1`, `tip_2`, `tip_3` (index, middle, ring) curl inward from one side
- `th_tip` (thumb) opposes from the other side
- Finger mocap targets are set to positions producing a cylindrical wrap around the ~6.6cm diameter can

## Control flow

Same sim loop as `test_scene.py`:
1. State machine updates mocap targets based on current phase
2. IK solver (mink) converts EE + finger mocap targets to joint commands
3. Joint commands written to `data.ctrl`
4. `mj_step` advances physics
5. Viewer syncs

## What gets reused

- `build_scene()`, `build_robot_spec()` from `test_scene.py`
- All IK setup: mink Configuration, FrameTask, RelativeFrameTask for fingers
- Contact parameters, proxy geoms, hand-object pair settings

## What's new

- State machine driving mocap targets through waypoints
- Finger open/close target positions (palm-relative)
- Phase transition logic (position error threshold + optional settling time)
- Hardcoded waypoint positions and palm-down orientation quaternion

## Object details

- Tomato soup can: ~6.6cm diameter, ~10cm tall, 349g
- Spawns at `[0.4, 0.0, 0.12]`
- Collision geom: `005_tomato_soup_can_collision_geom`

## Success criteria

- Hand moves to can, closes fingers, lifts can off the ground
- Can stays in hand without slipping or jittering
- Viewer shows the full sequence smoothly
