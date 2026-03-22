# Hardcoded Tomato Can Grasp

## Goal

Create a script that drives the xArm + LEAP hand through a predefined side-grasp sequence to pick up and hold the tomato soup can (`005_tomato_soup_can`), validating that the simulation's contact/friction parameters support grasping a standing vertical cylinder.

## File

`scripts/hardcoded_grasp.py` — imports `build_scene()` from `test_scene.py`.

## Grasp sequence (state machine)

Six phases, each defined by an EE mocap target and finger configuration:

| Phase | EE Mocap Target | Fingers | Transition |
|-------|----------------|---------|------------|
| **SETTLE** | Hold home pose while object settles | Open | Fixed duration: 800 steps |
| **PRE_GRASP** | Side-hover pose, offset away from can and slightly above grasp height | Open | EE pos error < 1cm AND orientation error < 0.1 rad |
| **APPROACH** | Lateral approach from the `+Y` side to the can mid-height | Open | EE pos error < 1cm AND orientation error < 0.1 rad |
| **SEAT** | Final push-in to move the open hand onto the can | Open | Any hand-can proxy contact OR max seat bias reached |
| **CLOSE** | Hold seated side-grasp pose | Close around can | Any seated hand-can contact, then finger qvel < 0.05 rad/s for 0.5s; log two-sided proxy contact when available |
| **LIFT** | Raise vertically while holding the side-grasp pose | Hold closed | EE pos error < 1cm, then hold indefinitely |

### Phase timeout

Each phase after `SETTLE` has a maximum duration (e.g. 5s). If the IK does not converge within that window, print a warning with both position and orientation error and either advance or stop as described below. This prevents infinite loops on unreachable targets.

## Runtime object targeting

The tomato soup can mesh is not centered at the free-joint origin, so the grasp must target the real can geometry instead of the spawn pose constants.

- Load the collision mesh bounds from `data/ycb/processed/005_tomato_soup_can/collision.obj`
- Compute:
  - local mesh center: approximately `(-0.0092, 0.0840, 0.0510)`
  - local extents: approximately `(0.0679, 0.0677, 0.1019)`
- After settle, read the object body's world pose and derive:
  - `can_center_world = body_xpos + body_xmat @ can_center_local`
  - `can_axis_world = normalize(body_xmat[:, 2])`

This ensures the grasp tracks the can's actual settled position rather than the object body origin.

## Side-grasp orientation

Use a side-grasp frame instead of a palm-down frame:

- attachment `y` axis aligns with the cylinder axis
- attachment `z` axis points from the hand toward the can from the `+Y` side
- attachment `x` axis is the right-handed cross product

Build the target frame as:

```python
y_axis = can_axis_world
z_axis = normalize(project(-world_y, orthogonal_to=y_axis))
x_axis = normalize(cross(y_axis, z_axis))
R_target = [x_axis, y_axis, z_axis]
```

Convert `R_target` to a MuJoCo quaternion and use it for `PRE_GRASP`, `APPROACH`, `SEAT`, `CLOSE`, and `LIFT`.

## Contact anchor

Do not aim the attachment site using fingertip sites. The can contacts the proxy pads and palm, so the close pose must be anchored to the contact geometry:

- Temporarily set the hand to `FINGER_CLOSED`
- Read these proxy geoms in the attachment frame:
  - finger side: `proxy_pad_1`, `proxy_pad_2`, `proxy_pad_3`
  - thumb side: `proxy_th_tip`
- Compute:

```python
finger_pad_center = mean([proxy_pad_1, proxy_pad_2, proxy_pad_3])
contact_anchor_local = 0.5 * (finger_pad_center + proxy_th_tip)
```

Use:

```python
base_close_pos = can_center_world - R_target @ contact_anchor_local
```

## Waypoints

Waypoints are derived from `base_close_pos`, the side-approach direction, and a short seat-bias ladder:

```python
PRE_GRASP = base_close_pos - 0.06 * z_axis + [0, 0, 0.05]
APPROACH  = base_close_pos - 0.06 * z_axis
SEAT      = base_close_pos + seat_bias * z_axis
CLOSE     = SEAT
LIFT      = SEAT + [0, 0, 0.12]

seat_bias ladder = [0.0, 0.005, 0.010, 0.015]
```

This keeps the large move lateral while still allowing a short final press onto the can.

## Finger close strategy

The LEAP hand has 4 fingers total (3 standard + thumb):

- `tip_1` (index), `tip_2` (middle), `tip_3` (ring) close from one side
- `th_tip` (thumb) closes from the opposite side

### Close approach: direct joint angle targets

During `SEAT`, `CLOSE`, and `LIFT`, bypass finger IK and solve arm IK only. `SEAT` keeps the hand open while the wrist presses onto the can. `CLOSE` and `LIFT` drive the 16 LEAP joint actuators directly via `data.ctrl`.

Starting joint angle targets for a cylindrical wrap:

- **Index/Middle/Ring** (joints 0-11, 4 per finger): MCP spread ~0, MCP flex ~1.2 rad, DIP ~1.0 rad, fingertip ~1.0 rad
- **Thumb** (joints 12-15): PIP_4 ~1.0 rad, thumb PIP ~1.2 rad, thumb DIP ~0.5 rad, thumb tip ~0.8 rad

### Contact classification

Classify can contacts by proxy name:

- **thumb contact**: `proxy_th_tip` or `proxy_th_pad`
- **finger contact**: `proxy_pad_1..3` or `proxy_tip_1..3`
- **seat contact**: any thumb/finger proxy above, plus `proxy_palm` for diagnostics

Palm-only contact is acceptable during `SEAT`, and the script now treats sustained seated can contact as enough to continue into `LIFT` if the grasp has visibly captured the can even when simultaneous two-sided proxy contact is not observed.

### CLOSE transition criterion

Monitor `data.qvel` for the LEAP hand joints (indices 7-22 in `qvel`, corresponding to the 16 hand DOFs after the 7 arm joints). Transition to `LIFT` only when:

- some seated hand-can contact has been observed during `SEAT` or `CLOSE`
- `max(abs(qvel[7:23])) < 0.05` has held for at least 0.5 seconds (100 timesteps at 200 Hz)

Prefer two-sided proxy contact and log it when it appears. If `CLOSE` times out after seated can contact has already been observed, continue to `LIFT`; only stop before lift when no hand-can contact was observed at all.

## Control flow

1. `SETTLE`: hold home pose while the can settles under gravity
2. Compute the settled can center, can axis, side-grasp frame, proxy-based contact anchor, and initial waypoints
3. `PRE_GRASP`: finger mocap targets follow the palm; IK solves for arm + fingers
4. `APPROACH`: finger mocap targets continue following the palm while the EE moves laterally in from the side
5. `SEAT`: skip finger IK, keep fingers open, and press the wrist onto the can; if no contact is seen, increment `seat_bias`
6. `CLOSE`: hold the seated target and write finger joint targets directly to `data.ctrl[7:23]`; arm IK still holds the wrist
7. `LIFT`: after the closed hand has settled with seated can contact, keep fingers closed and raise vertically
8. `mj_step` advances physics and the viewer syncs

## What gets reused

- `build_scene()` from `test_scene.py`
- IK setup: mink `Configuration`, `FrameTask`, `RelativeFrameTask` for fingers
- Contact parameters, proxy geoms, hand-object pair settings
- Temp directory cleanup pattern (`finally: shutil.rmtree(temp_dir)`)

## What's new

- Startup `SETTLE` phase
- Runtime collision-mesh center extraction
- Runtime proxy-based contact-anchor extraction from the closed LEAP hand
- Side-approach orientation and lateral waypoints
- `SEAT` phase with final press-in bias ladder
- Contact classification for thumb-side, finger-side, and seat contact
- Orientation-gated phase transitions for `PRE_GRASP` and `APPROACH`
- Direct finger joint angle targets for `CLOSE` and `LIFT`
- Stop-before-lift behavior only when no can contact was captured at all

## Success criteria

- Hand waits for the can to settle, then moves to the can's side rather than above its top
- Wrist is oriented so the finger row runs along the can height
- Hand approaches laterally from the `+Y` side, seats onto the can, and closes around the standing can
- `LIFT` begins once the closed hand has settled on the can, even if the proxy classifier did not report simultaneous thumb-side and finger-side contact
- Arm lifts the can without reverting to a top-down orientation
- Viewer shows the full sequence smoothly
