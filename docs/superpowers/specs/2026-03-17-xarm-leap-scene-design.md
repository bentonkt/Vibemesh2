# xArm + LEAP Hand Scene Design

## Goal

Create a Python script that assembles a MuJoCo scene with an xArm7 robot arm, LEAP right hand end-effector, a table, and three YCB objects. This is the foundation for future hard-coded grasp work (starting with the tomato soup can).

## Approach

Single Python script (`scenes/xarm_leap_scene.py`) that programmatically builds the scene using `MjSpec`, following the same pattern as mink's `arm_hand_xarm_leap.py`.

## Scene Assembly

### Robot (xArm7 + LEAP Hand)

- Load mink's `ufactory_xarm7/scene.xml` as the base spec (includes arm, floor, lighting)
- Load mink's `leap_hand/right_hand.xml` as the hand spec
- Attach hand to arm's `attachment_site`:
  - `palm_lower` body pos: `(0.065, -0.04, 0)`
  - `palm_lower` body quat: `(0, 1, 0, 0)`
  - Prefix: `leap_right/`
- Replace the `home` keyframe with 23-DOF qpos (7 arm + 16 hand):
  ```python
  [0, -0.247, 0, 0.909, 0, 1.15644, 0,  # xarm
   0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0]  # leap
  ```
- Leave the mocap target body from scene.xml in place (harmless, and useful if IK control is added later)

### Table

- Static box body (no freejoint) positioned in front of the arm
- Body position: `(0.4, 0, 0.2)` — centered in front of arm base, at half the table height
- Single box geom with half-extents `(0.4, 0.3, 0.2)` — table surface ends up at z = 0.4
- Collision enabled, high friction to prevent objects sliding off
- Neutral color (e.g., light wood tone)

### YCB Objects

Three objects placed on the table surface (table top at z = 0.4, objects start slightly above to settle under gravity):

| Object | Mass | Inertia | Inertial pos (CoM) | Friction | Body position |
|--------|------|---------|-------------------|----------|--------------|
| 003_cracker_box | 0.411 kg | `(0.002481, 0.001736, 0.001098)` | `(-0.0149, -0.0142, 0.1022)` | 1.0 | `(0.25, -0.15, 0.55)` |
| 004_sugar_box | 0.514 kg | `(0.001707, 0.001432, 0.000485)` | `(-0.0077, -0.0171, 0.0860)` | 1.0 | `(0.25, 0.15, 0.50)` |
| 005_tomato_soup_can | 0.305 kg | `(0.000352, 0.000352, 0.000175)` | `(-0.0093, 0.0842, 0.0500)` | 0.8 | `(0.45, 0, 0.47)` |

Mesh files per object: `nontextured_clean.stl` (visual) and `nontextured_collision.stl` (collision), located in `assets/ycb/<object>/google_16k/`.

Each object uses the dual-geom pattern from `ycb_resting.xml`:
- Visual geom: `contype=0, conaffinity=0` (render only)
- Collision geom: `contype=1, conaffinity=1` with `solref="0.02 1"`, `solimp="0.9 0.95 0.001"`

All objects have freejoints for unconstrained dynamics.

YCB mesh assets must be added with absolute paths (e.g., `PROJECT_ROOT / "assets" / "ycb" / ...`) since the programmatically assembled spec won't inherit a `meshdir` compiler directive for YCB meshes.

### Physics

- Timestep: `0.002`
- Gravity: `(0, 0, -9.81)`
- Integrator: `implicitfast`
- Contact parameters carried over from `ycb_resting.xml`

### Viewer Loop

- Compile model, create data, reset to `home` keyframe
- Launch `mujoco.viewer.launch_passive`
- Step physics in a loop so objects settle on the table
- No control input — arm stays at home pose

## File Structure

```
scenes/
  xarm_leap_scene.py    # New script (this design)
  ycb_resting.xml       # Existing scene (unchanged)
```

## Dependencies

- `mujoco` (already in project)
- `loop_rate_limiters` (already used by mink examples, may need to add to project deps)
- Mink submodule must be initialized (`git submodule update --init`)

## Future Extensions

- Add hard-coded grasp trajectories (next task, starting with tomato soup can)
- Swap objects programmatically for different grasp experiments
- Add IK control via mink for arm positioning
