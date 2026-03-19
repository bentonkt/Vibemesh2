#!/usr/bin/env python3
"""Hardcoded grasp sequence: xArm + LEAP hand picks up a tomato soup can.

A state machine drives the end-effector through waypoints
(pre-grasp -> descend -> close fingers -> lift) using IK for the arm
and direct joint control for the fingers during the close phase.

Usage:
    python scripts/hardcoded_grasp.py
"""

from __future__ import annotations

import enum
import shutil
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

# Ensure the project root is on the path so `scripts.test_scene` resolves
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.test_scene import FINGERS, OBJECT_SPAWN_Z, build_scene  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OBJECT_ID = "005_tomato_soup_can"
FREQ = 200.0
DT = 1.0 / FREQ

ARM_NU = 7  # xArm7 actuators
HAND_NU = 16  # LEAP hand actuators

# Can position (matches build_scene placement)
CAN_XY = np.array([0.4, 0.0])
CAN_TOP_Z = OBJECT_SPAWN_Z + 0.05  # can is ~10 cm tall, centre at spawn z

# Palm-down orientation: rotate 180 deg around X so fingers point down
# quaternion (w, x, y, z) for 180-deg rotation about X
PALM_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])  # wxyz

# Waypoint heights (absolute z)
PRE_GRASP_Z = CAN_TOP_Z + 0.15
GRASP_Z = CAN_TOP_Z + 0.02  # slightly above can top
LIFT_Z = GRASP_Z + 0.15

# Phase transition thresholds
POS_ERROR_THRESH = 0.01  # 1 cm
PHASE_TIMEOUT_STEPS = 1000  # 5 s at 200 Hz

# Finger closed targets (radians)
# fmt: off
FINGER_CLOSED = np.array([
    1.2, 0.0, 1.0, 1.0,   # finger 1: MCP-flex, spread, DIP, tip
    1.2, 0.0, 1.0, 1.0,   # finger 2: MCP-flex, spread, DIP, tip
    1.2, 0.0, 1.0, 1.0,   # finger 3: MCP-flex, spread, DIP, tip
    1.0, 1.2, 0.5, 0.8,   # thumb: PIP_4, PIP, DIP, tip
])
# fmt: on

# Settle detection for fingers
SETTLE_VEL_THRESH = 0.05
SETTLE_STEPS_REQUIRED = 100  # 0.5 s at 200 Hz


# ---------------------------------------------------------------------------
# Phase enum
# ---------------------------------------------------------------------------


class Phase(enum.Enum):
    PRE_GRASP = "pre_grasp"
    DESCEND = "descend"
    CLOSE = "close"
    LIFT = "lift"
    DONE = "done"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_waypoint(x: float, y: float, z: float) -> np.ndarray:
    return np.array([x, y, z])


def _ee_pos_error(model: mujoco.MjModel, data: mujoco.MjData, target_pos: np.ndarray) -> float:
    """Position error between attachment_site and target."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    ee_pos = data.site_xpos[site_id].copy()
    return float(np.linalg.norm(ee_pos - target_pos))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"Building scene with {OBJECT_ID}...")
    model, data, temp_dir = build_scene(OBJECT_ID)
    print(f"Scene loaded: {model.nbody} bodies, {model.ngeom} geoms, {model.njnt} joints")

    # Sanity check: 7 arm + 16 hand = 23 actuators
    assert model.nu == 23, f"Expected 23 actuators, got {model.nu}"

    robot_nu = model.nu

    # --- IK setup ---
    configuration = mink.Configuration(model)
    ee_task = mink.FrameTask(
        frame_name="attachment_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=5e-2)

    finger_tasks = []
    for finger in FINGERS:
        finger_tasks.append(
            mink.RelativeFrameTask(
                frame_name=f"leap_right/{finger}",
                frame_type="site",
                root_name="leap_right/palm_lower",
                root_type="body",
                position_cost=1.0,
                orientation_cost=0.0,
                lm_damping=1e-3,
            )
        )

    limits = [mink.ConfigurationLimit(model=model)]

    # --- Waypoints ---
    waypoints = {
        Phase.PRE_GRASP: _make_waypoint(CAN_XY[0], CAN_XY[1], PRE_GRASP_Z),
        Phase.DESCEND: _make_waypoint(CAN_XY[0], CAN_XY[1], GRASP_Z),
        Phase.CLOSE: _make_waypoint(CAN_XY[0], CAN_XY[1], GRASP_Z),
        Phase.LIFT: _make_waypoint(CAN_XY[0], CAN_XY[1], LIFT_Z),
        Phase.DONE: _make_waypoint(CAN_XY[0], CAN_XY[1], LIFT_Z),
    }

    try:
        with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)

            # Initialize IK from home pose
            configuration.update(data.qpos)
            posture_task.set_target_from_configuration(configuration)

            # Place mocap EE target at current end-effector pose
            mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")
            for finger in FINGERS:
                mink.move_mocap_to_frame(
                    model, data, f"{finger}_target", f"leap_right/{finger}", "site"
                )

            T_palm_prev = configuration.get_transform_frame_to_world(
                "leap_right/palm_lower", "body"
            )

            # Set initial ctrl to home position
            data.ctrl[:robot_nu] = data.qpos[:robot_nu]

            # --- State machine ---
            phase = Phase.PRE_GRASP
            phase_steps = 0
            settle_count = 0

            # Move mocap target to first waypoint with palm-down orientation
            mocap_id_ee = model.body("target").mocapid[0]
            data.mocap_pos[mocap_id_ee] = waypoints[phase]
            data.mocap_quat[mocap_id_ee] = PALM_DOWN_QUAT

            print(f"Phase: {phase.value}")

            rate = RateLimiter(frequency=FREQ, warn=False)
            while viewer.is_running():
                # Determine which tasks to use based on phase
                use_finger_ik = phase in (Phase.PRE_GRASP, Phase.DESCEND)

                if use_finger_ik:
                    active_tasks = [ee_task, posture_task, *finger_tasks]
                else:
                    active_tasks = [ee_task, posture_task]

                # --- IK solve ---
                configuration.update(data.qpos)

                # Update EE task from mocap target
                T_wt = mink.SE3.from_mocap_name(model, data, "target")
                ee_task.set_target(T_wt)

                # Update palm tracking (always, to prevent stale delta)
                T_palm = configuration.get_transform_frame_to_world(
                    "leap_right/palm_lower", "body"
                )
                T_delta = T_palm @ T_palm_prev.inverse()
                T_palm_prev = T_palm.copy()

                if use_finger_ik:
                    # Move finger mocap targets with palm
                    for finger in FINGERS:
                        fmocap_id = model.body(f"{finger}_target").mocapid[0]
                        T_w_mocap = mink.SE3.from_mocap_id(data, fmocap_id)
                        T_new = T_delta @ T_w_mocap
                        data.mocap_pos[fmocap_id] = T_new.translation()
                        data.mocap_quat[fmocap_id] = T_new.rotation().wxyz

                    # Update finger tasks from mocap
                    world_to_palm = configuration.get_transform_frame_to_world(
                        "leap_right/palm_lower", "body"
                    ).inverse()
                    for finger, task in zip(FINGERS, finger_tasks):
                        T_world_target = mink.SE3.from_mocap_name(
                            model, data, f"{finger}_target"
                        )
                        task.set_target(world_to_palm @ T_world_target)

                # Solve IK
                vel = mink.solve_ik(
                    configuration, active_tasks, rate.dt, "daqp", damping=1e-3, limits=limits
                )
                configuration.integrate_inplace(vel, rate.dt)

                # Write ctrl outputs
                if use_finger_ik:
                    # Full IK for arm + fingers
                    data.ctrl[:robot_nu] = configuration.q[:robot_nu]
                else:
                    # IK only for arm; direct control for fingers
                    data.ctrl[:ARM_NU] = configuration.q[:ARM_NU]
                    data.ctrl[ARM_NU:] = FINGER_CLOSED

                # --- Phase transitions ---
                phase_steps += 1
                target_pos = waypoints[phase]
                pos_err = _ee_pos_error(model, data, target_pos)
                reached = pos_err < POS_ERROR_THRESH
                timed_out = phase_steps >= PHASE_TIMEOUT_STEPS

                next_phase = phase
                if phase == Phase.PRE_GRASP:
                    if reached or timed_out:
                        next_phase = Phase.DESCEND
                elif phase == Phase.DESCEND:
                    if reached or timed_out:
                        next_phase = Phase.CLOSE
                elif phase == Phase.CLOSE:
                    # Check finger settle
                    finger_vels = data.qvel[ARM_NU : ARM_NU + HAND_NU]
                    if np.max(np.abs(finger_vels)) < SETTLE_VEL_THRESH:
                        settle_count += 1
                    else:
                        settle_count = 0
                    if settle_count >= SETTLE_STEPS_REQUIRED or timed_out:
                        next_phase = Phase.LIFT
                        settle_count = 0
                elif phase == Phase.LIFT:
                    if reached or timed_out:
                        next_phase = Phase.DONE

                if next_phase != phase:
                    phase = next_phase
                    phase_steps = 0
                    print(f"Phase: {phase.value} (pos_err={pos_err:.4f})")

                    # Update mocap target for new phase
                    data.mocap_pos[mocap_id_ee] = waypoints[phase]
                    data.mocap_quat[mocap_id_ee] = PALM_DOWN_QUAT

                # Step physics
                mujoco.mj_step(model, data)

                viewer.sync()
                rate.sleep()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
