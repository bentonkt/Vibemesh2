#!/usr/bin/env python3
"""Hardcoded grasp sequence: xArm + LEAP hand picks up a tomato soup can.

A state machine drives the end-effector through waypoints
(settle -> pre-grasp -> approach -> close fingers -> lift) using IK for the arm
and direct joint control for the fingers during the close and lift phases.

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
import trimesh
from loop_rate_limiters import RateLimiter

import mink

# Ensure the project root is on the path so `scripts.test_scene` resolves
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.test_scene import FINGERS, build_scene  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OBJECT_ID = "005_tomato_soup_can"
COLLISION_MESH_PATH = PROJECT_ROOT / "data" / "ycb" / "processed" / OBJECT_ID / "collision.obj"

FREQ = 1000.0
DT = 1.0 / FREQ

ARM_NU = 7  # xArm7 actuators
HAND_NU = 16  # LEAP hand actuators

WORLD_Y_AXIS = np.array([0.0, 1.0, 0.0])
HAND_OPEN = np.zeros(HAND_NU)

SETTLE_STEPS = 4000  # 4 s at 1000 Hz
APPROACH_OFFSET = 0.06
PRE_GRASP_Z_OFFSET = 0.05
LIFT_Z_OFFSET = 0.12
DROP_STEPS = 2000  # 2 s at 1000 Hz
SEAT_BIAS_STEPS = (0.0, 0.005, 0.010, 0.015)

PRIMARY_FINGER_ANCHOR_PROXIES = (
    "leap_right/proxy_pad_1",
    "leap_right/proxy_pad_2",
    "leap_right/proxy_pad_3",
)
PRIMARY_THUMB_ANCHOR_PROXY = "leap_right/proxy_th_tip"
FINGER_CONTACT_PROXIES = PRIMARY_FINGER_ANCHOR_PROXIES + (
    "leap_right/proxy_tip_1",
    "leap_right/proxy_tip_2",
    "leap_right/proxy_tip_3",
)
THUMB_CONTACT_PROXIES = (
    "leap_right/proxy_th_tip",
    "leap_right/proxy_th_pad",
)
SEAT_CONTACT_PROXIES = FINGER_CONTACT_PROXIES + THUMB_CONTACT_PROXIES + (
    "leap_right/proxy_palm",
)

# Phase transition thresholds
POS_ERROR_THRESH = 0.01  # 1 cm
ORI_ERROR_THRESH = 0.1  # radians
PHASE_TIMEOUT_STEPS = 5000  # 5 s at 1000 Hz

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
SETTLE_STEPS_REQUIRED = 500  # 0.5 s at 1000 Hz


# ---------------------------------------------------------------------------
# Phase enum
# ---------------------------------------------------------------------------


class Phase(enum.Enum):
    SETTLE = "settle"
    PRE_GRASP = "pre_grasp"
    APPROACH = "approach"
    SEAT = "seat"
    CLOSE = "close"
    LIFT = "lift"
    DROP = "drop"
    DONE = "done"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        raise ValueError(f"Cannot normalize near-zero vector: {vec}")
    return vec / norm


def _project_orthogonal(vec: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return vec - np.dot(vec, axis) * axis


def _rotation_matrix_to_quat(rotation: np.ndarray) -> np.ndarray:
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, rotation.reshape(-1))
    return quat


def _format_vec(vec: np.ndarray) -> str:
    return np.array2string(vec, precision=4, suppress_small=True)


def _make_waypoints(
    base_close_pos: np.ndarray, z_axis: np.ndarray, seat_bias: float
) -> dict[Phase, np.ndarray]:
    seated_close_pos = base_close_pos + seat_bias * z_axis
    return {
        Phase.PRE_GRASP: base_close_pos - APPROACH_OFFSET * z_axis + np.array([0.0, 0.0, PRE_GRASP_Z_OFFSET]),
        Phase.APPROACH: base_close_pos - APPROACH_OFFSET * z_axis,
        Phase.SEAT: seated_close_pos.copy(),
        Phase.CLOSE: seated_close_pos.copy(),
        Phase.LIFT: seated_close_pos + np.array([0.0, 0.0, LIFT_Z_OFFSET]),
    }


def _site_quat(model: mujoco.MjModel, data: mujoco.MjData, site_name: str) -> np.ndarray:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    return _rotation_matrix_to_quat(data.site_xmat[site_id].reshape(3, 3))


def _ee_pos_error(model: mujoco.MjModel, data: mujoco.MjData, target_pos: np.ndarray) -> float:
    """Position error between attachment_site and target."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    ee_pos = data.site_xpos[site_id].copy()
    return float(np.linalg.norm(ee_pos - target_pos))


def _ee_ori_error(model: mujoco.MjModel, data: mujoco.MjData, target_quat: np.ndarray) -> float:
    """Orientation error angle between attachment_site and target quaternion."""
    ee_quat = _site_quat(model, data, "attachment_site")
    dot = float(np.clip(np.abs(np.dot(ee_quat, target_quat)), 0.0, 1.0))
    return float(2.0 * np.arccos(dot))


def _load_collision_mesh_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(path, force="mesh")
    mins, maxs = mesh.bounds
    center_local = 0.5 * (mins + maxs)
    extents = maxs - mins
    return center_local.astype(float), extents.astype(float)


def _geom_center_local(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    attachment_pos: np.ndarray,
    attachment_rot: np.ndarray,
    geom_name: str,
) -> np.ndarray:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    return attachment_rot.T @ (data.geom_xpos[geom_id].copy() - attachment_pos)


def _compute_contact_anchor_local(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Compute a can-contact anchor from the hand proxy geoms in closed pose."""
    saved_qpos = data.qpos.copy()
    saved_qvel = data.qvel.copy()
    saved_time = float(data.time)

    data.qpos[ARM_NU : ARM_NU + HAND_NU] = FINGER_CLOSED
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    attachment_pos = data.site_xpos[site_id].copy()
    attachment_rot = data.site_xmat[site_id].reshape(3, 3).copy()

    finger_pad_center = np.mean(
        np.stack(
            [
                _geom_center_local(model, data, attachment_pos, attachment_rot, geom_name)
                for geom_name in PRIMARY_FINGER_ANCHOR_PROXIES
            ],
            axis=0,
        ),
        axis=0,
    )
    thumb_center = _geom_center_local(
        model, data, attachment_pos, attachment_rot, PRIMARY_THUMB_ANCHOR_PROXY
    )
    contact_anchor_local = 0.5 * (finger_pad_center + thumb_center)

    data.qpos[:] = saved_qpos
    data.qvel[:] = saved_qvel
    data.time = saved_time
    mujoco.mj_forward(model, data)
    return contact_anchor_local


def _classify_can_contacts(
    model: mujoco.MjModel, data: mujoco.MjData, object_id: str = OBJECT_ID
) -> tuple[bool, bool, bool, tuple[str, ...]]:
    can_geom = f"{object_id}_collision_geom"
    contact_names = set()

    for i in range(data.ncon):
        contact = data.contact[i]
        geom1 = model.geom(contact.geom1).name or ""
        geom2 = model.geom(contact.geom2).name or ""
        if can_geom == geom1 and geom2 in SEAT_CONTACT_PROXIES:
            contact_names.add(geom2)
        elif can_geom == geom2 and geom1 in SEAT_CONTACT_PROXIES:
            contact_names.add(geom1)

    thumb_contact = any(name in THUMB_CONTACT_PROXIES for name in contact_names)
    finger_contact = any(name in FINGER_CONTACT_PROXIES for name in contact_names)
    seat_contact = any(name in SEAT_CONTACT_PROXIES for name in contact_names)
    return thumb_contact, finger_contact, seat_contact, tuple(sorted(contact_names))


def _compute_target_frame(can_axis_world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_axis = _normalize(can_axis_world)
    z_axis = _project_orthogonal(-WORLD_Y_AXIS, y_axis)
    if np.linalg.norm(z_axis) <= 1e-8:
        z_axis = _project_orthogonal(np.array([1.0, 0.0, 0.0]), y_axis)
    z_axis = _normalize(z_axis)
    x_axis = _normalize(np.cross(y_axis, z_axis))
    return x_axis, y_axis, z_axis


def _compute_grasp_plan(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    can_center_local: np.ndarray,
    contact_anchor_local: np.ndarray,
) -> tuple[dict[Phase, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_ID)
    body_pos = data.xpos[object_body_id].copy()
    body_rot = data.xmat[object_body_id].reshape(3, 3).copy()

    can_center_world = body_pos + body_rot @ can_center_local
    can_axis_world = _normalize(body_rot[:, 2])
    x_axis, y_axis, z_axis = _compute_target_frame(can_axis_world)
    target_rot = np.column_stack((x_axis, y_axis, z_axis))
    target_quat = _rotation_matrix_to_quat(target_rot)

    base_close_pos = can_center_world - target_rot @ contact_anchor_local
    waypoints = _make_waypoints(base_close_pos, z_axis, SEAT_BIAS_STEPS[0])

    print(f"Settled can center: {_format_vec(can_center_world)}")
    print(f"Can axis: {_format_vec(can_axis_world)}")
    print(f"Target frame x: {_format_vec(x_axis)}")
    print(f"Target frame y: {_format_vec(y_axis)}")
    print(f"Target frame z: {_format_vec(z_axis)}")
    print(f"Contact anchor (attachment frame): {_format_vec(contact_anchor_local)}")
    print(f"Base close pose: {_format_vec(base_close_pos)}")
    print(f"Seat bias: {SEAT_BIAS_STEPS[0]:.3f}")
    for phase in (Phase.PRE_GRASP, Phase.APPROACH, Phase.SEAT, Phase.CLOSE, Phase.LIFT):
        print(f"{phase.value} waypoint: {_format_vec(waypoints[phase])}")

    return waypoints, target_quat, base_close_pos, z_axis


# ---------------------------------------------------------------------------
# Headless initial grasp (for use by GraspEnv)
# ---------------------------------------------------------------------------


def execute_initial_grasp(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_id: str = OBJECT_ID,
    settle_steps: int = 800,
) -> None:
    """Run SETTLE → PRE_GRASP → APPROACH → SEAT → CLOSE headlessly.

    Drives the arm via IK to approach the object, seats the hand against it,
    and closes the fingers until contact settles. No viewer, no rate limiter.
    Modifies ``data`` in place; ``model`` is read-only.
    """
    collision_path = (
        PROJECT_ROOT / "data" / "ycb" / "processed" / object_id / "collision.obj"
    )
    can_center_local, _ = _load_collision_mesh_bounds(collision_path)
    contact_anchor_local = _compute_contact_anchor_local(model, data)

    # -- IK setup (mirrors main) --
    configuration = mink.Configuration(model)
    ee_task = mink.FrameTask(
        frame_name="attachment_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=5e-2)
    finger_tasks = [
        mink.RelativeFrameTask(
            frame_name=f"leap_right/{finger}",
            frame_type="site",
            root_name="leap_right/palm_lower",
            root_type="body",
            position_cost=1.0,
            orientation_cost=0.0,
            lm_damping=1e-3,
        )
        for finger in FINGERS
    ]
    limits = [mink.ConfigurationLimit(model=model)]

    # -- Initialize --
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
    data.ctrl[: model.nu] = data.qpos[: model.nu]

    # -- State machine --
    phase = Phase.SETTLE
    phase_steps = 0
    settle_count = 0
    waypoints: dict[Phase, np.ndarray] = {}
    base_close_pos: np.ndarray | None = None
    z_axis: np.ndarray | None = None
    seat_bias_index = 0
    grasp_contact_seen = False
    two_sided_contact_seen = False

    mocap_id_ee = model.body("target").mocapid[0]
    target_quat = data.mocap_quat[mocap_id_ee].copy()

    while True:
        use_finger_ik = phase in (Phase.SETTLE, Phase.PRE_GRASP, Phase.APPROACH)
        active_tasks = (
            [ee_task, posture_task, *finger_tasks]
            if use_finger_ik
            else [ee_task, posture_task]
        )

        # -- IK solve --
        configuration.update(data.qpos)
        T_wt = mink.SE3.from_mocap_name(model, data, "target")
        ee_task.set_target(T_wt)

        T_palm = configuration.get_transform_frame_to_world(
            "leap_right/palm_lower", "body"
        )
        T_delta = T_palm @ T_palm_prev.inverse()
        T_palm_prev = T_palm.copy()

        if use_finger_ik:
            for finger in FINGERS:
                fmid = model.body(f"{finger}_target").mocapid[0]
                T_w_mocap = mink.SE3.from_mocap_id(data, fmid)
                T_new = T_delta @ T_w_mocap
                data.mocap_pos[fmid] = T_new.translation()
                data.mocap_quat[fmid] = T_new.rotation().wxyz

            world_to_palm = configuration.get_transform_frame_to_world(
                "leap_right/palm_lower", "body"
            ).inverse()
            for finger, task in zip(FINGERS, finger_tasks):
                T_world_target = mink.SE3.from_mocap_name(
                    model, data, f"{finger}_target"
                )
                task.set_target(world_to_palm @ T_world_target)

        vel = mink.solve_ik(
            configuration, active_tasks, DT, "daqp", damping=1e-3, limits=limits
        )
        configuration.integrate_inplace(vel, DT)

        # -- Write ctrl --
        if use_finger_ik:
            data.ctrl[: model.nu] = configuration.q[: model.nu]
        else:
            data.ctrl[:ARM_NU] = configuration.q[:ARM_NU]
            data.ctrl[ARM_NU:] = (
                HAND_OPEN if phase == Phase.SEAT else FINGER_CLOSED
            )

        # -- Phase transitions --
        phase_steps += 1
        target_pos = data.mocap_pos[mocap_id_ee].copy()
        pos_err = _ee_pos_error(model, data, target_pos)
        ori_err = _ee_ori_error(model, data, target_quat)
        thumb_contact, finger_contact, seat_contact, _ = _classify_can_contacts(
            model, data, object_id
        )
        grasp_contact_seen = grasp_contact_seen or seat_contact
        two_sided_contact_seen = two_sided_contact_seen or (
            thumb_contact and finger_contact
        )

        next_phase = phase

        if phase == Phase.SETTLE:
            if phase_steps >= settle_steps:
                seat_bias_index = 0
                waypoints, target_quat, base_close_pos, z_axis = (
                    _compute_grasp_plan(
                        model, data, can_center_local, contact_anchor_local
                    )
                )
                next_phase = Phase.PRE_GRASP

        elif phase == Phase.PRE_GRASP:
            reached = pos_err < POS_ERROR_THRESH and ori_err < ORI_ERROR_THRESH
            if reached or phase_steps >= PHASE_TIMEOUT_STEPS:
                next_phase = Phase.APPROACH

        elif phase == Phase.APPROACH:
            reached = pos_err < POS_ERROR_THRESH and ori_err < ORI_ERROR_THRESH
            if reached or phase_steps >= PHASE_TIMEOUT_STEPS:
                next_phase = Phase.SEAT

        elif phase == Phase.SEAT:
            reached = pos_err < POS_ERROR_THRESH and ori_err < ORI_ERROR_THRESH
            if seat_contact:
                next_phase = Phase.CLOSE
            elif reached and base_close_pos is not None and z_axis is not None:
                if seat_bias_index + 1 < len(SEAT_BIAS_STEPS):
                    seat_bias_index += 1
                    waypoints = _make_waypoints(
                        base_close_pos, z_axis, SEAT_BIAS_STEPS[seat_bias_index]
                    )
                    data.mocap_pos[mocap_id_ee] = waypoints[Phase.SEAT]
                    data.mocap_quat[mocap_id_ee] = target_quat
                    phase_steps = 0
                else:
                    next_phase = Phase.CLOSE
            elif phase_steps >= PHASE_TIMEOUT_STEPS:
                next_phase = Phase.CLOSE

        elif phase == Phase.CLOSE:
            finger_vels = data.qvel[ARM_NU : ARM_NU + HAND_NU]
            close_contact_ready = seat_contact or grasp_contact_seen
            if close_contact_ready:
                if np.max(np.abs(finger_vels)) < SETTLE_VEL_THRESH:
                    settle_count += 1
                else:
                    settle_count = 0
                if settle_count >= SETTLE_STEPS_REQUIRED:
                    break  # Grasp achieved
            else:
                settle_count = 0
                if phase_steps >= PHASE_TIMEOUT_STEPS:
                    break  # Timed out without contact
            if phase_steps >= PHASE_TIMEOUT_STEPS and grasp_contact_seen:
                break  # Timeout with prior contact — proceed anyway

        if next_phase != phase:
            phase = next_phase
            phase_steps = 0
            if phase in waypoints:
                data.mocap_pos[mocap_id_ee] = waypoints[phase]
                data.mocap_quat[mocap_id_ee] = target_quat

        mujoco.mj_step(model, data)


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
    can_center_local, can_extents = _load_collision_mesh_bounds(COLLISION_MESH_PATH)
    contact_anchor_local = _compute_contact_anchor_local(model, data)

    print(f"Can center (local): {_format_vec(can_center_local)}")
    print(f"Can extents (local): {_format_vec(can_extents)}")
    print(f"Contact anchor (attachment frame): {_format_vec(contact_anchor_local)}")

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
            phase = Phase.SETTLE
            phase_steps = 0
            settle_count = 0
            waypoints: dict[Phase, np.ndarray] = {}
            base_close_pos: np.ndarray | None = None
            z_axis: np.ndarray | None = None
            seat_bias_index = 0
            last_contact_names: tuple[str, ...] = ()
            grasp_contact_seen = False
            two_sided_contact_seen = False

            mocap_id_ee = model.body("target").mocapid[0]
            target_quat = data.mocap_quat[mocap_id_ee].copy()

            print(f"Phase: {phase.value}")

            rate = RateLimiter(frequency=FREQ, warn=False)
            while viewer.is_running():
                use_finger_ik = phase in (Phase.SETTLE, Phase.PRE_GRASP, Phase.APPROACH)
                active_tasks = [ee_task, posture_task, *finger_tasks] if use_finger_ik else [ee_task, posture_task]

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
                        finger_mocap_id = model.body(f"{finger}_target").mocapid[0]
                        T_w_mocap = mink.SE3.from_mocap_id(data, finger_mocap_id)
                        T_new = T_delta @ T_w_mocap
                        data.mocap_pos[finger_mocap_id] = T_new.translation()
                        data.mocap_quat[finger_mocap_id] = T_new.rotation().wxyz

                    # Update finger tasks from mocap
                    world_to_palm = configuration.get_transform_frame_to_world(
                        "leap_right/palm_lower", "body"
                    ).inverse()
                    for finger, task in zip(FINGERS, finger_tasks):
                        T_world_target = mink.SE3.from_mocap_name(
                            model, data, f"{finger}_target"
                        )
                        task.set_target(world_to_palm @ T_world_target)

                vel = mink.solve_ik(
                    configuration, active_tasks, rate.dt, "daqp", damping=1e-3, limits=limits
                )
                configuration.integrate_inplace(vel, rate.dt)

                # Write ctrl outputs
                if use_finger_ik:
                    data.ctrl[:robot_nu] = configuration.q[:robot_nu]
                else:
                    data.ctrl[:ARM_NU] = configuration.q[:ARM_NU]
                    data.ctrl[ARM_NU:] = HAND_OPEN if phase in (Phase.SEAT, Phase.DROP) else FINGER_CLOSED

                # --- Phase transitions ---
                phase_steps += 1
                target_pos = data.mocap_pos[mocap_id_ee].copy()
                pos_err = _ee_pos_error(model, data, target_pos)
                ori_err = _ee_ori_error(model, data, target_quat)
                thumb_contact, finger_contact, seat_contact, contact_names = _classify_can_contacts(
                    model, data
                )
                grasp_contact_seen = grasp_contact_seen or seat_contact
                two_sided_contact_seen = two_sided_contact_seen or (thumb_contact and finger_contact)

                if phase in (Phase.SEAT, Phase.CLOSE) and contact_names != last_contact_names:
                    prefix = "SEAT" if phase == Phase.SEAT else "CLOSE"
                    summary = ", ".join(contact_names) if contact_names else "none"
                    print(f"{prefix} contacts: {summary}")
                    last_contact_names = contact_names

                next_phase = phase
                if phase == Phase.SETTLE:
                    if phase_steps >= SETTLE_STEPS:
                        seat_bias_index = 0
                        waypoints, target_quat, base_close_pos, z_axis = _compute_grasp_plan(
                            model, data, can_center_local, contact_anchor_local
                        )
                        next_phase = Phase.PRE_GRASP
                elif phase == Phase.PRE_GRASP:
                    reached = pos_err < POS_ERROR_THRESH and ori_err < ORI_ERROR_THRESH
                    timed_out = phase_steps >= PHASE_TIMEOUT_STEPS
                    if reached or timed_out:
                        if timed_out and not reached:
                            print(
                                f"Warning: {phase.value} timeout "
                                f"(pos_err={pos_err:.4f}, ori_err={ori_err:.4f})"
                            )
                        next_phase = Phase.APPROACH
                elif phase == Phase.APPROACH:
                    reached = pos_err < POS_ERROR_THRESH and ori_err < ORI_ERROR_THRESH
                    timed_out = phase_steps >= PHASE_TIMEOUT_STEPS
                    if reached or timed_out:
                        if timed_out and not reached:
                            print(
                                f"Warning: {phase.value} timeout "
                                f"(pos_err={pos_err:.4f}, ori_err={ori_err:.4f})"
                            )
                        next_phase = Phase.SEAT
                elif phase == Phase.SEAT:
                    reached = pos_err < POS_ERROR_THRESH and ori_err < ORI_ERROR_THRESH
                    timed_out = phase_steps >= PHASE_TIMEOUT_STEPS
                    if seat_contact:
                        next_phase = Phase.CLOSE
                    elif reached and base_close_pos is not None and z_axis is not None:
                        if seat_bias_index + 1 < len(SEAT_BIAS_STEPS):
                            seat_bias_index += 1
                            waypoints = _make_waypoints(
                                base_close_pos, z_axis, SEAT_BIAS_STEPS[seat_bias_index]
                            )
                            data.mocap_pos[mocap_id_ee] = waypoints[Phase.SEAT]
                            data.mocap_quat[mocap_id_ee] = target_quat
                            phase_steps = 0
                            print(f"SEAT bias: {SEAT_BIAS_STEPS[seat_bias_index]:.3f}")
                        else:
                            print(
                                f"Warning: {phase.value} reached max seat bias "
                                f"without contact (pos_err={pos_err:.4f}, ori_err={ori_err:.4f})"
                            )
                            next_phase = Phase.CLOSE
                    elif timed_out:
                        print(
                            f"Warning: {phase.value} timeout "
                            f"(pos_err={pos_err:.4f}, ori_err={ori_err:.4f})"
                        )
                        next_phase = Phase.CLOSE
                elif phase == Phase.CLOSE:
                    timed_out = phase_steps >= PHASE_TIMEOUT_STEPS
                    finger_vels = data.qvel[ARM_NU : ARM_NU + HAND_NU]
                    close_contact_ready = seat_contact or grasp_contact_seen
                    if close_contact_ready:
                        if np.max(np.abs(finger_vels)) < SETTLE_VEL_THRESH:
                            settle_count += 1
                        else:
                            settle_count = 0
                        if settle_count >= SETTLE_STEPS_REQUIRED:
                            if not two_sided_contact_seen:
                                print(
                                    "CLOSE: lifting with seated can contact but without "
                                    "simultaneous two-sided proxy confirmation"
                                )
                            next_phase = Phase.LIFT
                            settle_count = 0
                    else:
                        settle_count = 0
                        if timed_out:
                            print(
                                "Failed grasp: CLOSE timed out without can contact "
                                f"(thumb_contact={thumb_contact}, finger_contact={finger_contact}, "
                                f"seat_contact={seat_contact})"
                            )
                            next_phase = Phase.DONE
                    if timed_out and next_phase == phase and grasp_contact_seen:
                        print(
                            "Warning: CLOSE timed out after seated can contact; "
                            "advancing to lift anyway"
                        )
                        if not two_sided_contact_seen:
                            print(
                                "CLOSE: timeout fallback is using seated contact without "
                                "simultaneous two-sided proxy confirmation"
                            )
                        next_phase = Phase.LIFT
                        settle_count = 0
                elif phase == Phase.LIFT:
                    reached = pos_err < POS_ERROR_THRESH
                    timed_out = phase_steps >= PHASE_TIMEOUT_STEPS
                    if reached or timed_out:
                        if timed_out and not reached:
                            print(
                                f"Warning: {phase.value} timeout "
                                f"(pos_err={pos_err:.4f}, ori_err={ori_err:.4f})"
                            )
                        next_phase = Phase.DROP
                elif phase == Phase.DROP:
                    if phase_steps >= DROP_STEPS:
                        next_phase = Phase.DONE

                if next_phase != phase:
                    phase = next_phase
                    phase_steps = 0
                    last_contact_names = ()

                    if phase != Phase.DONE and phase in waypoints:
                        data.mocap_pos[mocap_id_ee] = waypoints[phase]
                        data.mocap_quat[mocap_id_ee] = target_quat

                    if phase in (
                        Phase.PRE_GRASP,
                        Phase.APPROACH,
                        Phase.SEAT,
                        Phase.CLOSE,
                        Phase.LIFT,
                        Phase.DROP,
                        Phase.DONE,
                    ):
                        print(
                            f"Phase: {phase.value} "
                            f"(pos_err={pos_err:.4f}, ori_err={ori_err:.4f})"
                        )
                    else:
                        print(f"Phase: {phase.value}")

                # Step physics
                mujoco.mj_step(model, data)

                viewer.sync()
                rate.sleep()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
