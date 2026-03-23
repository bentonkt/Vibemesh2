#!/usr/bin/env python3
"""Minimal test scene: xArm + LEAP hand + one YCB object.

The arm follows the red mocap box via IK (drag it in the viewer).

Usage:
    python scripts/test_scene.py
    python scripts/test_scene.py --object 006_mustard_bottle
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MINK_ROOT = PROJECT_ROOT / "mink"

_ARM_XML = MINK_ROOT / "examples" / "ufactory_xarm7" / "scene.xml"
_HAND_XML = MINK_ROOT / "examples" / "leap_hand" / "right_hand.xml"
_XARM_ASSETS = MINK_ROOT / "examples" / "ufactory_xarm7" / "assets"
_LEAP_ASSETS = MINK_ROOT / "examples" / "leap_hand" / "assets"

# fmt: off
HOME_QPOS = [
    # xarm (7 joints)
    0, -0.247, 0, 0.909, 0, 1.15644, 0,
    # leap (16 joints)
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
]
# fmt: on

OBJECT_SPAWN_Z = 0.12  # above the floor plane at z=0

FINGERS = ("tip_1", "tip_2", "tip_3", "th_tip")

# Contact properties applied to LEAP mesh geoms and the object geom post-load.
# Without explicit pairs, MuJoCo uses sqrt(f1*f2) for friction, so setting both
# sides to the same value gives that value as the effective friction.
MESH_CONTACT = {
    "friction": np.array([8.0, 0.1, 0.01]),
    "condim": 6,
    "solref": np.array([0.01, 1.0]),
    "solimp": np.array([0.96, 0.999, 0.006, 0.5, 2.0]),
}


def build_robot_spec() -> mujoco.MjSpec:
    """Compose xArm + LEAP hand with finger mocap targets."""
    arm = mujoco.MjSpec.from_file(_ARM_XML.as_posix())
    hand = mujoco.MjSpec.from_file(_HAND_XML.as_posix())

    # Attach hand to arm
    palm = hand.body("palm_lower")
    palm.quat[:] = (0, 1, 0, 0)
    palm.pos[:] = (0.065, -0.04, 0)
    site = arm.site("attachment_site")
    arm.attach(hand, prefix="leap_right/", site=site)

    # Replace home keyframe
    home_key = arm.key("home")
    arm.delete(home_key)
    arm.add_key(name="home", qpos=HOME_QPOS)

    # Add finger mocap targets
    for finger in FINGERS:
        body = arm.worldbody.add_body(name=f"{finger}_target", mocap=True)
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(0.012,) * 3,
            contype=0, conaffinity=0, rgba=(0.2, 0.75, 0.25, 0.7),
        )

    return arm


def build_scene(object_id: str) -> tuple[mujoco.MjModel, mujoco.MjData, Path]:
    """Build the full scene: robot + object in a temp directory."""
    object_xml = PROJECT_ROOT / "mjcf" / "objects" / "ycb" / f"{object_id}.xml"
    if not object_xml.exists():
        raise FileNotFoundError(f"Object MJCF not found: {object_xml}")

    temp_dir = Path(tempfile.mkdtemp(prefix="vibemesh_scene_"))

    # Write robot XML
    robot_spec = build_robot_spec()
    robot_xml_path = temp_dir / "robot.xml"
    robot_xml_path.write_text(robot_spec.to_xml(), encoding="utf-8")

    # Copy robot mesh assets
    assets_dir = temp_dir / "assets"
    assets_dir.mkdir()
    for source_dir in (_XARM_ASSETS, _LEAP_ASSETS):
        for f in source_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, assets_dir / f.name)

    # Compose scene XML
    scene_xml = f"""\
<mujoco model="vibemesh_test_scene">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="0.001" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic"
    iterations="200" ls_iterations="80" impratio="100" noslip_iterations="50">
    <flag multiccd="enable"/>
  </option>

  <include file="{robot_xml_path.resolve()}"/>
  <include file="{object_xml.resolve()}"/>

  <worldbody>
    <light name="key_light" pos="0 0 3" dir="0 0 -1" directional="true"/>
  </worldbody>
</mujoco>"""

    scene_path = temp_dir / "scene.xml"
    scene_path.write_text(scene_xml, encoding="utf-8")

    model = mujoco.MjModel.from_xml_path(scene_path.as_posix())

    # Apply contact properties to LEAP hand mesh collision geoms
    for geom_id in range(model.ngeom):
        body_name = model.body(int(model.geom_bodyid[geom_id])).name or ""
        if not body_name.startswith("leap_right/"):
            continue
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        if int(model.geom_group[geom_id]) != 3:
            continue
        model.geom_condim[geom_id] = MESH_CONTACT["condim"]
        model.geom_friction[geom_id, :3] = MESH_CONTACT["friction"]
        model.geom_solref[geom_id, :2] = MESH_CONTACT["solref"]
        model.geom_solimp[geom_id, :5] = MESH_CONTACT["solimp"]

    # Match object geom friction so geometric mean equals the target value
    obj_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, f"{object_id}_collision_geom"
    )
    if obj_geom_id >= 0:
        model.geom_condim[obj_geom_id] = MESH_CONTACT["condim"]
        model.geom_friction[obj_geom_id, :3] = MESH_CONTACT["friction"]
        model.geom_solref[obj_geom_id, :2] = MESH_CONTACT["solref"]
        model.geom_solimp[obj_geom_id, :5] = MESH_CONTACT["solimp"]

    data = mujoco.MjData(model)

    # Set home pose
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)

    # Place object in front of the arm
    obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_id)
    if obj_body_id >= 0:
        joint_id = int(model.body_jntadr[obj_body_id])
        qpos_adr = int(model.jnt_qposadr[joint_id])
        data.qpos[qpos_adr:qpos_adr + 3] = [0.4, 0.0, OBJECT_SPAWN_Z]
        data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]

    mujoco.mj_forward(model, data)
    return model, data, temp_dir


def main():
    parser = argparse.ArgumentParser(description="Test scene: xArm + LEAP + YCB object")
    parser.add_argument("--object", type=str, default="005_tomato_soup_can")
    args = parser.parse_args()

    print(f"Building scene with {args.object}...")
    model, data, temp_dir = build_scene(args.object)
    print(f"Scene loaded: {model.nbody} bodies, {model.ngeom} geoms, {model.njnt} joints")

    # Count robot actuators (for writing ctrl targets)
    robot_nu = model.nu

    # IK configuration runs on a *copy* — we only read desired joint positions from it
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

            # Initialize IK configuration from home pose
            configuration.update(data.qpos)
            posture_task.set_target_from_configuration(configuration)

            # Place mocap target at current end-effector pose
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

            rate = RateLimiter(frequency=200.0, warn=False)
            while viewer.is_running():
                # Sync IK configuration from actual physics state
                configuration.update(data.qpos)

                # Update end-effector task from mocap target
                T_wt = mink.SE3.from_mocap_name(model, data, "target")
                ee_task.set_target(T_wt)

                # Move finger mocap targets with the palm (before reading them)
                T_palm = configuration.get_transform_frame_to_world(
                    "leap_right/palm_lower", "body"
                )
                T_delta = T_palm @ T_palm_prev.inverse()
                T_palm_prev = T_palm.copy()
                for finger in FINGERS:
                    mocap_id = model.body(f"{finger}_target").mocapid[0]
                    T_w_mocap = mink.SE3.from_mocap_id(data, mocap_id)
                    T_new = T_delta @ T_w_mocap
                    data.mocap_pos[mocap_id] = T_new.translation()
                    data.mocap_quat[mocap_id] = T_new.rotation().wxyz

                # Update finger tasks: read mocap poses from data, convert to palm-relative
                world_to_palm = configuration.get_transform_frame_to_world(
                    "leap_right/palm_lower", "body"
                ).inverse()
                for finger, task in zip(FINGERS, finger_tasks):
                    T_world_target = mink.SE3.from_mocap_name(
                        model, data, f"{finger}_target"
                    )
                    task.set_target(world_to_palm @ T_world_target)

                # Solve IK for desired joint positions
                vel = mink.solve_ik(
                    configuration, tasks, rate.dt, "daqp", damping=1e-3, limits=limits
                )
                configuration.integrate_inplace(vel, rate.dt)

                # Write IK result as actuator targets (position servos)
                data.ctrl[:robot_nu] = configuration.q[:robot_nu]

                # Step physics
                mujoco.mj_step(model, data)

                viewer.sync()
                rate.sleep()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
