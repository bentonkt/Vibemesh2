#!/usr/bin/env python3
"""Minimal test scene: xArm + LEAP hand + one YCB object on a table.

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

TABLE_Z = 0.42

# Hand proxy geom specs: (parent_body, geom_name, type, size, pos)
PROXY_GEOMS = [
    ("leap_right/palm_lower", "leap_right/proxy_palm", mujoco.mjtGeom.mjGEOM_BOX,
     (0.035, 0.050, 0.020), (-0.0486, -0.0371, -0.0117)),
    ("leap_right/fingertip", "leap_right/proxy_tip_1", mujoco.mjtGeom.mjGEOM_SPHERE,
     (0.011,) * 3, (0.0, -0.040, 0.015)),
    ("leap_right/fingertip_2", "leap_right/proxy_tip_2", mujoco.mjtGeom.mjGEOM_SPHERE,
     (0.011,) * 3, (0.0, -0.040, 0.015)),
    ("leap_right/fingertip_3", "leap_right/proxy_tip_3", mujoco.mjtGeom.mjGEOM_SPHERE,
     (0.011,) * 3, (0.0, -0.040, 0.015)),
    ("leap_right/thumb_fingertip", "leap_right/proxy_th_tip", mujoco.mjtGeom.mjGEOM_SPHERE,
     (0.011,) * 3, (0.0, -0.045, -0.015)),
    ("leap_right/dip", "leap_right/proxy_pad_1", mujoco.mjtGeom.mjGEOM_SPHERE,
     (0.010,) * 3, (0.010, -0.020, 0.012)),
    ("leap_right/dip_2", "leap_right/proxy_pad_2", mujoco.mjtGeom.mjGEOM_SPHERE,
     (0.010,) * 3, (0.010, -0.020, 0.012)),
    ("leap_right/dip_3", "leap_right/proxy_pad_3", mujoco.mjtGeom.mjGEOM_SPHERE,
     (0.010,) * 3, (0.010, -0.020, 0.012)),
    ("leap_right/thumb_dip", "leap_right/proxy_th_pad", mujoco.mjtGeom.mjGEOM_SPHERE,
     (0.0085,) * 3, (0.0, 0.0, 0.0)),
]

PROXY_GEOM_NAMES = [name for _, name, *_ in PROXY_GEOMS]

# Hand-object contact pair parameters
HAND_OBJECT_PAIR = {
    "condim": "6",
    "friction": "60 8 2",
    "solref": "0.005 1",
    "solimp": "0.96 0.999 0.003",
    "margin": "0.004",
    "gap": "0.001",
}


def build_robot_spec() -> mujoco.MjSpec:
    """Compose xArm + LEAP hand with proxy collision geoms."""
    arm = mujoco.MjSpec.from_file(_ARM_XML.as_posix())
    hand = mujoco.MjSpec.from_file(_HAND_XML.as_posix())

    # Attach hand to arm
    palm = hand.body("palm_lower")
    palm.quat[:] = (0, 1, 0, 0)
    palm.pos[:] = (0.065, -0.04, 0)
    site = arm.site("attachment_site")
    arm.attach(hand, prefix="leap_right/", site=site)

    # Set table height
    floor = arm.geom("floor")
    if floor is not None:
        floor.pos[2] = TABLE_Z

    # Replace home keyframe
    home_key = arm.key("home")
    arm.delete(home_key)
    arm.add_key(name="home", qpos=HOME_QPOS)

    # Add proxy collision geoms to hand
    for body_name, geom_name, geom_type, size, pos in PROXY_GEOMS:
        body = arm.body(body_name)
        body.add_geom(
            name=geom_name,
            type=geom_type,
            size=size,
            pos=pos,
            contype=2,
            conaffinity=1,
            group=3,
            rgba=(0.1, 0.9, 0.2, 0.2),
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

    # Hand-object contact pairs
    object_geom = f"{object_id}_collision_geom"
    pair_attrs = " ".join(f'{k}="{v}"' for k, v in HAND_OBJECT_PAIR.items())
    pair_lines = "\n".join(
        f'    <pair geom1="{proxy}" geom2="{object_geom}" {pair_attrs}/>'
        for proxy in PROXY_GEOM_NAMES
    )

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

  <contact>
{pair_lines}
  </contact>

  <worldbody>
    <light name="key_light" pos="0 0 3" dir="0 0 -1" directional="true"/>
  </worldbody>
</mujoco>"""

    scene_path = temp_dir / "scene.xml"
    scene_path.write_text(scene_xml, encoding="utf-8")

    model = mujoco.MjModel.from_xml_path(scene_path.as_posix())

    # Disable LEAP mesh collisions (only proxies should collide)
    for geom_id in range(model.ngeom):
        body_name = model.body(int(model.geom_bodyid[geom_id])).name or ""
        geom_name = model.geom(geom_id).name or ""
        if not body_name.startswith("leap_right/"):
            continue
        if geom_name in set(PROXY_GEOM_NAMES):
            continue
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        if int(model.geom_group[geom_id]) != 3:
            continue
        model.geom_contype[geom_id] = 0
        model.geom_conaffinity[geom_id] = 0

    data = mujoco.MjData(model)

    # Set home pose
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)

    # Place object on table
    obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_id)
    if obj_body_id >= 0:
        joint_id = int(model.body_jntadr[obj_body_id])
        qpos_adr = int(model.jnt_qposadr[joint_id])
        data.qpos[qpos_adr:qpos_adr + 3] = [0.4, 0.0, TABLE_Z + 0.10]
        data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]

    mujoco.mj_forward(model, data)
    return model, data, temp_dir


def main():
    parser = argparse.ArgumentParser(description="Test scene: xArm + LEAP + YCB object")
    parser.add_argument("--object", type=str, default="005_tomato_soup_can",
                        help="YCB object ID (default: 005_tomato_soup_can)")
    args = parser.parse_args()

    print(f"Building scene with {args.object}...")
    model, data, temp_dir = build_scene(args.object)
    print(f"Scene loaded: {model.nbody} bodies, {model.ngeom} geoms, {model.njnt} joints")

    try:
        with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
