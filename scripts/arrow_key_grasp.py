#!/usr/bin/env python3
"""Interactive grasp scene: use arrow keys to open/close the LEAP hand.

Up arrow:   open grasp
Down arrow: close grasp

Usage:
    python scripts/arrow_key_grasp.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.test_scene import build_scene  # noqa: E402

OBJECT_ID = "005_tomato_soup_can"

ARM_NU = 7
HAND_NU = 16

# fmt: off
HOME_ARM_QPOS = np.array([0, -0.247, 0, 0.909, 0, 1.15644, 0])

# Thumb pre-positioned opposite the three fingers (PIP_4 abducted), fingers open
HAND_OPEN_PRE = np.array([
    0.0, 0.0, 0.0, 0.0,   # finger 1: fully open
    0.0, 0.0, 0.0, 0.0,   # finger 2: fully open
    0.0, 0.0, 0.0, 0.0,   # finger 3: fully open
    1.0, 0.0, 0.0, 0.0,   # thumb: PIP_4=1.0 (opposed), rest open
])

FINGER_CLOSED = np.array([
    1.2, 0.0, 1.0, 1.0,   # finger 1: MCP-flex, spread, DIP, tip
    1.2, 0.0, 1.0, 1.0,   # finger 2: MCP-flex, spread, DIP, tip
    1.2, 0.0, 1.0, 1.0,   # finger 3: MCP-flex, spread, DIP, tip
    1.0, 1.2, 0.5, 0.8,   # thumb: PIP_4, PIP, DIP, tip
])
# fmt: on

GRASP_SPEED = 1.5  # full open→close in ~0.67 s

# GLFW keycodes
KEY_UP = 265
KEY_DOWN = 264

OBJECT_SPAWN_Z = 0.12  # can center height above floor


def main() -> None:
    print(f"Building scene with {OBJECT_ID}...")
    model, data, temp_dir = build_scene(OBJECT_ID)
    print(f"Scene loaded: {model.nbody} bodies, {model.ngeom} geoms")
    print("Controls: Down arrow = close grasp | Up arrow = open grasp")

    # Set arm to home and thumb to pre-opposed position, then forward kinematics
    data.qpos[:ARM_NU] = HOME_ARM_QPOS
    data.qpos[ARM_NU:ARM_NU + HAND_NU] = HAND_OPEN_PRE
    data.ctrl[:ARM_NU] = HOME_ARM_QPOS
    data.ctrl[ARM_NU:ARM_NU + HAND_NU] = HAND_OPEN_PRE
    mujoco.mj_forward(model, data)

    # Place can directly below the hand (same x,y as attachment_site)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    hand_xy = data.site_xpos[site_id][:2].copy()
    obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_ID)
    if obj_body_id >= 0:
        joint_id = int(model.body_jntadr[obj_body_id])
        qpos_adr = int(model.jnt_qposadr[joint_id])
        data.qpos[qpos_adr:qpos_adr + 3] = [hand_xy[0], hand_xy[1], OBJECT_SPAWN_Z]
        data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    # IK setup (arm only — hand is controlled by arrow keys)
    configuration = mink.Configuration(model)
    ee_task = mink.FrameTask(
        frame_name="attachment_site", frame_type="site",
        position_cost=1.0, orientation_cost=1.0, lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=5e-2)
    limits = [mink.ConfigurationLimit(model=model)]

    grasp_t = 0.0       # 0 = open, 1 = closed
    grasp_target = 0.0  # updated by key presses

    def key_callback(keycode: int) -> None:
        nonlocal grasp_target
        if keycode == KEY_DOWN:
            grasp_target = 1.0
        elif keycode == KEY_UP:
            grasp_target = 0.0

    try:
        with mujoco.viewer.launch_passive(
            model=model, data=data,
            show_left_ui=False, show_right_ui=False,
            key_callback=key_callback,
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)

            # Initialize IK configuration from current pose
            configuration.update(data.qpos)
            posture_task.set_target_from_configuration(configuration)

            # Place red mocap box at the current end-effector pose
            mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")

            rate = RateLimiter(frequency=200.0, warn=False)
            while viewer.is_running():
                # Smooth grasp_t toward grasp_target
                step = GRASP_SPEED * rate.dt
                diff = grasp_target - grasp_t
                grasp_t = float(np.clip(grasp_t + np.sign(diff) * min(step, abs(diff)), 0.0, 1.0))

                # Solve IK: arm follows the red mocap box
                configuration.update(data.qpos)
                T_wt = mink.SE3.from_mocap_name(model, data, "target")
                ee_task.set_target(T_wt)
                vel = mink.solve_ik(
                    configuration, [ee_task, posture_task], rate.dt,
                    "daqp", damping=1e-3, limits=limits,
                )
                configuration.integrate_inplace(vel, rate.dt)

                # Arm follows IK, hand follows arrow key grasp
                data.ctrl[:ARM_NU] = configuration.q[:ARM_NU]
                data.ctrl[ARM_NU:ARM_NU + HAND_NU] = (
                    (1.0 - grasp_t) * HAND_OPEN_PRE + grasp_t * FINGER_CLOSED
                )

                mujoco.mj_step(model, data)
                viewer.sync()
                rate.sleep()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
