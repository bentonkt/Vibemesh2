#!/usr/bin/env python3
"""Interactive grasp scene: use arrow keys to open/close the LEAP hand.

↓ / ↑:  close / open grasp
I / K:  move hand forward / back
J / L:  move hand left / right
U / O:  move hand up / down
, / .:  roll hand left / right

Usage:
    python scripts/arrow_key_grasp.py
"""

from __future__ import annotations

import argparse
import ctypes
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

ARM_NU = 7
HAND_NU = 16

# fmt: off
HOME_ARM_QPOS = np.array([0, -0.247, 0, 0.909, 0, 1.15644, 0])

# Thumb pre-positioned opposite the three fingers (PIP_4 abducted), fingers open
HAND_OPEN_PRE = np.array([
    0.0, 0.0, 0.0, 0.0,   # finger 1: fully open
    0.0, 0.0, 0.0, 0.0,   # finger 2: fully open
    0.0, 0.0, 0.0, 0.0,   # finger 3: fully open
    1.9, 0.0, 0.0, 0.0,   # thumb: PIP_4=1.9 (fully opposed), rest open
])

FINGER_CLOSED = np.array([
    1.2, 0.0, 1.0, 1.0,   # finger 1: MCP-flex, spread, DIP, tip
    1.2, 0.0, 1.0, 1.0,   # finger 2: MCP-flex, spread, DIP, tip
    1.2, 0.0, 1.0, 1.0,   # finger 3: MCP-flex, spread, DIP, tip
    1.9, 1.2, 0.5, 0.8,   # thumb: PIP_4 (fully opposed), PIP, DIP, tip
])
# fmt: on

GRASP_SPEED = 1.5  # full open→close in ~0.67 s
MOVE_SPEED = 0.3   # m/s for hand translation
ROLL_SPEED = 1.0   # rad/s for hand roll

# Windows virtual-key codes for movement (polled each physics step)
_user32 = ctypes.windll.user32
MOVE_VK: dict[int, np.ndarray] = {
    0x49: np.array([ 1.0,  0.0,  0.0]),  # I — forward
    0x4B: np.array([-1.0,  0.0,  0.0]),  # K — back
    0x4A: np.array([ 0.0,  1.0,  0.0]),  # J — left
    0x4C: np.array([ 0.0, -1.0,  0.0]),  # L — right
    0x55: np.array([ 0.0,  0.0,  1.0]),  # U — up
    0x4F: np.array([ 0.0,  0.0, -1.0]),  # O — down
}
ROLL_VK: dict[int, float] = {
    0xBC: -1.0,  # , — roll left
    0xBE:  1.0,  # . — roll right
}

OBJECT_SPAWN_Z = 0.04  # can center height — low enough to clear the hand at home pose


def _is_held(vk: int) -> bool:
    return bool(_user32.GetAsyncKeyState(vk) & 0x8000)


def _rotate_quat(quat_wxyz: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Return quat_wxyz rotated by angle radians around axis."""
    half = angle / 2.0
    dq = np.array([np.cos(half), *(np.sin(half) * axis)])
    w1, x1, y1, z1 = dq
    w2, x2, y2, z2 = quat_wxyz
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive grasp scene")
    parser.add_argument("--object", default="005_tomato_soup_can",
                        help="YCB object ID to place under the hand (default: 005_tomato_soup_can)")
    args = parser.parse_args()
    OBJECT_ID = args.object

    print(f"Building scene with {OBJECT_ID}...")
    model, data, temp_dir = build_scene(OBJECT_ID)
    print(f"Scene loaded: {model.nbody} bodies, {model.ngeom} geoms")
    print("Controls: ↓/↑ = close/open grasp | IJKL = move horizontal | U/O = move vertical | ,/. = roll")

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
    grasp_target = 0.0  # set by arrow keys

    def key_callback(keycode: int) -> None:
        nonlocal grasp_target
        if keycode == 264:   # GLFW_KEY_DOWN
            grasp_target = 1.0
        elif keycode == 265: # GLFW_KEY_UP
            grasp_target = 0.0

    try:
        with mujoco.viewer.launch_passive(
            model=model, data=data,
            show_left_ui=False, show_right_ui=False,
            key_callback=key_callback,
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)

            configuration.update(data.qpos)
            posture_task.set_target_from_configuration(configuration)
            mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")
            mocap_id = model.body("target").mocapid[0]

            rate = RateLimiter(frequency=200.0, warn=False)
            while viewer.is_running():
                # Smooth grasp_t toward grasp_target
                step = GRASP_SPEED * rate.dt
                diff = grasp_target - grasp_t
                grasp_t = float(np.clip(
                    grasp_t + np.sign(diff) * min(step, abs(diff)), 0.0, 1.0
                ))

                # Translate mocap box from held movement keys (polled via Windows API)
                move_dir = sum(
                    (v for k, v in MOVE_VK.items() if _is_held(k)), np.zeros(3)
                )
                if np.any(move_dir):
                    data.mocap_pos[mocap_id] += move_dir * MOVE_SPEED * rate.dt

                # Roll hand around world X-axis (, / . keys)
                roll = sum(v for k, v in ROLL_VK.items() if _is_held(k))
                if roll:
                    data.mocap_quat[mocap_id] = _rotate_quat(
                        data.mocap_quat[mocap_id],
                        np.array([1.0, 0.0, 0.0]),
                        roll * ROLL_SPEED * rate.dt,
                    )

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
