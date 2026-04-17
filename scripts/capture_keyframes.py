#!/usr/bin/env python3
"""Capture grasp keyframes interactively for headless replay.

Same controls as arrow_key_grasp.py, plus keyframe capture:

Movement:
    Arrow Up/Down     move hand forward / back
    Arrow Left/Right  move hand left / right
    PgUp / PgDn       move hand up / down
    , / .             roll hand left / right

Grasp:
    G                 close grasp
    R                 open grasp

Keyframes:
    S                 save current state as a keyframe
    D                 delete last keyframe
    Q                 quit and save all keyframes to file

Keyframes are saved as a JSON file containing a list of snapshots.
Each snapshot stores the full qpos, ctrl, grasp_t, and EE/object poses.
The env can replay these in reset() by interpolating between keyframes.

Usage:
    python scripts/capture_keyframes.py
    python scripts/capture_keyframes.py --object 006_mustard_bottle
    python scripts/capture_keyframes.py --output grasp_keyframes.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
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

GRASP_SPEED = 1.5
MOVE_SPEED = 0.3
ROLL_SPEED = 1.0

OBJECT_SPAWN_Z = 0.04

# Windows virtual-key codes
_user32 = ctypes.windll.user32
MOVE_VK: dict[int, np.ndarray] = {
    0x26: np.array([1.0, 0.0, 0.0]),   # up arrow    — forward
    0x28: np.array([-1.0, 0.0, 0.0]),  # down arrow  — back
    0x25: np.array([0.0, 1.0, 0.0]),   # left arrow  — left
    0x27: np.array([0.0, -1.0, 0.0]),  # right arrow — right
    0x21: np.array([0.0, 0.0, 1.0]),   # PgUp        — up
    0x22: np.array([0.0, 0.0, -1.0]),  # PgDn        — down
}
ROLL_VK: dict[int, float] = {
    0xBC: -1.0,  # , — roll left
    0xBE: 1.0,   # . — roll right
}
GRASP_VK = {
    0x47: 1.0,  # G — close
    0x52: 0.0,  # R — open
}
# Keyframe capture keys
VK_S = 0x53  # S — save keyframe
VK_D = 0x44  # D — delete last keyframe
VK_Q = 0x51  # Q — quit and save


def _is_held(vk: int) -> bool:
    return bool(_user32.GetAsyncKeyState(vk) & 0x8000)


def _is_pressed(vk: int) -> bool:
    """Detect a fresh key press (transition from up to down)."""
    return bool(_user32.GetAsyncKeyState(vk) & 0x0001)


def _rotate_quat(quat_wxyz: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    half = angle / 2.0
    dq = np.array([np.cos(half), *(np.sin(half) * axis)])
    w1, x1, y1, z1 = dq
    w2, x2, y2, z2 = quat_wxyz
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _snapshot(
    data: mujoco.MjData,
    model: mujoco.MjModel,
    grasp_t: float,
    object_id: str,
) -> dict:
    """Capture the current state as a serializable dict."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_id)

    return {
        "qpos": data.qpos.copy().tolist(),
        "ctrl": data.ctrl.copy().tolist(),
        "grasp_t": float(grasp_t),
        "ee_pos": data.site_xpos[site_id].copy().tolist(),
        "obj_pos": data.xpos[obj_body_id].copy().tolist(),
        "time": float(data.time),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture grasp keyframes interactively")
    parser.add_argument("--object", default="005_tomato_soup_can",
                        help="YCB object ID (default: 005_tomato_soup_can)")
    parser.add_argument("--output", default="config/grasp_keyframes.json",
                        help="Output file for keyframes (default: config/grasp_keyframes.json)")
    args = parser.parse_args()
    OBJECT_ID = args.object
    output_path = PROJECT_ROOT / args.output

    print(f"Building scene with {OBJECT_ID}...")
    model, data, temp_dir = build_scene(OBJECT_ID)
    print(f"Scene loaded: {model.nbody} bodies, {model.ngeom} geoms")

    # Set arm to home and thumb to pre-opposed position
    data.qpos[:ARM_NU] = HOME_ARM_QPOS
    data.qpos[ARM_NU:ARM_NU + HAND_NU] = HAND_OPEN_PRE
    data.ctrl[:ARM_NU] = HOME_ARM_QPOS
    data.ctrl[ARM_NU:ARM_NU + HAND_NU] = HAND_OPEN_PRE
    mujoco.mj_forward(model, data)

    # Place object below the hand
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    hand_xy = data.site_xpos[site_id][:2].copy()
    obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_ID)
    if obj_body_id >= 0:
        joint_id = int(model.body_jntadr[obj_body_id])
        qpos_adr = int(model.jnt_qposadr[joint_id])
        data.qpos[qpos_adr:qpos_adr + 3] = [hand_xy[0], hand_xy[1], OBJECT_SPAWN_Z]
        data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    # IK setup (arm only)
    configuration = mink.Configuration(model)
    ee_task = mink.FrameTask(
        frame_name="attachment_site", frame_type="site",
        position_cost=1.0, orientation_cost=1.0, lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=5e-2)
    limits = [mink.ConfigurationLimit(model=model)]

    grasp_t = 0.0
    grasp_target = 0.0
    keyframes: list[dict] = []

    print()
    print("=== KEYFRAME CAPTURE MODE ===")
    print("Controls: arrows=move | PgUp/PgDn=vertical | ,/.=roll | G/R=close/open")
    print("S = save keyframe | D = delete last | Q = quit & save")
    print(f"Output: {output_path}")
    print()

    try:
        with mujoco.viewer.launch_passive(
            model=model, data=data,
            show_left_ui=False, show_right_ui=False,
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)

            configuration.update(data.qpos)
            posture_task.set_target_from_configuration(configuration)
            mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")
            mocap_id = model.body("target").mocapid[0]

            rate = RateLimiter(frequency=200.0, warn=False)
            n_substeps = max(1, round(1.0 / (200.0 * model.opt.timestep)))

            while viewer.is_running():
                # --- Grasp control ---
                for vk, target in GRASP_VK.items():
                    if _is_held(vk):
                        grasp_target = target

                step = GRASP_SPEED * rate.dt
                diff = grasp_target - grasp_t
                grasp_t = float(np.clip(
                    grasp_t + np.sign(diff) * min(step, abs(diff)), 0.0, 1.0
                ))

                # --- Movement ---
                move_dir = sum(
                    (v for k, v in MOVE_VK.items() if _is_held(k)), np.zeros(3)
                )
                if np.any(move_dir):
                    data.mocap_pos[mocap_id] += move_dir * MOVE_SPEED * rate.dt

                roll = sum(v for k, v in ROLL_VK.items() if _is_held(k))
                if roll:
                    data.mocap_quat[mocap_id] = _rotate_quat(
                        data.mocap_quat[mocap_id],
                        np.array([1.0, 0.0, 0.0]),
                        roll * ROLL_SPEED * rate.dt,
                    )

                # --- IK solve ---
                configuration.update(data.qpos)
                T_wt = mink.SE3.from_mocap_name(model, data, "target")
                ee_task.set_target(T_wt)
                vel = mink.solve_ik(
                    configuration, [ee_task, posture_task], rate.dt,
                    "daqp", damping=1e-3, limits=limits,
                )
                configuration.integrate_inplace(vel, rate.dt)

                # --- Write ctrl ---
                data.ctrl[:ARM_NU] = configuration.q[:ARM_NU]
                data.ctrl[ARM_NU:ARM_NU + HAND_NU] = (
                    (1.0 - grasp_t) * HAND_OPEN_PRE + grasp_t * FINGER_CLOSED
                )

                # --- Keyframe capture ---
                if _is_pressed(VK_S):
                    snap = _snapshot(data, model, grasp_t, OBJECT_ID)
                    keyframes.append(snap)
                    ee = snap["ee_pos"]
                    obj = snap["obj_pos"]
                    print(
                        f"  [KF {len(keyframes)}] grasp={grasp_t:.2f} "
                        f"EE=[{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}] "
                        f"obj=[{obj[0]:.3f}, {obj[1]:.3f}, {obj[2]:.3f}]"
                    )

                if _is_pressed(VK_D) and keyframes:
                    removed = keyframes.pop()
                    print(f"  [DEL] Removed keyframe {len(keyframes) + 1}")

                if _is_pressed(VK_Q):
                    print("\nQuitting...")
                    break

                # --- Physics ---
                for _ in range(n_substeps):
                    mujoco.mj_step(model, data)
                viewer.sync()
                rate.sleep()

    finally:
        # Save keyframes
        if keyframes:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output = {
                "object_id": OBJECT_ID,
                "n_keyframes": len(keyframes),
                "keyframes": keyframes,
            }
            output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
            print(f"\nSaved {len(keyframes)} keyframes to {output_path}")
        else:
            print("\nNo keyframes captured.")

        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
