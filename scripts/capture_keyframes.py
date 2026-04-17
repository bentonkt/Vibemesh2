#!/usr/bin/env python3
"""Capture grasp keyframes interactively for headless replay.

Cross-platform (Mac/Windows/Linux) using MuJoCo's key_callback.

Movement (each press = one step):
    Arrow Up/Down     move hand forward / back
    Arrow Left/Right  move hand left / right
    PgUp / PgDn       move hand up / down
    , / .             roll hand left / right

Grasp:
    G                 start closing grasp (ramps smoothly)
    R                 start opening grasp (ramps smoothly)

Keyframes:
    S                 save current state as a keyframe
    D                 delete last keyframe
    Q                 quit and save all keyframes to file

Usage:
    python scripts/capture_keyframes.py
    python scripts/capture_keyframes.py --object 006_mustard_bottle
    python scripts/capture_keyframes.py --output config/my_keyframes.json
    python scripts/capture_keyframes.py --step-size 0.02
"""

from __future__ import annotations

import argparse
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
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    1.9, 0.0, 0.0, 0.0,
])

FINGER_CLOSED = np.array([
    1.2, 0.0, 1.0, 1.0,
    1.2, 0.0, 1.0, 1.0,
    1.2, 0.0, 1.0, 1.0,
    1.9, 1.2, 0.5, 0.8,
])
# fmt: on

GRASP_SPEED = 1.5
OBJECT_SPAWN_Z = 0.04

# GLFW key codes (cross-platform)
KEY_UP = 265
KEY_DOWN = 264
KEY_LEFT = 263
KEY_RIGHT = 262
KEY_PAGE_UP = 266
KEY_PAGE_DOWN = 267
KEY_G = 71
KEY_R = 82
KEY_S = 83
KEY_D = 68
KEY_Q = 81
KEY_COMMA = 44
KEY_PERIOD = 46


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
    parser.add_argument("--object", default="005_tomato_soup_can")
    parser.add_argument("--output", default="config/grasp_keyframes.json")
    parser.add_argument("--step-size", type=float, default=0.01,
                        help="Movement per key press in meters (default: 0.01)")
    parser.add_argument("--roll-step", type=float, default=0.05,
                        help="Roll per key press in radians (default: 0.05)")
    args = parser.parse_args()
    OBJECT_ID = args.object
    STEP_SIZE = args.step_size
    ROLL_STEP = args.roll_step
    output_path = PROJECT_ROOT / args.output

    print(f"Building scene with {OBJECT_ID}...")
    model, data, temp_dir = build_scene(OBJECT_ID)
    print(f"Scene loaded: {model.nbody} bodies, {model.ngeom} geoms")

    # Set arm to home + thumb pre-opposed
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

    # IK setup
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
    quit_flag = False

    # Key callback — fires on each key press
    configuration.update(data.qpos)
    posture_task.set_target_from_configuration(configuration)
    mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")
    mocap_id = model.body("target").mocapid[0]

    # Movement/action vectors keyed by GLFW key code
    MOVE_KEYS: dict[int, np.ndarray] = {
        KEY_UP:        np.array([1.0, 0.0, 0.0]) * STEP_SIZE,
        KEY_DOWN:      np.array([-1.0, 0.0, 0.0]) * STEP_SIZE,
        KEY_LEFT:      np.array([0.0, 1.0, 0.0]) * STEP_SIZE,
        KEY_RIGHT:     np.array([0.0, -1.0, 0.0]) * STEP_SIZE,
        KEY_PAGE_UP:   np.array([0.0, 0.0, 1.0]) * STEP_SIZE,
        KEY_PAGE_DOWN: np.array([0.0, 0.0, -1.0]) * STEP_SIZE,
    }

    def on_key(key: int) -> None:
        nonlocal grasp_target, quit_flag

        # Movement
        if key in MOVE_KEYS:
            data.mocap_pos[mocap_id] += MOVE_KEYS[key]

        # Roll
        elif key == KEY_COMMA:
            data.mocap_quat[mocap_id] = _rotate_quat(
                data.mocap_quat[mocap_id], np.array([1.0, 0.0, 0.0]), -ROLL_STEP
            )
        elif key == KEY_PERIOD:
            data.mocap_quat[mocap_id] = _rotate_quat(
                data.mocap_quat[mocap_id], np.array([1.0, 0.0, 0.0]), ROLL_STEP
            )

        # Grasp
        elif key == KEY_G:
            grasp_target = 1.0
        elif key == KEY_R:
            grasp_target = 0.0

        # Keyframe capture
        elif key == KEY_S:
            snap = _snapshot(data, model, grasp_t, OBJECT_ID)
            keyframes.append(snap)
            ee = snap["ee_pos"]
            obj = snap["obj_pos"]
            print(
                f"  [KF {len(keyframes)}] grasp={grasp_t:.2f} "
                f"EE=[{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}] "
                f"obj=[{obj[0]:.3f}, {obj[1]:.3f}, {obj[2]:.3f}]"
            )
        elif key == KEY_D:
            if keyframes:
                keyframes.pop()
                print(f"  [DEL] Removed keyframe, {len(keyframes)} remaining")
            else:
                print("  [DEL] No keyframes to delete")
        elif key == KEY_Q:
            quit_flag = True

    print()
    print("=== KEYFRAME CAPTURE MODE ===")
    print(f"Movement: arrows (step={STEP_SIZE}m) | PgUp/PgDn | ,/. (roll={ROLL_STEP}rad)")
    print("Grasp: G=close | R=open")
    print("Capture: S=save keyframe | D=delete last | Q=quit & save")
    print(f"Output: {output_path}")
    print()

    try:
        with mujoco.viewer.launch_passive(
            model=model, data=data,
            show_left_ui=False, show_right_ui=False,
            key_callback=on_key,
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)

            rate = RateLimiter(frequency=200.0, warn=False)
            n_substeps = max(1, round(1.0 / (200.0 * model.opt.timestep)))

            while viewer.is_running() and not quit_flag:
                # Smooth grasp ramp
                step = GRASP_SPEED * rate.dt
                diff = grasp_target - grasp_t
                grasp_t = float(np.clip(
                    grasp_t + np.sign(diff) * min(step, abs(diff)), 0.0, 1.0
                ))

                # IK solve
                configuration.update(data.qpos)
                T_wt = mink.SE3.from_mocap_name(model, data, "target")
                ee_task.set_target(T_wt)
                vel = mink.solve_ik(
                    configuration, [ee_task, posture_task], rate.dt,
                    "daqp", damping=1e-3, limits=limits,
                )
                configuration.integrate_inplace(vel, rate.dt)

                # Write ctrl
                data.ctrl[:ARM_NU] = configuration.q[:ARM_NU]
                data.ctrl[ARM_NU:ARM_NU + HAND_NU] = (
                    (1.0 - grasp_t) * HAND_OPEN_PRE + grasp_t * FINGER_CLOSED
                )

                # Physics
                for _ in range(n_substeps):
                    mujoco.mj_step(model, data)
                viewer.sync()
                rate.sleep()

    finally:
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
