#!/usr/bin/env python3
"""Barebones Gymnasium Env wrapping build_scene for Phase 4 RL work.

Minimal implementation of item 3 from Uksang's Phase 4 TODO (2026-04-16):
- reset() replays captured keyframes to achieve a grasp, then returns
  the first observation.
- step() runs n_substeps of mj_step, computes reward, checks termination.
- Episode terminates if the object drops below drop_z or step_count hits
  timeout_steps.

Keyframes are captured interactively via scripts/capture_keyframes.py
and saved to config/grasp_keyframes.json.

Usage:
    from scripts.env import GraspEnv
    env = GraspEnv()
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.test_scene import build_scene  # noqa: E402

DEFAULT_KEYFRAMES = PROJECT_ROOT / "config" / "grasp_keyframes.json"

# Steps to interpolate between each pair of keyframes during replay
INTERP_STEPS_PER_KF = 200  # 1 second at 200 Hz
# Physics substeps per interpolation step (matches arrow_key_grasp)
REPLAY_SUBSTEPS = 5

# Reward coefficients (PDF item 7)
REWARD_RETENTION_SCALE = 1.0    # weight on palm-relative displacement
REWARD_DROP_PENALTY = -10.0     # terminal reward on drop
REWARD_SMOOTH_ALPHA = 0.01      # weight on action delta penalty

# Disturbance force (PDF item 8)
DEFAULT_FORCE_MAG = 5.0         # max force magnitude in Newtons


class GraspEnv(gym.Env):
    """Gymnasium env: LEAP hand grasps an object and must maintain contact."""

    def __init__(
        self,
        object_id: str = "005_tomato_soup_can",
        n_substeps: int = 5,
        timeout_steps: int = 500,
        drop_threshold: float = 0.05,
        force_mag: float = DEFAULT_FORCE_MAG,
        keyframes_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self._object_id = object_id
        self.n_substeps = n_substeps
        self.timeout_steps = timeout_steps
        self.drop_threshold = drop_threshold
        self.force_mag = force_mag

        self.model, self.data, self._temp_dir = build_scene(object_id)
        self._home_key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self._ee_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )
        self._obj_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, object_id
        )
        joint = int(self.model.body_jntadr[self._obj_body])
        self._obj_qadr = int(self.model.jnt_qposadr[joint])
        self._palm_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "leap_right/palm_lower"
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=self.model.actuator_ctrlrange[:, 0].astype(np.float32),
            high=self.model.actuator_ctrlrange[:, 1].astype(np.float32),
            dtype=np.float32,
        )
        self._step_count = 0
        self._prev_action = np.zeros(self.model.nu, dtype=np.float64)
        self._initial_palm_rel: np.ndarray | None = None

        # Load keyframes
        kf_path = Path(keyframes_path) if keyframes_path else DEFAULT_KEYFRAMES
        if kf_path.exists():
            raw = json.loads(kf_path.read_text(encoding="utf-8"))
            self._keyframe_ctrls = [
                np.array(kf["ctrl"], dtype=np.float64) for kf in raw["keyframes"]
            ]
        else:
            self._keyframe_ctrls = []

    def _palm_relative_obj_pos(self) -> np.ndarray:
        """Object position in palm_lower's local frame (3D)."""
        palm_xmat = self.data.xmat[self._palm_body].reshape(3, 3)
        palm_xpos = self.data.xpos[self._palm_body]
        obj_xpos = self.data.xpos[self._obj_body]
        return palm_xmat.T @ (obj_xpos - palm_xpos)

    def _compute_slip(self) -> np.ndarray:
        """Object COM velocity in palm_lower's local frame (3D)."""
        palm_xmat = self.data.xmat[self._palm_body].reshape(3, 3)
        obj_vel_world = self.data.cvel[self._obj_body][3:]  # [angular, linear] → linear
        return palm_xmat.T @ obj_vel_world

    def _obs(self) -> np.ndarray:
        slip = self._compute_slip()
        ee_pos = self.data.site_xpos[self._ee_site]
        return np.concatenate([slip, ee_pos]).astype(np.float32)

    def _replay_keyframes(self) -> None:
        """Interpolate through captured keyframes to achieve the grasp."""
        if not self._keyframe_ctrls:
            return

        # Start from current ctrl
        current_ctrl = self.data.ctrl.copy()

        for target_ctrl in self._keyframe_ctrls:
            start_ctrl = current_ctrl.copy()
            for step in range(INTERP_STEPS_PER_KF):
                t = (step + 1) / INTERP_STEPS_PER_KF
                self.data.ctrl[:] = (1.0 - t) * start_ctrl + t * target_ctrl
                for _ in range(REPLAY_SUBSTEPS):
                    mujoco.mj_step(self.model, self.data)
            current_ctrl = target_ctrl.copy()

        # Hold final keyframe briefly to let contacts settle
        for _ in range(100):
            for _ in range(REPLAY_SUBSTEPS):
                mujoco.mj_step(self.model, self.data)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key)

        # Place object below the hand (same XY as EE, low Z)
        mujoco.mj_forward(self.model, self.data)
        hand_xy = self.data.site_xpos[self._ee_site][:2].copy()
        self.data.qpos[self._obj_qadr:self._obj_qadr + 3] = [
            hand_xy[0], hand_xy[1], 0.04,
        ]
        self.data.qpos[self._obj_qadr + 3:self._obj_qadr + 7] = [1, 0, 0, 0]

        # Set initial ctrl to home (arm + hand open pre-opposed)
        self.data.ctrl[:] = self.data.qpos[: self.model.nu]
        mujoco.mj_forward(self.model, self.data)

        # Replay captured keyframes to achieve grasp
        self._replay_keyframes()

        # Record initial palm-relative object pose for retention reward
        self._initial_palm_rel = self._palm_relative_obj_pos()
        self._prev_action = self.data.ctrl.copy()
        self._step_count = 0
        return self._obs(), {"slip_mag": 0.0, "retention": 0.0}

    def step(self, action):
        self.data.ctrl[:] = action

        # Random disturbance force on object (PDF item 8)
        if self.force_mag > 0:
            direction = self.np_random.standard_normal(3)
            direction /= max(np.linalg.norm(direction), 1e-8)
            magnitude = self.np_random.uniform(0, self.force_mag)
            self.data.xfrc_applied[self._obj_body][:3] = direction * magnitude

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        # Clear force after stepping (applied fresh each step)
        self.data.xfrc_applied[self._obj_body][:] = 0

        self._step_count += 1

        # Drop detection: palm-relative displacement exceeds threshold
        displacement = float(np.linalg.norm(
            self._palm_relative_obj_pos() - self._initial_palm_rel
        ))
        dropped = displacement > self.drop_threshold
        r_retention = -REWARD_RETENTION_SCALE * displacement
        r_drop = REWARD_DROP_PENALTY if dropped else 0.0
        r_smooth = -REWARD_SMOOTH_ALPHA * float(np.linalg.norm(action - self._prev_action))
        reward = r_retention + r_drop + r_smooth

        self._prev_action = np.array(action, dtype=np.float64)
        truncated = self._step_count >= self.timeout_steps
        slip = self._compute_slip()
        info = {
            "slip_mag": float(np.linalg.norm(slip)),
            "dropped": dropped,
            "retention": displacement,
            "r_retention": r_retention,
            "r_smooth": r_smooth,
        }
        return self._obs(), reward, dropped, truncated, info

    def close(self):
        if self._temp_dir and Path(self._temp_dir).exists():
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None
