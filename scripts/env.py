#!/usr/bin/env python3
"""Barebones Gymnasium Env wrapping build_scene for Phase 4 RL work.

Minimal implementation of item 3 from Uksang's Phase 4 TODO (2026-04-16):
- reset() spawns the object, runs an 800-step settle, executes a hardcoded
  close grasp, returns the first observation.
- step() runs n_substeps of mj_step, computes reward, checks termination.
- Episode terminates if the object drops below drop_z or step_count hits
  timeout_steps.

Usage:
    from scripts.env import GraspEnv
    env = GraspEnv()
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

from __future__ import annotations

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
from scripts.hardcoded_grasp import FINGER_CLOSED  # noqa: E402

ARM_NU = 7
HAND_NU = 16
SETTLE_STEPS = 800
CLOSE_STEPS = 500
OBJECT_SPAWN_QPOS = np.array([0.4, 0.0, 0.12, 1.0, 0.0, 0.0, 0.0])


class GraspEnv(gym.Env):
    """Gymnasium env: LEAP hand grasps an object and must maintain contact."""

    def __init__(
        self,
        object_id: str = "005_tomato_soup_can",
        n_substeps: int = 5,
        timeout_steps: int = 500,
        drop_z: float = 0.05,
    ) -> None:
        super().__init__()
        self.n_substeps = n_substeps
        self.timeout_steps = timeout_steps
        self.drop_z = drop_z

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

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=self.model.actuator_ctrlrange[:, 0].astype(np.float32),
            high=self.model.actuator_ctrlrange[:, 1].astype(np.float32),
            dtype=np.float32,
        )
        self._step_count = 0

    def _obs(self) -> np.ndarray:
        return self.data.site_xpos[self._ee_site].astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key)
        self.data.qpos[self._obj_qadr:self._obj_qadr + 7] = OBJECT_SPAWN_QPOS
        mujoco.mj_forward(self.model, self.data)

        # Settle: let physics come to rest with ctrl at home
        for _ in range(SETTLE_STEPS):
            mujoco.mj_step(self.model, self.data)

        # Hardcoded close grasp: set finger targets and run physics
        self.data.ctrl[ARM_NU:ARM_NU + HAND_NU] = FINGER_CLOSED
        for _ in range(CLOSE_STEPS):
            mujoco.mj_step(self.model, self.data)

        self._step_count = 0
        return self._obs(), {}

    def step(self, action):
        self.data.ctrl[:] = action
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        dropped = bool(self.data.xpos[self._obj_body, 2] < self.drop_z)
        reward = -10.0 if dropped else 0.0
        truncated = self._step_count >= self.timeout_steps
        return self._obs(), reward, dropped, truncated, {}

    def close(self):
        if self._temp_dir and Path(self._temp_dir).exists():
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None
