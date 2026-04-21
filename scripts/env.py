#!/usr/bin/env python3
"""Barebones Gymnasium Env wrapping build_scene for Phase 4 RL work.

Minimal implementation of item 3 from Uksang's Phase 4 TODO (2026-04-16):
- reset() replays captured keyframes to achieve a grasp, then returns
  the first observation.
- step() runs n_substeps of mj_step, computes reward, checks termination.
- Episode terminates if the object drops below drop_z or step_count hits
  timeout_steps.

Action space (22D per PDF item 6):
  [ee_pos_delta(3), ee_rot_delta(3), hand_ctrl(16)]
  EE deltas are resolved via mink differential IK each step.

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
import mink
import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.test_scene import build_scene  # noqa: E402

DEFAULT_KEYFRAMES = PROJECT_ROOT / "config" / "grasp_keyframes.json"

# Steps to interpolate between each pair of keyframes during replay.
INTERP_STEPS_PER_KF = 80
REPLAY_SUBSTEPS = 5

# xArm7 has 7 arm actuators (joint1–joint7), LEAP hand has 16.
ARM_NU = 7
HAND_NU = 16

# Reward coefficients (PDF item 7)
REWARD_RETENTION_SCALE = 10.0   # weight on palm-relative displacement
REWARD_DROP_PENALTY = -10.0     # terminal reward on drop
REWARD_SMOOTH_ALPHA = 0.001     # weight on action delta penalty
REWARD_ALIVE_BONUS = 0.0        # per-step survival bonus (0=disabled)

# Disturbance force (PDF item 8)
# Sustained-pull model: resample direction + magnitude at a random interval,
# hold between resamples. More realistic than per-step white noise and matches
# the "yank" semantics in the spec. Intervals in env steps (each = n_substeps
# * 5ms = 0.025s of sim time by default).
DEFAULT_FORCE_MAG = 5.0
DEFAULT_FORCE_PERIOD_MIN = 20   # 0.5 s at 40 Hz env rate
DEFAULT_FORCE_PERIOD_MAX = 80   # 2.0 s at 40 Hz env rate

# EE-delta action bounds (PDF item 6)
EE_POS_DELTA_BOUND = 0.01   # ±0.01 m per step
EE_ROT_DELTA_BOUND = 0.05   # ±0.05 rad per step


class GraspEnv(gym.Env):
    """Gymnasium env: LEAP hand grasps an object and must maintain contact.

    Action space: 22D = [ee_pos_delta(3), ee_rot_delta(3), hand_ctrl(16)]
    Observation space: 45D = [slip(3), ee_pos(3), arm_qpos(7), hand_qpos(16), hand_qvel(16)]
    """

    def __init__(
        self,
        object_id: str = "005_tomato_soup_can",
        n_substeps: int = 5,
        timeout_steps: int = 500,
        drop_threshold: float = 0.05,
        force_mag: float = DEFAULT_FORCE_MAG,
        force_period_min: int = DEFAULT_FORCE_PERIOD_MIN,
        force_period_max: int = DEFAULT_FORCE_PERIOD_MAX,
        survival_bonus: float = REWARD_ALIVE_BONUS,
        keyframes_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self._object_id = object_id
        self.n_substeps = n_substeps
        self.timeout_steps = timeout_steps
        self.drop_threshold = drop_threshold
        self.force_mag = force_mag
        self.force_period_min = max(1, int(force_period_min))
        self.force_period_max = max(self.force_period_min, int(force_period_max))
        self.survival_bonus = survival_bonus

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

        # Cache joint index arrays for robot state obs (PDF item 6)
        _arm_joints = [f"joint{i}" for i in range(1, 8)]
        _hand_joints = [f"leap_right/{i}" for i in range(16)]
        self._arm_qpos_idx = np.array([
            self.model.jnt_qposadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in _arm_joints
        ])
        self._hand_qpos_idx = np.array([
            self.model.jnt_qposadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in _hand_joints
        ])
        self._hand_qvel_idx = np.array([
            self.model.jnt_dofadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in _hand_joints
        ])

        # Mink IK setup for EE-delta action space (PDF item 6)
        self._mink_config = mink.Configuration(self.model)
        self._ee_task = mink.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        self._posture_task = mink.PostureTask(model=self.model, cost=5e-2)
        self._mink_limits = [mink.ConfigurationLimit(model=self.model)]
        self._ik_dt = float(self.n_substeps * self.model.opt.timestep)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(45,), dtype=np.float32
        )

        # 22D action space: [ee_pos_delta(3), ee_rot_delta(3), hand_ctrl(16)]
        hand_low = self.model.actuator_ctrlrange[ARM_NU:ARM_NU + HAND_NU, 0].astype(np.float32)
        hand_high = self.model.actuator_ctrlrange[ARM_NU:ARM_NU + HAND_NU, 1].astype(np.float32)
        self.action_space = gym.spaces.Box(
            low=np.concatenate([
                np.full(3, -EE_POS_DELTA_BOUND, dtype=np.float32),
                np.full(3, -EE_ROT_DELTA_BOUND, dtype=np.float32),
                hand_low,
            ]),
            high=np.concatenate([
                np.full(3, EE_POS_DELTA_BOUND, dtype=np.float32),
                np.full(3, EE_ROT_DELTA_BOUND, dtype=np.float32),
                hand_high,
            ]),
            dtype=np.float32,
        )

        self._step_count = 0
        self._prev_action = np.zeros(22, dtype=np.float64)
        self._initial_palm_rel: np.ndarray | None = None

        # Sustained-pull force state (resampled periodically in step)
        self._current_force = np.zeros(3, dtype=np.float64)
        self._force_resample_at = 0

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
        arm_qpos = self.data.qpos[self._arm_qpos_idx]
        hand_qpos = self.data.qpos[self._hand_qpos_idx]
        hand_qvel = self.data.qvel[self._hand_qvel_idx]
        return np.concatenate([slip, ee_pos, arm_qpos, hand_qpos, hand_qvel]).astype(np.float32)

    def _replay_keyframes(self) -> None:
        """Interpolate through captured keyframes to achieve the grasp."""
        if not self._keyframe_ctrls:
            return

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
        for _ in range(20):
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

        # Sync mink configuration to post-grasp state; lock posture for regularization
        self._mink_config.update(self.data.qpos)
        self._posture_task.set_target_from_configuration(self._mink_config)

        # Record initial palm-relative object pose for retention reward
        self._initial_palm_rel = self._palm_relative_obj_pos()

        # Baseline: EE delta=0 (hold position), hand=current grasp ctrl
        self._prev_action = np.concatenate([
            np.zeros(6, dtype=np.float64),
            self.data.ctrl[ARM_NU:ARM_NU + HAND_NU].copy(),
        ])
        self._step_count = 0
        # Force will resample immediately on first step
        self._current_force = np.zeros(3, dtype=np.float64)
        self._force_resample_at = 0
        return self._obs(), {"slip_mag": 0.0, "retention": 0.0}

    def step(self, action):
        # Decode 22D action: [ee_pos_delta(3), ee_rot_delta(3), hand_ctrl(16)]
        ee_pos_delta = np.asarray(action[:3], dtype=np.float64)
        ee_rot_delta = np.asarray(action[3:6], dtype=np.float64)
        hand_ctrl = np.asarray(action[6:22], dtype=np.float64)

        # Read current EE pose from MuJoCo site
        ee_pos = self.data.site_xpos[self._ee_site].copy()
        ee_xmat = self.data.site_xmat[self._ee_site].reshape(3, 3).copy()

        # Integrate target EE pose: position delta in world frame, rotation via axis-angle
        target_pos = ee_pos + ee_pos_delta
        target_so3 = mink.SO3.from_matrix(ee_xmat) @ mink.SO3.exp(ee_rot_delta)
        target_se3 = mink.SE3.from_rotation_and_translation(target_so3, target_pos)

        # Solve differential IK for arm joints
        self._mink_config.update(self.data.qpos)
        self._ee_task.set_target(target_se3)
        vel = mink.solve_ik(
            self._mink_config,
            [self._ee_task, self._posture_task],
            self._ik_dt,
            "daqp",
            damping=1e-3,
            limits=self._mink_limits,
        )
        self._mink_config.integrate_inplace(vel, self._ik_dt)

        # Write arm ctrl (from IK) and hand ctrl (direct absolute targets)
        self.data.ctrl[:ARM_NU] = self._mink_config.q[:ARM_NU]
        self.data.ctrl[ARM_NU:ARM_NU + HAND_NU] = hand_ctrl

        # Sustained-pull disturbance force (PDF item 8). Resample direction +
        # magnitude at a random interval (force_period_min..max env steps),
        # hold constant between resamples. More realistic than per-step white
        # noise and matches the "yank" semantics of the spec.
        if self.force_mag > 0 and self._step_count >= self._force_resample_at:
            direction = self.np_random.standard_normal(3)
            direction /= max(np.linalg.norm(direction), 1e-8)
            magnitude = self.np_random.uniform(0, self.force_mag)
            self._current_force = direction * magnitude
            period = int(self.np_random.integers(
                self.force_period_min, self.force_period_max + 1
            ))
            self._force_resample_at = self._step_count + period
        elif self.force_mag <= 0:
            self._current_force = np.zeros(3, dtype=np.float64)

        self.last_applied_force = self._current_force.copy()
        self.data.xfrc_applied[self._obj_body][:3] = self._current_force

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        # Clear xfrc after stepping — next step re-writes from _current_force.
        # (Clearing here also ensures force_mag=0 paths don't leak stale force.)
        self.data.xfrc_applied[self._obj_body][:] = 0

        self._step_count += 1

        # Drop detection: palm-relative displacement exceeds threshold
        displacement = float(np.linalg.norm(
            self._palm_relative_obj_pos() - self._initial_palm_rel
        ))
        dropped = displacement > self.drop_threshold
        r_retention = -REWARD_RETENTION_SCALE * displacement
        r_drop = REWARD_DROP_PENALTY if dropped else 0.0
        r_smooth = -REWARD_SMOOTH_ALPHA * float(
            np.linalg.norm(np.asarray(action, dtype=np.float64) - self._prev_action)
        )
        r_alive = self.survival_bonus if not dropped else 0.0
        reward = r_retention + r_drop + r_smooth + r_alive

        self._prev_action = np.asarray(action, dtype=np.float64)
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
