#!/usr/bin/env python3
"""MuJoCo MJX (JAX-backend) port of GraspEnv.

Key differences from scripts/env.py:
- Action space is 23D direct joint ctrl (7 arm + 16 hand), NOT EE-delta.
  Mink IK is incompatible with MJX's JAX data structures; JAX-native
  differential IK is left as a follow-up (see feasibility study).
- Reset uses a pre-computed post-grasp snapshot (qpos/ctrl arrays captured
  offline via build_init_snapshot()), not the 2000-step keyframe replay.
- step() and reset() are pure JAX functions suitable for jax.jit + jax.vmap.
- Observation and reward logic mirrors env.py exactly.

Usage:
    from scripts.env_jax import GraspEnvMJX, build_init_snapshot
    snapshot = build_init_snapshot()           # run once, CPU
    env = GraspEnvMJX(snapshot)
    # Vectorised smoke test (n_envs parallel):
    from scripts.smoke_jax import run_smoke_test
    run_smoke_test(env, n_envs=64, n_steps=100)

Action space: Box(-inf, inf, (23,)) mapped via actuator_ctrlrange clipping.
Observation space: (45,) float32 — identical to env.py.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Deferred JAX imports so the file can be imported on machines without JAX
# (e.g., import-time checks). Call require_jax() before any JAX ops.
_jax = None
_jnp = None
_mjx = None


def _require_jax():
    global _jax, _jnp, _mjx
    if _jax is None:
        import jax
        import jax.numpy as jnp
        from mujoco import mjx
        _jax = jax
        _jnp = jnp
        _mjx = mjx


# ─── Scene constants (mirror env.py) ──────────────────────────────────────────

ARM_NU = 7
HAND_NU = 16
TOTAL_NU = ARM_NU + HAND_NU  # 23

REWARD_RETENTION_SCALE = 10.0
REWARD_DROP_PENALTY = -10.0
REWARD_SMOOTH_ALPHA = 0.001
DEFAULT_FORCE_MAG = 5.0

# Keyframe replay params — used during offline snapshot generation
INTERP_STEPS_PER_KF = 80
REPLAY_SUBSTEPS = 5
SETTLE_STEPS = 20

DEFAULT_KEYFRAMES = PROJECT_ROOT / "config" / "grasp_keyframes.json"
DEFAULT_OBJECT = "005_tomato_soup_can"


# ─── Snapshot type ────────────────────────────────────────────────────────────

class GraspSnapshot(NamedTuple):
    """Pre-computed post-grasp state for fast JIT-compatible resets."""
    qpos: np.ndarray          # shape (nq,) — full joint positions at grasp
    ctrl: np.ndarray          # shape (nu,) — actuator ctrl at grasp
    obj_qadr: int             # index into qpos for object freejoint
    arm_qpos_idx: np.ndarray  # shape (7,) — arm joint qpos indices
    hand_qpos_idx: np.ndarray # shape (16,) — hand joint qpos indices
    hand_qvel_idx: np.ndarray # shape (16,) — hand joint vel indices
    palm_body_id: int
    ee_site_id: int
    obj_body_id: int
    ctrl_range: np.ndarray    # shape (23, 2) — actuator ctrl ranges
    initial_palm_rel: np.ndarray  # shape (3,) — initial palm-relative obj pos


# ─── Offline snapshot builder (CPU) ───────────────────────────────────────────

def build_init_snapshot(
    object_id: str = DEFAULT_OBJECT,
    keyframes_path: str | Path | None = None,
) -> GraspSnapshot:
    """Run keyframe replay on CPU MuJoCo, return a post-grasp snapshot.

    This is called once at startup. The result is cheap to store and can be
    broadcast across many MJX envs as the initial state.
    """
    # Import here to avoid circular dep
    from scripts.test_scene import build_scene  # noqa: PLC0415

    model, data, temp_dir = build_scene(object_id)
    try:
        _run_keyframe_replay(model, data, keyframes_path, object_id=object_id)
        return _extract_snapshot(model, data, object_id)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _run_keyframe_replay(model, data, keyframes_path, object_id: str = DEFAULT_OBJECT):
    kf_path = Path(keyframes_path) if keyframes_path else DEFAULT_KEYFRAMES
    if not kf_path.exists():
        return

    raw = json.loads(kf_path.read_text(encoding="utf-8"))
    keyframe_ctrls = [np.array(kf["ctrl"], dtype=np.float64) for kf in raw["keyframes"]]

    home_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home_key)

    ee_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    mujoco.mj_forward(model, data)
    hand_xy = data.site_xpos[ee_site][:2].copy()

    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_id)
    if obj_body < 0:
        raise ValueError(f"Object body '{object_id}' not found in model")

    joint = int(model.body_jntadr[obj_body])
    obj_qadr = int(model.jnt_qposadr[joint])
    data.qpos[obj_qadr:obj_qadr + 3] = [hand_xy[0], hand_xy[1], 0.04]
    data.qpos[obj_qadr + 3:obj_qadr + 7] = [1, 0, 0, 0]
    data.ctrl[:] = data.qpos[:model.nu]
    mujoco.mj_forward(model, data)

    current_ctrl = data.ctrl.copy()
    for target_ctrl in keyframe_ctrls:
        start_ctrl = current_ctrl.copy()
        for step in range(INTERP_STEPS_PER_KF):
            t = (step + 1) / INTERP_STEPS_PER_KF
            data.ctrl[:] = (1.0 - t) * start_ctrl + t * target_ctrl
            for _ in range(REPLAY_SUBSTEPS):
                mujoco.mj_step(model, data)
        current_ctrl = target_ctrl.copy()

    for _ in range(SETTLE_STEPS):
        for _ in range(REPLAY_SUBSTEPS):
            mujoco.mj_step(model, data)


def _extract_snapshot(model, data, object_id: str) -> GraspSnapshot:
    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_id)
    joint = int(model.body_jntadr[obj_body])
    obj_qadr = int(model.jnt_qposadr[joint])
    palm_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "leap_right/palm_lower")
    ee_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

    arm_joints = [f"joint{i}" for i in range(1, 8)]
    hand_joints = [f"leap_right/{i}" for i in range(16)]
    arm_qpos_idx = np.array([
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in arm_joints
    ])
    hand_qpos_idx = np.array([
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in hand_joints
    ])
    hand_qvel_idx = np.array([
        model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in hand_joints
    ])

    palm_xmat = data.xmat[palm_body].reshape(3, 3)
    palm_xpos = data.xpos[palm_body]
    obj_xpos = data.xpos[obj_body]
    initial_palm_rel = palm_xmat.T @ (obj_xpos - palm_xpos)

    ctrl_range = model.actuator_ctrlrange[:TOTAL_NU].copy()  # (23, 2)

    return GraspSnapshot(
        qpos=data.qpos.copy(),
        ctrl=data.ctrl[:TOTAL_NU].copy(),
        obj_qadr=obj_qadr,
        arm_qpos_idx=arm_qpos_idx,
        hand_qpos_idx=hand_qpos_idx,
        hand_qvel_idx=hand_qvel_idx,
        palm_body_id=palm_body,
        ee_site_id=ee_site,
        obj_body_id=obj_body,
        ctrl_range=ctrl_range,
        initial_palm_rel=initial_palm_rel,
    )


# ─── MJX Scene Builder ────────────────────────────────────────────────────────

def build_mjx_model(
    object_id: str = DEFAULT_OBJECT,
) -> tuple:
    """Build the MuJoCo model and convert to MJX.

    Returns (mj_model, mx_model) where mx_model is on GPU.
    Must be called after _require_jax().
    """
    _require_jax()
    from scripts.test_scene import build_scene  # noqa: PLC0415

    model, data, temp_dir = build_scene(object_id)
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Drop multiccd — not supported in MJX (see feasibility study §1)
    model.opt.enableflags &= ~mujoco.mjtEnableBit.mjENBL_MULTICCD

    mx = _mjx.put_model(model)
    return model, mx


# ─── Pure JAX step and reset functions ────────────────────────────────────────

def make_step_fn(mx, snapshot: GraspSnapshot, n_substeps: int = 5,
                 force_mag: float = DEFAULT_FORCE_MAG,
                 drop_threshold: float = 0.05):
    """Return a JIT+vmap-able step function.

    step_fn(data, action, rng) -> (next_data, obs, reward, done, info_dict)

    action: (23,) float32 in actuator_ctrlrange units
    """
    _require_jax()
    jax, jnp, mjx = _jax, _jnp, _mjx

    ctrl_low = jnp.array(snapshot.ctrl_range[:, 0], dtype=jnp.float32)
    ctrl_high = jnp.array(snapshot.ctrl_range[:, 1], dtype=jnp.float32)
    init_palm_rel = jnp.array(snapshot.initial_palm_rel, dtype=jnp.float32)

    palm_id = snapshot.palm_body_id
    obj_id = snapshot.obj_body_id
    ee_id = snapshot.ee_site_id
    arm_idx = snapshot.arm_qpos_idx
    hand_idx = snapshot.hand_qpos_idx
    hand_vel_idx = snapshot.hand_qvel_idx
    obj_qadr = snapshot.obj_qadr

    def _obs(d):
        # slip: object COM linear velocity in palm-local frame
        palm_xmat = d.xmat[palm_id].reshape(3, 3)
        obj_cvel = d.cvel[obj_id][3:]  # linear part of spatial velocity
        slip = palm_xmat.T @ obj_cvel

        ee_pos = d.site_xpos[ee_id]
        arm_qpos = d.qpos[arm_idx]
        hand_qpos = d.qpos[hand_idx]
        hand_qvel = d.qvel[hand_vel_idx]
        return jnp.concatenate([slip, ee_pos, arm_qpos, hand_qpos, hand_qvel]).astype(jnp.float32)

    def _palm_rel(d):
        palm_xmat = d.xmat[palm_id].reshape(3, 3)
        palm_xpos = d.xpos[palm_id]
        obj_xpos = d.xpos[obj_id]
        return palm_xmat.T @ (obj_xpos - palm_xpos)

    def step_fn(d, action, rng):
        # Clip action to ctrl range
        ctrl = jnp.clip(action.astype(jnp.float32), ctrl_low, ctrl_high)

        # Apply random disturbance force to object
        rng, rng_force = jax.random.split(rng)
        if force_mag > 0:
            raw_dir = jax.random.normal(rng_force, shape=(3,))
            norm = jnp.linalg.norm(raw_dir) + 1e-8
            direction = raw_dir / norm
            mag = jax.random.uniform(rng_force, minval=0.0, maxval=force_mag)
            force = direction * mag
            xfrc = d.xfrc_applied.at[obj_id, :3].set(force.astype(d.xfrc_applied.dtype))
            d = d.replace(ctrl=ctrl, xfrc_applied=xfrc)
        else:
            d = d.replace(ctrl=ctrl)

        # Step physics n_substeps times
        def _substep(d, _):
            return mjx.step(mx, d), None

        d, _ = jax.lax.scan(_substep, d, None, length=n_substeps)

        # Clear disturbance force
        if force_mag > 0:
            d = d.replace(xfrc_applied=d.xfrc_applied.at[obj_id, :3].set(0.0))

        # Reward computation
        palm_rel = _palm_rel(d)
        displacement = jnp.linalg.norm(palm_rel - init_palm_rel)
        dropped = displacement > drop_threshold
        r_retention = -REWARD_RETENTION_SCALE * displacement
        r_drop = jnp.where(dropped, REWARD_DROP_PENALTY, 0.0)
        reward = r_retention + r_drop

        obs = _obs(d)
        return d, obs, reward, dropped, {"displacement": displacement, "slip_mag": jnp.linalg.norm(_obs(d)[:3])}

    return jax.jit(step_fn)


def make_reset_fn(mx, snapshot: GraspSnapshot):
    """Return a JIT+vmap-able reset function.

    reset_fn(rng) -> (data, obs)
    """
    _require_jax()
    jax, jnp, mjx = _jax, _jnp, _mjx

    # MJX stores arrays as float32 on GPU (JAX default)
    init_qpos = jnp.array(snapshot.qpos, dtype=jnp.float32)
    init_ctrl = jnp.array(snapshot.ctrl, dtype=jnp.float32)
    arm_idx = snapshot.arm_qpos_idx
    hand_idx = snapshot.hand_qpos_idx
    hand_vel_idx = snapshot.hand_qvel_idx
    palm_id = snapshot.palm_body_id
    ee_id = snapshot.ee_site_id
    obj_id = snapshot.obj_body_id

    def _obs(d):
        palm_xmat = d.xmat[palm_id].reshape(3, 3)
        obj_cvel = d.cvel[obj_id][3:]
        slip = palm_xmat.T @ obj_cvel
        ee_pos = d.site_xpos[ee_id]
        arm_qpos = d.qpos[arm_idx]
        hand_qpos = d.qpos[hand_idx]
        hand_qvel = d.qvel[hand_vel_idx]
        return jnp.concatenate([slip, ee_pos, arm_qpos, hand_qpos, hand_qvel]).astype(jnp.float32)

    def reset_fn(rng):
        # Initialise MJX data from the pre-grasp snapshot
        d = mjx.make_data(mx)
        d = d.replace(qpos=init_qpos, ctrl=init_ctrl)
        d = mjx.forward(mx, d)
        obs = _obs(d)
        return d, obs

    return jax.jit(reset_fn)


# ─── GraspEnvMJX wrapper ──────────────────────────────────────────────────────

class GraspEnvMJX:
    """Thin wrapper holding the MJX model + compiled step/reset functions.

    Not a Gymnasium env — designed for direct JAX vmap usage.
    For sbx/SB3 training, wrap via GraspEnvMJXGym (future work).
    """

    def __init__(
        self,
        snapshot: GraspSnapshot,
        object_id: str = DEFAULT_OBJECT,
        n_substeps: int = 5,
        force_mag: float = DEFAULT_FORCE_MAG,
        drop_threshold: float = 0.05,
    ):
        _require_jax()
        self.snapshot = snapshot
        self.object_id = object_id
        self.n_substeps = n_substeps
        self.force_mag = force_mag
        self.drop_threshold = drop_threshold

        self.mj_model, self.mx = build_mjx_model(object_id)

        self.reset_fn = make_reset_fn(self.mx, snapshot)
        self.step_fn = make_step_fn(
            self.mx, snapshot,
            n_substeps=n_substeps,
            force_mag=force_mag,
            drop_threshold=drop_threshold,
        )

        # Vmapped versions for parallel envs
        self.reset_batch = _jax.jit(_jax.vmap(self.reset_fn))
        self.step_batch = _jax.jit(_jax.vmap(self.step_fn))

    @property
    def obs_dim(self):
        return 45

    @property
    def act_dim(self):
        return TOTAL_NU  # 23

    def reset(self, n_envs: int, rng):
        """Reset n_envs in parallel. Returns (batch_data, batch_obs)."""
        rngs = _jax.random.split(rng, n_envs)
        return self.reset_batch(rngs)

    def step(self, batch_data, batch_action, rng):
        """Step n_envs in parallel. Returns (next_data, obs, reward, done, info)."""
        rngs = _jax.random.split(rng, batch_action.shape[0])
        return self.step_batch(batch_data, batch_action, rngs)
