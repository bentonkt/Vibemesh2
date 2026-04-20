# MJX Port Feasibility Study
**Date:** 2026-04-20  
**Branch:** `jax-port`  
**Author:** vibemesh-jax-port worker agent  
**Verdict: FEASIBLE** with one required adaptation (IK strategy)

---

## 1. Does MJX Support Our Scene?

### Robot model summary
| Component | DOF | Joint type | Actuator type | Tendons | Equality constraints |
|-----------|-----|------------|---------------|---------|----------------------|
| xArm7     | 7   | hinge      | general (pos/vel servo) | None | None |
| LEAP Hand | 16  | revolute   | position      | None | None |
| YCB object | 7 (free) | free | — | — | — |
| **Total** | **30 robot + 7 obj = 37 qpos** | | | | |

**Note on DOF count:** The environment tracks 45D observations but that includes joint velocities (16D hand qvel) stacked on 29D state (slip 3 + ee_pos 3 + arm_qpos 7 + hand_qpos 16). The physics system has 30 robot DOFs + 6 object DOFs = 36 physics DOFs total.

### Collision geometry check
Our scene uses:
- LEAP Hand: mesh geoms (`condim=6`, `group=3`)
- xArm: mesh geoms (links)
- YCB object: mesh geom (convex hull of collision.obj)
- Floor: plane

**MJX collision support matrix (from source):**
| Geom type | Supported in MJX? |
|-----------|-------------------|
| sphere    | ✅ |
| capsule   | ✅ |
| box       | ✅ |
| cylinder  | ✅ |
| ellipsoid | ✅ |
| plane     | ✅ |
| mesh (convex hull) | ✅ |
| heightfield | ✅ |
| plane–plane | ❌ (excluded pair) |

**Verdict:** All our geometry types are supported. MJX treats mesh geoms as convex hulls — same as our collision.obj meshes which are already convex hull approximations from `process_ycb.py`.

### condim=6 support
`geom_condim` is explicitly enumerated in MJX types as supporting values 1, 3, 4, or 6. Our `condim=6` (elliptic cone with tangential friction) is **supported** ✅.

### Known unsupported features we don't use
- Flex bodies: not used ✅
- Some actuator dyntype variants: our `general` and `position` types are standard ✅
- Fluid drag with implicitfast: not used ✅

### `<flag multiccd="enable"/>` — known gap
`multiccd` (multiple-pass convex collision detection) is **not present in MJX types**. The MJX step will silently use its own collision pipeline, which runs one convex-hull pass per geom pair. In practice, this means:
- Slightly degraded contact fidelity for mesh-mesh pairs at high penetration
- No functional blocker — the simulation will still run
- **Mitigation:** Drop `multiccd` from the MJX scene XML, or ignore and accept slightly softer contacts

### Tendon support
Neither xArm7 nor LEAP Hand uses tendons. MJX does implement spatial/fixed tendon passive forces and equality constraints for tendons, but we don't need them. ✅

**Overall scene compatibility: GREEN** — no hard blockers on contacts or mesh collisions.

---

## 2. Keyframe-Replay Reset in MJX

MJX resets are JIT-compiled pure functions. The reset pattern is:

```python
# Precompute a batched initial state from keyframes (CPU, once at startup)
init_qpos = jnp.array(keyframe_qpos)  # shape [n_kf, nq]
init_ctrl = jnp.array(keyframe_ctrl)  # shape [n_kf, nu]

@jax.jit
def reset_fn(rng):
    # Pick a random keyframe per env
    idx = jax.random.randint(rng, (), 0, n_kf)
    d = mjx.make_data(mx)
    d = d.replace(qpos=init_qpos[idx], ctrl=init_ctrl[idx])
    return mjx.forward(mx, d)  # forward kinematics to settle state
```

For batched reset via `vmap`:
```python
reset_batch = jax.vmap(reset_fn)
rngs = jax.random.split(key, n_envs)
batch_data = reset_batch(rngs)
```

**Key constraint:** The full keyframe replay loop (80 interp steps × 5 substeps × 5 keyframes = 2000 MuJoCo steps) **cannot run inside JIT as a Python loop**. Options:
- Option A (recommended): Run replay on CPU once, snapshot qpos/ctrl at the final grasp state, save these as the "keyframe init" JAX array. Reset just restores this pre-grasp snapshot.
- Option B: Use `jax.lax.fori_loop` to do the replay inside JIT (complex but possible).

**Recommendation:** Option A — precompute the post-grasp snapshot during model initialization, treat it as a constant JAX array. This is clean, fast, and fully JIT-compatible.

---

## 3. EE-Delta Action Space — IK Strategy

### The problem
`mink.solve_ik()` is a CPU-side QP solver that calls into the standard MuJoCo C API. It reads `mujoco.MjData` from RAM and calls DAQP or OSQP. This is **incompatible** with MJX's JAX data structures and cannot be used inside a `jax.vmap` or `jax.jit` context.

### Options evaluated

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| (a) Jacobian pseudoinverse in JAX | Compute `J = d(site_xpos)/d(qpos)` via MJX, solve `dq = J+ * dx` | Fully differentiable, GPU-parallel | Requires analytical Jacobian in JAX; ~200-400 lines; numerical stability needs tuning |
| (b) Direct 23D joint ctrl | Replace 22D EE-delta action with 23D direct joint ctrl (7 arm + 16 hand) | Zero new code, immediate | Changes action space semantics vs CPU policy; harder to transfer behavior |
| (c) CPU IK, MJX physics | Keep mink on CPU, only use MJX for the physics step | Preserves action space | CPU bottleneck defeats the point of MJX; breaks vmap |

**Decision: Use option (b) for the initial port.**

Rationale:
- This is a **scouting/smoke-test mission**, not a final training policy
- Direct joint control is simpler and validates the MJX physics pipeline
- The action space change is **documented as a known limitation** — can be replaced with JAX-IK (option a) in a follow-up PR after validating the physics
- The CPU policy trained on EE-delta action space is **unaffected** on `main`

**Action space for JAX port:** `Box(-1, 1, shape=(23,))` — 7 arm joints normalized + 16 hand joints normalized, mapped to `model.actuator_ctrlrange`.

---

## 4. RL Framework Choice

| Framework | API | JAX native? | MJX integration | Maturity |
|-----------|-----|-------------|-----------------|----------|
| SB3 (current) | High-level | No (PyTorch) | Not compatible with vmap | Mature |
| **sbx** | SB3-compatible | Yes (JAX) | Works with Gymnasium wrapper | Active |
| purejaxrl | Low-level | Yes | Expects Gymnax-style envs; needs adapter | Research |
| brax | Google ecosystem | Yes | Built-in, but requires Brax env format | Mature |

**Recommendation: `sbx` (Stable-Baselines3 JAX)**

Rationale:
- Drop-in API replacement for SB3 — same `PPO(policy, env).learn(n_steps)` call
- Requires minimal change to `train_jax.py` vs `train_ppo.py`
- Supports standard Gymnasium environments wrapped for vectorization
- For Phase 2 smoke test, we don't use any RL framework — just `jax.vmap` over `mjx.step`
- `sbx` is the path for Phase 5 (full training), not required for Phase 2-3

For the smoke test (Phase 2-3), we use raw JAX:
```python
step_fn = jax.jit(jax.vmap(lambda d, a: mjx.step(mx, d.replace(ctrl=a))))
```

---

## 5. Expected Throughput on RTX 3090

**RTX 3090 specs relevant to MJX:**
- 35.6 TFLOPS FP32
- 24 GB GDDR6X
- 936 GB/s memory bandwidth

**MJX throughput estimates for our scene:**
- Our system: 36 physics DOF, mesh collisions (condim=6), implicitfast integrator
- MJX benchmark data (from DeepMind papers/benchmarks): ~1M env-steps/sec for 30-DOF systems on A100, ~500k on V100
- RTX 3090 ≈ 70% of A100 throughput in practice
- Mesh collision overhead: ~2× slower than capsule-only

**Conservative estimate:**
| n_envs | Expected env-steps/sec (3090) |
|--------|-------------------------------|
| 16     | ~5,000–15,000 |
| 64     | ~20,000–60,000 |
| 256    | ~50,000–150,000 |
| 1024   | ~80,000–200,000 (memory-bound) |

**Target for smoke test:** >5,000 steps/sec with `n_envs=64`. Realistic expectation: 20,000–60,000 steps/sec.

**Baseline (CPU SB3):** Our current `n_envs=4` CPU setup gets ~200-500 env-steps/sec on shitter. MJX with 64 envs should be **40–300× faster** wall-clock.

---

## 6. Scope to First Working Smoke Test

| Task | Estimated time |
|------|----------------|
| `scripts/env_jax.py` — MJX GraspEnv skeleton | 2–3 hrs |
| `scripts/smoke_jax.py` — random-action test, timing | 30 min |
| `scripts/jax/requirements-jax.txt` | 10 min |
| Install JAX+CUDA on 3090, run smoke test | 30–60 min |
| Debug any MJX import/shape issues | 0–2 hrs |
| **Total** | **3–6 hours** |

---

## Summary Table

| Question | Answer | Status |
|----------|--------|--------|
| Does MJX support our geoms? | Yes — mesh (convex), plane, condim=6 all work | ✅ Green |
| Does MJX support our integrator? | Yes — implicitfast supported | ✅ Green |
| Tendons/equality constraints needed? | No — neither robot uses them | ✅ Green |
| multiccd supported? | No — drop the flag in MJX scene XML | ⚠️ Minor |
| Keyframe reset feasible in JIT? | Yes — precompute snapshot, restore in reset_fn | ✅ Green |
| Mink IK compatible with MJX? | No — use direct joint ctrl for port | ⚠️ Adapted |
| RL framework for training? | sbx (SB3 JAX) | ✅ Ready |
| Expected speedup vs CPU? | 40–300× with n_envs=64 on 3090 | ✅ Strong |
| Time to smoke test? | 3–6 hours | ✅ Feasible |

**VERDICT: Port is feasible. Proceed to Phase 2.**

---

## Known Limitations / Future Work

1. **EE-delta action space:** The JAX port uses direct joint ctrl (23D). A full JAX implementation of differential IK via Jacobian pseudoinverse would restore the original action space semantics. This is a follow-up task.
2. **multiccd:** Dropped in MJX scene. Contact quality may differ slightly from CPU sim, especially at high grasp forces. Needs empirical validation with matching experiments.
3. **Keyframe replay fidelity:** The MJX reset uses a pre-computed snapshot. The CPU grasp replay warmup (which settles contacts) is run once offline to generate this snapshot. If the grasp quality changes (new keyframes), the snapshot must be regenerated.
4. **Policy transfer:** The CPU-trained EE-delta policy **cannot** be directly loaded into the JAX env (different action space). JAX training produces a new policy family.
5. **Scene XML patching:** The MJX scene must drop `multiccd` and potentially re-tune solver parameters (solref/solimp) since MJX contact solver behavior differs slightly from CPU.
