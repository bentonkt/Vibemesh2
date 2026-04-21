# Discord Log

Intended messages for channel #research (1486101259336552549). Discord MCP tool not available in this session — messages captured here for later manual relay or tool-restored replay.

---

## 2026-04-20 — MJX Port Scouting Report (branch: `jax-port`)

**Was it feasible?** Yes. No hard blockers on scene compatibility.
- xArm + LEAP hand: no tendons, no equality constraints → MJX-clean ✅
- YCB mesh geoms (condim=6): supported as convex hulls ✅
- implicitfast integrator: supported ✅
- multiccd flag: not in MJX — dropped (minor contact quality diff) ⚠️
- Mink IK: incompatible with JAX vmap → swapped to 23D direct joint ctrl for port (documented)

**What's on `jax-port`?**
- `docs/superpowers/specs/2026-04-20-mjx-port-feasibility.md` — full feasibility analysis
- `scripts/env_jax.py` — MJX GraspEnv skeleton: `put_model`, batched `jax.vmap` step/reset, disturbance forces, 45D obs, 23D joint-ctrl action space
- `scripts/smoke_jax.py` — random-action smoke test with throughput benchmark
- `scripts/jax/requirements-jax.txt` — `jax[cuda12]`, `mujoco-mjx>=3.7.0`, `sbx-rl`
- `docs/superpowers/specs/2026-04-20-mjx-smoke-test-results.md` — smoke test results

**Throughput numbers:**
- CPU only (Windows JAX) — N/A, can't test GPU throughput
- **BLOCKER: JAX CUDA wheels are Linux-only on PyPI.** The 3090 (Windows) gets CPU JAX only. `jax.devices()` → `[CpuDevice(id=0)]`. Model loads and JIT starts but on CPU (~50–200 steps/sec, same as baseline).
- Theoretical GPU throughput (per §5 of feasibility study): **20,000–60,000 env-steps/sec** with n_envs=64 on RTX 3090

**Blockers / follow-up:**
1. **CUDA on Windows** — need WSL2 with CUDA passthrough, OR install on Linux box to get actual GPU numbers. Alternative: try `warp-lang` backend (NVIDIA Warp, Windows-native CUDA).
2. **Mesh decimation** — xArm/LEAP meshes have coplanar faces >20 verts; MJX warns about performance impact. Recommend trimesh decimation before training.
3. **EE-delta IK** — JAX port uses direct joint ctrl (23D). Can add JAX-native Jacobian pseudoinverse later for EE-delta parity.

**Hours-to-working-GPU-training estimate:**
- If WSL2 CUDA or Linux box: ~1–2 hours to validate GPU smoke test + ~4–6 hours for full `sbx` PPO training loop
- If Windows-only: blocked until Warp backend tested or Linux box provisioned

`exp6-ee-delta` on shitter and 3090 are **untouched** — all JAX work in separate files.

