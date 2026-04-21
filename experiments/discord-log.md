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

---

## 2026-04-21 — Arc-2 Experiment Results

### Exp 1/3: exp7-ee-delta-survival (shitter)
```
exp7 (EE-delta + bonus=0.1, n_steps=512, 300k)
  eval ep_len: 37 → 135.0 @ 300k (prior best: exp5=100.2)
  Claim C7: partial — EE-delta + bonus=0.1 works, late breakthrough at 250k
  Note: survival_bonus=0.1 was insufficient to break out early, but policy
        still improved past exp5 eventually. Confirms C7 but slowly.
  Next: run exp8 with bonus=0.5 to test if higher bonus accelerates breakthrough
```

### Exp 2/3: exp8-ee-delta-sb05 (3090) — GOAL ACHIEVED
```
exp8 (EE-delta + bonus=0.5, n_steps=512, 300k)
  eval ep_len: 37 → 416.4 @ 200k steps  🎯 GOAL HIT (>= 400)
  Claim C7: STRONG-POSITIVE — EE-delta + bonus=0.5 = dramatic phase transition
  Claim C8: CONFIRMED — bonus >= 0.5 required for EE-delta action space
  Key finding: 175k→200k phase transition (+260 ep_len in one 25k window)
  Next: warm-start from best model to push toward 500/500
```

### Exp 3/3: exp10-warm-sb05 (3090 warm-start) — PERFECT SCORE
```
exp10 (warm-start from exp8 best + 500k steps)
  eval ep_len: 416 → 500.0 ± 0.00 @ 100k warm-start steps  🏆 PERFECT SCORE
  All 5 deterministic eval episodes hit 500-step timeout
  Consistent 500/500 from 400k warm-start onward
  Theoretical maximum performance — arc-2 COMPLETE
  Next: Arc-3 options — robustness testing, hardware transfer prep
```

### Arc-2 Summary
```
ARC 2 COMPLETE — Goal exceeded.

Started: eval 100.2 (exp5, arc-1 best)
Goal:    eval >= 400
Achieved: eval 416.4 (exp8, 200k steps)
Maximum:  eval 500/500 (exp10, 100k warm-start)

Key finding: EE-delta action space + survival_bonus=0.5 + PPO(n_steps=512)
discovered a qualitatively stable grasp strategy via phase transition at 175k–200k.
Warm-starting from the breakthrough model reaches theoretical maximum in 100k
additional steps.

Commits: worker/vibemesh-overseer-arc2 branch
Final report: experiments/final-report-arc2.md
```

