# Vibemesh2

MuJoCo simulation environment for audio-based slip recovery using a LEAP Hand on a UFactory xArm 7. The goal is to build a sim-validated reactive policy that detects object slip from audio and executes corrective finger motions, then transfer it to real hardware.

**PI / Advisor:** Uksang
**Implementer:** Benton (Zyfex)

---

## Project Goal

1. Build a stable grasping environment in MuJoCo with realistic physics
2. Train a reactive RL policy that detects slip and corrects finger positions
3. Transfer the policy to hardware by swapping the oracle slip signal for a learned audio model

---

## Setup

### Prerequisites

- Python 3.10+
- Conda environment recommended

### Install dependencies

```bash
git clone --recurse-submodules <repo-url>
cd Vibemesh2
pip install mujoco==3.6.0 mink==1.1.0 loop-rate-limiters trimesh
```

> The `mink` submodule (in `mink/`) provides the xArm 7 and LEAP Hand MJCF models and the IK library. It is included as a git submodule — make sure to clone with `--recurse-submodules`.

### Object datasets

Object meshes are **not committed** to the repo (large binaries). You need to process them before running scenes.

**YCB objects** — download and process:
```bash
# 1. Download YCB meshes (requires the YCB download script)
python scripts/download_ycb_dataset.py

# 2. Process into MuJoCo-ready meshes + MJCF
python scripts/process_ycb.py
```

**HOPE objects** — place the HOPE dataset zip in `HOPE/`, then:
```bash
python scripts/process_hope.py
```

Both scripts output to `data/{ycb,hope}/processed/` and generate MJCF files in `mjcf/objects/{ycb,hope}/`.

---

## Running the interactive grasp scene

```bash
python scripts/arrow_key_grasp.py
python scripts/arrow_key_grasp.py --object 006_mustard_bottle   # YCB object
python scripts/arrow_key_grasp.py --object Corn                 # HOPE object
python scripts/arrow_key_grasp.py --collision                   # show collision meshes
```

### Controls

| Key | Action |
|---|---|
| G | Close grasp |
| R | Open grasp |
| Arrow Up / Down | Move hand forward / back |
| Arrow Left / Right | Move hand left / right |
| Page Up / Page Down | Move hand up / down |
| `,` / `.` | Roll hand |

The arm follows a real-time IK solver (via `mink`). The hand joints are interpolated between open and closed postures.

---

## Repository structure

```
Vibemesh2/
├── mink/                        # Git submodule: xArm + LEAP Hand models + IK library
├── mjcf/
│   └── objects/
│       ├── ycb/                 # Generated MJCF files for YCB objects
│       └── hope/                # Generated MJCF files for HOPE objects
├── scripts/
│   ├── arrow_key_grasp.py       # Interactive grasping scene (main entry point)
│   ├── test_scene.py            # Scene builder used by all scripts
│   ├── process_ycb.py           # YCB mesh processing pipeline
│   ├── process_hope.py          # HOPE mesh processing pipeline
│   └── hardcoded_grasp.py       # State-machine grasp script (earlier prototype)
├── scenes/
│   └── view_collisions.xml      # Standalone scene for inspecting collision meshes
├── config/
│   ├── ycb_objects.json         # Object metadata (mass, bbox) for YCB
│   └── hope_objects.json        # Object metadata for HOPE
├── docs/                        # Reference documents and design specs
└── data/                        # Processed meshes (gitignored, regenerate with scripts)
```

---

## Physics setup

| Parameter | Value | Notes |
|---|---|---|
| Timestep | 1 ms | Stable for contact-rich manipulation |
| Control rate | 200 Hz (5 substeps) | Real-time |
| Integrator | `implicitfast` | Stable with stiff contacts |
| Contact cone | `elliptic` | More accurate friction |
| Hand-object friction | 2× realistic | Sliding: 1.6, torsional: 0.02, rolling: 0.002 |
| Hand collision | Actual mesh convex hulls | No proxy spheres |

### Object physics

Objects use a visual mesh (textured, for rendering) and a separate convex hull collision mesh. Mass values are either from published YCB data or estimated from approximate product weights for HOPE objects.

---

## Current status

| Phase | Description | Status |
|---|---|---|
| 0 | Server access & environment setup | Done |
| 1 | MuJoCo installation & verification | Done |
| 2 | Robot models (xArm + LEAP Hand) | Done |
| 3 | Object setup (YCB + HOPE datasets) | Done |
| 4 | Contact & slip recovery RL policy | Not started |

Phase 4 is the next milestone: designing the Gym environment wrapper, oracle slip signal from MuJoCo contact data, and training a PPO policy on the tomato soup can before generalizing to the full object set.

---

## Key design decisions

- **Oracle slip in sim, audio on hardware.** The sim policy uses MuJoCo's ground-truth contact data to detect slip direction. On hardware, this signal will be replaced by a learned audio model, enabling sim-to-real transfer without retraining the manipulation policy.
- **Mesh-based hand contacts.** Rather than proxy spheres, the LEAP Hand's actual convex hull collision meshes are used for hand-object contact. This gives more accurate contact patches at the cost of slightly more solver work.
- **HOPE + YCB.** Both datasets are supported. Objects are loaded by name: YCB objects use numeric IDs (e.g. `005_tomato_soup_can`), HOPE objects use CamelCase names (e.g. `Corn`).
