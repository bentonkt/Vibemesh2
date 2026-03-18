# Vibemesh2

## Git Rules

- Never add any mention of Claude (Co-Authored-By, etc.) in commit messages.

## Project Overview

MuJoCo simulation environment for robotic grasping with a LEAP hand and YCB/HOPE dataset objects.

## Project Setup Document

See `VibeMesh 2.0 Mujoco Setup (2).pdf` in the repo root for the full project plan from Uksang covering:
- Server access & environment consistency
- MuJoCo installation & verification
- Robot models (xArm + LEAP Hand) — models sourced from `mink/` submodule
- Object setup (YCB/HOPE objects)
- Contact & slip recovery (RL policy for reactive manipulation)

## Team & Roles

- **Zyfex (Benton)** — implementing the MuJoCo setup, importing objects, tuning simulation parameters
- **Uksang** — advisor/mentor providing guidance on simulation setup and contact modeling

## Object Datasets

- **YCB dataset**: 3D scanned objects common in robotics. Download script: `scripts/download_ycb_dataset.py` (originally from https://github.com/sea-bass/ycb-tools)
- **HOPE dataset**: Available at https://drive.google.com/drive/folders/1Hj5K9RIdcNxBFiU8qG0-oL3Ryd9f2gOY

## Physics Parameters Guidance (from Uksang)

### Mass & Inertia
- Each object needs a rough total mass (look up real values online)
- Calculate inertia from geometry assuming homogeneous density
- Values don't need to be scientifically precise, just approximately correct

### Contact Parameters
- MuJoCo uses penalty-based contact: overlapping meshes get "spring-like" penalty forces
- Key parameters: `solref` and `solimp` in scene XML
- **High stiffness** = more jittery (high penalty for overlap)
- **Low stiffness** = more penetration but smoother contacts
- Reference: https://mujoco.readthedocs.io/en/stable/computation/index.html

### Mesh Quality
- 3D scanned meshes can be messy, causing unrealistic contact behavior
- If meshes look bad, use remeshing libraries: pymesh, trimesh, or open3d
- Cleaner meshes = more stable contacts

### Friction
- Friction can be cranked up high without risking instability
- Needed for the hand to actually grip and lift objects

## Current Status (as of ~2/26/26)

- YCB objects imported and resting stably in simulation
- Contact jittering resolved by tuning contact parameters
- Working on friction tuning so the LEAP hand can pick up objects
- Have a script that closes fingers against thumb and moves the hand
- Objects not yet being picked up successfully — friction too low
