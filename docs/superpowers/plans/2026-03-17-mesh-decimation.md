# Mesh Decimation for Collision Geometry — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace primitive collision geoms (box/cylinder) with ~1000-face decimated mesh collision geoms for more realistic grasping contacts.

**Architecture:** Add a decimation step to the existing mesh processing script using `trimesh.simplify_quadric_decimation`, exporting a second STL per object. Update the scene XML to reference these new collision meshes instead of primitives.

**Tech Stack:** trimesh (already a dependency), MuJoCo MJCF XML

---

## File Structure

- Modify: `scripts/process_ycb_meshes.py` — add decimation + collision mesh export
- Modify: `scenes/ycb_resting.xml` — swap primitive collision geoms for mesh collision geoms
- Generated: `assets/ycb/<object>/google_16k/nontextured_collision.stl` (one per object)

---

### Task 1: Update mesh processing script with decimation

**Files:**
- Modify: `scripts/process_ycb_meshes.py`

- [ ] **Step 1: Update the docstring**

Replace the existing docstring to reflect the new decimation functionality:

```python
"""
Check, repair, and decimate YCB meshes for use in MuJoCo.

Issues addressed:
  - Non-watertight meshes (open boundaries cause incorrect contact normals)
  - Degenerate faces (zero-area triangles cause jitter)
  - Duplicate faces
  - Zero-length edges

Outputs per object:
  - nontextured_clean.stl     — full-resolution repaired mesh (visual geom)
  - nontextured_collision.stl — decimated to ~1000 faces (collision geom)
"""
```

- [ ] **Step 2: Add the `COLLISION_FACE_TARGET` constant**

After the `objects` list (line 24), add:

```python
COLLISION_FACE_TARGET = 1000
```

- [ ] **Step 3: Refactor the main loop to a linear flow with decimation**

Replace lines 69–98 (the `for` loop with `if/else` branching) with:

```python
for obj_name in objects:
    stl_path = os.path.join(
        REPO_ROOT, "assets", "ycb", obj_name, "google_16k", "nontextured.stl"
    )
    visual_path = os.path.join(
        REPO_ROOT, "assets", "ycb", obj_name, "google_16k", "nontextured_clean.stl"
    )
    collision_path = os.path.join(
        REPO_ROOT, "assets", "ycb", obj_name, "google_16k", "nontextured_collision.stl"
    )

    print(f"\n--- {obj_name} ---")
    mesh = trimesh.load(stl_path)

    print("  [before repair]")
    mesh_report(obj_name, mesh)

    # Always repair — these are safe no-ops on clean meshes
    mesh = repair_mesh(mesh)

    print("  [after repair]")
    mesh_report(obj_name, mesh)

    mesh.export(visual_path)
    print(f"  → Visual mesh saved: {visual_path}")

    # Decimate for collision
    repaired_faces = len(mesh.faces)
    if repaired_faces <= COLLISION_FACE_TARGET:
        print(f"  → Mesh already has {repaired_faces} faces (<= {COLLISION_FACE_TARGET}), skipping decimation")
        collision_mesh = mesh
    else:
        collision_mesh = mesh.simplify_quadric_decimation(COLLISION_FACE_TARGET)
        print(f"  → Decimated: {repaired_faces} → {len(collision_mesh.faces)} faces")

    collision_mesh.export(collision_path)
    print(f"  → Collision mesh saved: {collision_path}")
```

- [ ] **Step 4: Update the closing print**

Replace lines 100–102 with:

```python
print("\n" + "=" * 60)
print("Done. Visual: nontextured_clean.stl  Collision: nontextured_collision.stl")
print("=" * 60)
```

- [ ] **Step 5: Run the script**

Run: `cd /Users/bentontameling/Dev/Vibemesh2 && uv run python scripts/process_ycb_meshes.py`

Expected: Script prints repair and decimation reports for all 3 objects. Each object should show decimation from its original face count down to ~1000. No errors.

- [ ] **Step 6: Verify the collision meshes were created**

Run: `ls -la assets/ycb/*/google_16k/nontextured_collision.stl`

Expected: Three files exist, one per object.

- [ ] **Step 7: Commit**

```bash
git add scripts/process_ycb_meshes.py assets/ycb/*/google_16k/nontextured_collision.stl
git commit -m "Add mesh decimation step for collision geometry"
```

---

### Task 2: Update scene XML to use mesh collision geoms

**Files:**
- Modify: `scenes/ycb_resting.xml`

- [ ] **Step 1: Update the asset block**

Replace lines 13–15 (the three `<mesh>` entries) with:

```xml
    <mesh name="003_cracker_box_visual"     file="ycb/003_cracker_box/google_16k/nontextured_clean.stl"/>
    <mesh name="003_cracker_box_collision"  file="ycb/003_cracker_box/google_16k/nontextured_collision.stl"/>
    <mesh name="004_sugar_box_visual"       file="ycb/004_sugar_box/google_16k/nontextured_clean.stl"/>
    <mesh name="004_sugar_box_collision"    file="ycb/004_sugar_box/google_16k/nontextured_collision.stl"/>
    <mesh name="005_tomato_soup_can_visual"    file="ycb/005_tomato_soup_can/google_16k/nontextured_clean.stl"/>
    <mesh name="005_tomato_soup_can_collision" file="ycb/005_tomato_soup_can/google_16k/nontextured_collision.stl"/>
```

- [ ] **Step 2: Update the cracker box body (lines 23–33)**

Replace with:

```xml
    <!-- Cracker box: extents 0.0718 x 0.164 x 0.2134 m -->
    <body name="003_cracker_box" pos="-0.3 0 0.35">
      <freejoint/>
      <inertial pos="0 0 0" mass="0.411" diaginertia="0.002481 0.001736 0.001098"/>
      <geom type="mesh" mesh="003_cracker_box_visual" contype="0" conaffinity="0"/>
      <geom type="mesh" mesh="003_cracker_box_collision"
            friction="1.0 0.005 0.0001" solref="0.02 1" solimp="0.9 0.95 0.001"
            contype="1" conaffinity="1"/>
    </body>
```

- [ ] **Step 3: Update the sugar box body (lines 35–45)**

Replace with:

```xml
    <!-- Sugar box: extents 0.0495 x 0.0942 x 0.176 m -->
    <body name="004_sugar_box" pos="0 0 0.30">
      <freejoint/>
      <inertial pos="0 0 0" mass="0.514" diaginertia="0.001707 0.001432 0.000485"/>
      <geom type="mesh" mesh="004_sugar_box_visual" contype="0" conaffinity="0"/>
      <geom type="mesh" mesh="004_sugar_box_collision"
            friction="1.0 0.005 0.0001" solref="0.02 1" solimp="0.9 0.95 0.001"
            contype="1" conaffinity="1"/>
    </body>
```

- [ ] **Step 4: Update the tomato soup can body (lines 47–57)**

Replace with:

```xml
    <!-- Tomato soup can: diameter ~0.0678 m, height 0.1019 m -->
    <body name="005_tomato_soup_can" pos="0.3 0 0.25">
      <freejoint/>
      <inertial pos="0 0 0" mass="0.305" diaginertia="0.000352 0.000352 0.000175"/>
      <geom type="mesh" mesh="005_tomato_soup_can_visual" contype="0" conaffinity="0"/>
      <geom type="mesh" mesh="005_tomato_soup_can_collision"
            friction="0.8 0.005 0.0001" solref="0.02 1" solimp="0.9 0.95 0.001"
            contype="1" conaffinity="1"/>
    </body>
```

- [ ] **Step 5: Verify the scene loads in MuJoCo**

Run: `cd /Users/bentontameling/Dev/Vibemesh2 && uv run python -c "import mujoco; m = mujoco.MjModel.from_xml_path('scenes/ycb_resting.xml'); print('Loaded OK —', m.ngeom, 'geoms,', m.nmesh, 'meshes')"`

Expected: No errors. Should print `Loaded OK — 7 geoms, 6 meshes` (1 floor + 3 visual + 3 collision geoms, 6 mesh assets).

- [ ] **Step 6: Commit**

```bash
git add scenes/ycb_resting.xml
git commit -m "Use decimated mesh collision geoms instead of primitives"
```
