# Mesh Decimation for Collision Geometry

## Goal

Replace primitive collision geoms (box/cylinder) with decimated mesh collision geoms (~1000 faces) so that grasping contacts better match the actual object shape.

## Approach

Use trimesh's quadric decimation (`trimesh.simplify_quadric_decimation`) — no new dependencies needed.

**Convexity note:** MuJoCo automatically computes the convex hull of any mesh used as a collision geom. This means concave details (e.g. corrugations on the cracker box) will be filled in for collision purposes. This is acceptable for our grasping task — the convex hull of a ~1000-face decimated mesh is a much better approximation than a box or cylinder primitive, and convex collision is fast and stable.

## Changes

### 1. Update `scripts/process_ycb_meshes.py`

Refactor the script so that repair, decimation, and both exports happen in a single linear flow (removing the current `if needs_repair / else` branching). For each object:

1. **Load** `nontextured.stl`
2. **Repair** (always run: fix winding, normals, merge vertices, remove degenerate faces, fill holes — these are safe no-ops on clean meshes)
3. **Export** repaired mesh as `nontextured_clean.stl` (visual geom)
4. **Decimate** repaired mesh to ~1000 faces using `trimesh.simplify_quadric_decimation`. If the mesh already has <=1000 faces, skip decimation and use the repaired mesh directly.
5. **Export** decimated (or original if skipped) mesh as `nontextured_collision.stl` (collision geom)
6. **Print** face count before and after decimation for verification

### 2. Update `scenes/ycb_resting.xml`

Rename existing mesh assets to add `_visual` suffix, add `_collision` mesh entries, and update all geom references to match.

**Asset block** — rename existing entries and add collision meshes:
```xml
<!-- was: name="003_cracker_box" -->
<mesh name="003_cracker_box_visual"    file="ycb/003_cracker_box/google_16k/nontextured_clean.stl"/>
<mesh name="003_cracker_box_collision" file="ycb/003_cracker_box/google_16k/nontextured_collision.stl"/>
```

**Body block** — replace primitive collision geom with mesh collision geom, update visual geom reference:
```xml
<body name="003_cracker_box" pos="-0.3 0 0.35">
  <freejoint/>
  <inertial pos="0 0 0" mass="0.411" diaginertia="0.002481 0.001736 0.001098"/>
  <!-- visual: mesh="003_cracker_box" renamed to "003_cracker_box_visual" -->
  <geom type="mesh" mesh="003_cracker_box_visual" contype="0" conaffinity="0"/>
  <!-- collision: was type="box", now type="mesh" -->
  <geom type="mesh" mesh="003_cracker_box_collision"
        friction="1.0 0.005 0.0001" solref="0.02 1" solimp="0.9 0.95 0.001"
        contype="1" conaffinity="1"/>
</body>
```

Same pattern for `004_sugar_box` and `005_tomato_soup_can`.

## Steps to Apply

1. Update `scripts/process_ycb_meshes.py` with the decimation changes
2. Run `python scripts/process_ycb_meshes.py` to generate both `nontextured_clean.stl` and `nontextured_collision.stl` for each object
3. Update `scenes/ycb_resting.xml` with the new asset names and collision geoms
4. Load the scene in MuJoCo and verify: no `mjERROR` in terminal output, objects rest stably on the floor, and contact points appear on the mesh surface in the viewer

## Files Modified

- `scripts/process_ycb_meshes.py` — add decimation step and second export
- `scenes/ycb_resting.xml` — rename mesh assets, swap primitive collision geoms for mesh collision geoms

## Files Generated (by running the script)

- `<object>/google_16k/nontextured_clean.stl` (one per object, regenerated)
- `<object>/google_16k/nontextured_collision.stl` (one per object, new)

(Paths relative to `assets/ycb/`)
