# MuJoCo Object Dataset Import Plan (Steps 2–3 kickoff)

This plan focuses on the **first actionable milestone** for Steps 2 and 3: getting YCB and HOPE object assets downloaded, normalized, and ready to be referenced in MuJoCo XML scenes.

## Goal of this first step

Create a reproducible object-asset pipeline that works both:
- locally (laptops)
- on the shared lab machine/server

so every simulation run references the same object meshes, scales, and metadata.

## Scope for this first milestone

1. Download the YCB dataset assets (primary source: `ycb-tools` downloader).
2. Download the HOPE dataset assets (from the provided Drive folder).
3. Organize both datasets into a consistent project directory structure.
4. Run a lightweight validation pass to ensure meshes are loadable and unit scales are explicit.
5. Generate a machine-readable object manifest to feed later MuJoCo import scripts.

---

## Proposed directory layout

Use a data directory that can be mirrored between local and server machines.

```text
project_root/
  data/
    objects/
      ycb/
        raw/
        processed/
      hope/
        raw/
        processed/
      manifests/
        objects_manifest.yaml
  scripts/
    fetch_ycb.sh
    fetch_hope.md
    validate_meshes.py
    build_object_manifest.py
```

Notes:
- `raw/` keeps original downloaded assets untouched.
- `processed/` contains converted/cleaned meshes prepared for MuJoCo.
- `objects_manifest.yaml` is the single source of truth for object names, mesh paths, and scale factors.

---

## Step-by-step plan

### 1) Freeze environment assumptions

- Define one Python environment version (e.g., Python 3.10+).
- Pin key geometry dependencies used in preprocessing/validation:
  - `trimesh`
  - `numpy`
  - `pyyaml`
- Record versions in a requirements file for reproducibility.

Deliverable: `requirements-objects.txt` (or equivalent).

### 2) Import YCB assets

Primary reference: `download_ycb_dataset.py` from ycb-tools.

Plan:
- Add a wrapper script (`scripts/fetch_ycb.sh`) that:
  1. creates `data/objects/ycb/raw`
  2. runs the downloader with explicit output directory
  3. logs exact downloader commit/hash or script version used
- Prefer downloading textured + collision-relevant meshes where available.

Validation after download:
- Confirm non-empty object folders.
- Confirm each object has at least one mesh file (`.obj`, `.ply`, or `.stl`).
- Store a simple inventory (`ycb_inventory.csv`).

### 3) Import HOPE assets

Source: provided Google Drive folder.

Plan:
- Manually download or scripted sync (depending on permissions/tools available).
- Save archive(s) under `data/objects/hope/raw`.
- Unpack with folder names preserved.
- Document exact snapshot date and source URL in a short metadata file.

Validation after download:
- Confirm mesh presence for each HOPE object.
- Build `hope_inventory.csv` with object id, mesh path, and file format.

### 4) Normalize and preprocess meshes

For each object in YCB and HOPE:
- Ensure axis orientation and units are clearly documented.
- If needed, convert meshes to a MuJoCo-friendly format (`.obj` or `.stl`) into `processed/`.
- Keep original mesh + processed mesh mapping in a CSV.

At this stage, do **not** tune dynamics yet (mass/inertia/friction tuning is next milestone).

### 5) Build unified object manifest

Create `data/objects/manifests/objects_manifest.yaml` with fields:
- `dataset`: `ycb` or `hope`
- `object_id`
- `display_name`
- `mesh_visual`
- `mesh_collision`
- `scale`
- `units`
- `notes`

This manifest becomes input for later MuJoCo XML generation and object randomization.

### 6) Smoke-test mesh loading

Write `scripts/validate_meshes.py` to:
- iterate all manifest entries
- load each mesh with `trimesh`
- report missing files/invalid geometry
- output pass/fail summary

Success criterion for this milestone:
- 100% of manifest-listed objects load without parser errors.

---

## Acceptance criteria for the “import dataset” milestone

- YCB and HOPE raw assets are present in the agreed directory structure.
- A processed mesh directory exists with documented transformations.
- An object manifest is created and version-controlled.
- Automated mesh validation passes.
- The same commands work on both local machine and server path conventions.

---

## Risks and mitigations

1. **YCB downloader link/tool changes**
   - Mitigation: pin downloader commit and keep fallback mirror URLs in docs.

2. **HOPE download friction (Drive permissions/manual steps)**
   - Mitigation: document exact manual download procedure and expected checksums.

3. **Inconsistent units/orientation across datasets**
   - Mitigation: add explicit per-object `scale` + `units` fields in manifest and validate dimensions before MuJoCo import.

4. **Dataset size / storage limits on laptops**
   - Mitigation: support subset download list for quick local tests and full dataset on server.

---

## Immediate next actions (what to do now)

1. Set up folder skeleton under `data/objects/...`.
2. Run YCB download into `data/objects/ycb/raw`.
3. Download HOPE snapshot into `data/objects/hope/raw`.
4. Generate `ycb_inventory.csv` + `hope_inventory.csv`.
5. Draft `objects_manifest.yaml` with at least 5 pilot objects (mix of YCB + HOPE).
6. Run mesh validation script and fix missing/invalid entries.

Once this is complete, we can move to Step 2/3 integration: adding these objects into MuJoCo scenes with stable contact parameters and repeatable grasp/release trajectories.
