"""Minimal YCB mesh processing pipeline for stable MuJoCo simulation.

For each object in models/ycb/{id}/:
  1. Load the best available mesh with trimesh
  2. Generate a convex hull for collision
  3. Look up real mass (fallback: density * volume)
  4. Write a per-object MJCF XML with visual/collision separation

Usage:
    python scripts/process_ycb.py                          # process all
    python scripts/process_ycb.py --objects 002_master_chef_can,025_mug
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# YCB masses (kg) — official YCB-Video 21-object subset
# ---------------------------------------------------------------------------
YCB_MASSES_KG: dict[str, float] = {
    "002_master_chef_can": 0.414,
    "003_cracker_box": 0.411,
    "004_sugar_box": 0.514,
    "005_tomato_soup_can": 0.349,
    "006_mustard_bottle": 0.603,
    "007_tuna_fish_can": 0.171,
    "008_pudding_box": 0.187,
    "009_gelatin_box": 0.097,
    "010_potted_meat_can": 0.370,
    "011_banana": 0.066,
    "019_pitcher_base": 0.178,
    "021_bleach_cleanser": 1.131,
    "024_bowl": 0.147,
    "025_mug": 0.118,
    "035_power_drill": 0.895,
    "036_wood_block": 0.729,
    "037_scissors": 0.082,
    "040_large_marker": 0.0158,
    "051_large_clamp": 0.125,
    "052_extra_large_clamp": 0.202,
    "061_foam_brick": 0.028,
}

FALLBACK_DENSITY = 1000.0  # kg/m^3

# ---------------------------------------------------------------------------
# Contact defaults (object-level; hand-object overrides applied at scene level)
# ---------------------------------------------------------------------------
COLLISION_DEFAULTS = {
    "condim": "3",
    "friction": "0.8 0.01 0.001",
    "solref": "0.008 1",
    "solimp": "0.97 0.995 0.001",
    "margin": "0.001",
    "gap": "0",
    "priority": "1",
}

MESH_SUFFIXES = {".obj", ".stl", ".ply"}


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------
def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False, force="mesh", maintain_order=True)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"No geometry in {path}")
        mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError(f"Unsupported type from {path}: {type(loaded)!r}")
    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    return mesh


def _find_best_mesh(object_dir: Path) -> Path:
    """Pick the best mesh file from an object directory."""
    # Prefer textured.obj, then any .obj, then .stl, then .ply
    for name in ("textured.obj", "nontextured.stl", "nontextured.ply"):
        candidate = object_dir / name
        if candidate.exists():
            return candidate
    # Fallback: any mesh file
    for path in sorted(object_dir.iterdir()):
        if path.suffix.lower() in MESH_SUFFIXES:
            return path
    raise FileNotFoundError(f"No mesh files in {object_dir}")


def _mesh_volume(mesh: trimesh.Trimesh) -> float:
    try:
        hull = mesh if mesh.is_volume and mesh.volume > 0 else mesh.convex_hull
        vol = abs(float(hull.volume))
        if vol > 0:
            return vol
    except Exception:
        pass
    extents = np.maximum(mesh.bounding_box.extents, 1e-6)
    return float(np.prod(extents))


def _get_mass(object_id: str, mesh: trimesh.Trimesh) -> float:
    mass = YCB_MASSES_KG.get(object_id)
    if mass is not None:
        return mass
    return max(FALLBACK_DENSITY * _mesh_volume(mesh), 1e-4)


# ---------------------------------------------------------------------------
# Per-object processing
# ---------------------------------------------------------------------------
def process_object(
    object_id: str,
    models_dir: Path,
    processed_dir: Path,
    mjcf_dir: Path,
) -> dict:
    """Process one object: convex hull + MJCF."""
    source_dir = models_dir / object_id
    source_mesh_path = _find_best_mesh(source_dir)

    mesh = _load_mesh(source_mesh_path)

    # Export visual and collision meshes
    out_dir = processed_dir / object_id
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    visual_path = out_dir / "visual.obj"
    collision_path = out_dir / "collision.obj"

    mesh.export(visual_path)

    try:
        hull = mesh.convex_hull
    except Exception:
        LOGGER.warning("%s: convex hull failed, using bounding box", object_id)
        extents = np.maximum(mesh.bounding_box.extents, 1e-4)
        hull = trimesh.creation.box(extents=extents)
    hull.export(collision_path)

    # Copy texture files alongside visual mesh
    for tex_file in source_dir.iterdir():
        if tex_file.suffix.lower() in (".png", ".jpg", ".jpeg", ".mtl"):
            shutil.copy2(tex_file, out_dir / tex_file.name)

    mass = _get_mass(object_id, mesh)

    # Write MJCF
    xml_path = _write_mjcf(object_id, visual_path, collision_path, mass, mjcf_dir)

    bbox = mesh.bounding_box.extents.tolist()
    LOGGER.info("OK  %-30s  mass=%.3fkg  bbox=%s  -> %s", object_id, mass, bbox, xml_path)

    return {
        "object_id": object_id,
        "mass_kg": mass,
        "bbox_extents_m": bbox,
        "source_mesh": str(source_mesh_path),
        "visual_mesh": str(visual_path),
        "collision_mesh": str(collision_path),
    }


def _write_mjcf(
    object_id: str,
    visual_path: Path,
    collision_path: Path,
    mass: float,
    mjcf_dir: Path,
) -> Path:
    xml_path = mjcf_dir / f"{object_id}.xml"

    visual_rel = os.path.relpath(visual_path, start=mjcf_dir)
    collision_rel = os.path.relpath(collision_path, start=mjcf_dir)

    root = ET.Element("mujoco", attrib={"model": object_id})
    ET.SubElement(root, "compiler", attrib={
        "angle": "radian",
        "inertiafromgeom": "true",
    })

    # Defaults
    default = ET.SubElement(root, "default")

    vis_default = ET.SubElement(default, "default", attrib={"class": f"{object_id}_visual"})
    ET.SubElement(vis_default, "geom", attrib={
        "type": "mesh", "group": "2", "contype": "0", "conaffinity": "0", "density": "0",
    })

    col_default = ET.SubElement(default, "default", attrib={"class": f"{object_id}_collision"})
    col_geom = ET.SubElement(col_default, "geom", attrib={"type": "mesh", "group": "3"})
    for key, val in COLLISION_DEFAULTS.items():
        col_geom.set(key, val)

    # Assets
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", attrib={
        "name": f"{object_id}_visual_mesh", "file": visual_rel,
    })
    ET.SubElement(asset, "mesh", attrib={
        "name": f"{object_id}_collision_mesh", "file": collision_rel,
    })

    # Body
    worldbody = ET.SubElement(root, "worldbody")
    body = ET.SubElement(worldbody, "body", attrib={"name": object_id, "pos": "0 0 0"})
    ET.SubElement(body, "joint", attrib={
        "name": f"{object_id}_freejoint", "type": "free", "damping": "0.1",
    })
    ET.SubElement(body, "geom", attrib={
        "name": f"{object_id}_visual_geom",
        "class": f"{object_id}_visual",
        "type": "mesh",
        "mesh": f"{object_id}_visual_mesh",
    })

    col_attrib = {
        "name": f"{object_id}_collision_geom",
        "class": f"{object_id}_collision",
        "type": "mesh",
        "mesh": f"{object_id}_collision_mesh",
        "mass": f"{mass:.6g}",
    }
    col_attrib.update(COLLISION_DEFAULTS)
    ET.SubElement(body, "geom", attrib=col_attrib)

    # Write
    mjcf_dir.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    return xml_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Process YCB meshes for MuJoCo.")
    parser.add_argument(
        "--objects", type=str, default="all",
        help="'all' or comma-separated IDs (e.g. 002_master_chef_can,025_mug)",
    )
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models" / "ycb")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "ycb" / "processed")
    parser.add_argument("--mjcf-dir", type=Path, default=PROJECT_ROOT / "mjcf" / "objects" / "ycb")
    parser.add_argument("--config-out", type=Path, default=PROJECT_ROOT / "config" / "ycb_objects.json")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # Discover objects
    if args.objects.strip().lower() == "all":
        object_ids = sorted(
            d.name for d in args.models_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
    else:
        object_ids = sorted(s.strip() for s in args.objects.split(",") if s.strip())

    LOGGER.info("Processing %d objects from %s", len(object_ids), args.models_dir)

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.mjcf_dir.mkdir(parents=True, exist_ok=True)

    results = []
    errors = []
    for object_id in object_ids:
        try:
            info = process_object(object_id, args.models_dir, args.processed_dir, args.mjcf_dir)
            results.append(info)
        except Exception as exc:
            LOGGER.error("FAIL %-30s  %s", object_id, exc)
            errors.append({"object_id": object_id, "error": str(exc)})

    # Write config
    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    args.config_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOGGER.info("Config written to %s", args.config_out)

    LOGGER.info("Done: %d succeeded, %d failed", len(results), len(errors))
    if errors:
        for e in errors:
            LOGGER.error("  %s: %s", e["object_id"], e["error"])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
