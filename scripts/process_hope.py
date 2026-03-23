"""HOPE mesh processing pipeline for MuJoCo simulation.

Extracts HOPE meshes from the dataset zip and processes them into
MuJoCo-ready visual + collision mesh pairs with MJCF XML files.

Steps per object:
  1. Extract OBJ / MTL / JPG from the nested zip
  2. Generate a convex hull for collision
  3. Estimate mass from density * volume (or use known values)
  4. Write a per-object MJCF XML

Usage:
    python scripts/process_hope.py
    python scripts/process_hope.py --objects AlphabetSoup,Corn,Tuna
    python scripts/process_hope.py --zip path/to/outer.zip
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import trimesh

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Approximate masses (kg) for HOPE objects
# ---------------------------------------------------------------------------
HOPE_MASSES_KG: dict[str, float] = {
    "AlphabetSoup":      0.425,
    "BBQSauce":          0.425,
    "Butter":            0.454,
    "Cherries":          0.425,
    "ChocolatePudding":  0.102,
    "Cookies":           0.170,
    "Corn":              0.425,
    "CreamCheese":       0.227,
    "GranolaBars":       0.175,
    "GreenBeans":        0.425,
    "Ketchup":           0.397,
    "MacaroniAndCheese": 0.206,
    "Mayo":              0.443,
    "Milk":              0.500,
    "Mushrooms":         0.284,
    "Mustard":           0.226,
    "OrangeJuice":       0.450,
    "Parmesan":          0.227,
    "Peaches":           0.425,
    "PeasAndCarrots":    0.425,
    "Pineapple":         0.540,
    "Popcorn":           0.283,
    "Raisins":           0.170,
    "SaladDressing":     0.340,
    "Spaghetti":         0.454,
    "TomatoSauce":       0.425,
    "Tuna":              0.170,
    "Yogurt":            0.227,
}

FALLBACK_DENSITY = 800.0  # kg/m^3 — typical for packaged goods

# ---------------------------------------------------------------------------
# Contact defaults (same as YCB pipeline)
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

INNER_ZIP_NAME = "HOPE-dataset-release/hope_meshes_full.zip"


# ---------------------------------------------------------------------------
# Zip helpers
# ---------------------------------------------------------------------------
def _find_outer_zip(hope_dir: Path) -> Path:
    zips = sorted(hope_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No zip file found in {hope_dir}")
    return zips[0]


def _list_objects(outer_zip: Path) -> list[str]:
    """Return sorted list of object names found in the inner mesh zip."""
    with zipfile.ZipFile(outer_zip) as oz:
        data = oz.read(INNER_ZIP_NAME)
    with zipfile.ZipFile(io.BytesIO(data)) as iz:
        names = {
            Path(n).stem
            for n in iz.namelist()
            if n.lower().endswith(".obj")
        }
    return sorted(names)


def _rmtree(path: Path) -> None:
    """Remove a directory tree, ignoring errors (e.g. Windows file locks)."""
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _extract_object(outer_zip: Path, object_name: str, dest_dir: Path) -> None:
    """Extract OBJ + MTL + JPG for one object into dest_dir/{object_name}/."""
    obj_dir = dest_dir / object_name
    _rmtree(obj_dir)
    obj_dir.mkdir(parents=True)

    with zipfile.ZipFile(outer_zip) as oz:
        data = oz.read(INNER_ZIP_NAME)
    with zipfile.ZipFile(io.BytesIO(data)) as iz:
        for member in iz.namelist():
            if Path(member).stem == object_name:
                content = iz.read(member)
                (obj_dir / Path(member).name).write_bytes(content)


# ---------------------------------------------------------------------------
# Mesh helpers (same logic as process_ycb.py)
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


def _get_mass(object_name: str, mesh: trimesh.Trimesh) -> float:
    mass = HOPE_MASSES_KG.get(object_name)
    if mass is not None:
        return mass
    # mesh is in mm, so volume is mm³; convert to m³ before applying density
    return max(FALLBACK_DENSITY * _mesh_volume(mesh) * 1e-9, 1e-4)


# ---------------------------------------------------------------------------
# Per-object processing
# ---------------------------------------------------------------------------
def process_object(
    object_name: str,
    outer_zip: Path,
    raw_dir: Path,
    processed_dir: Path,
    mjcf_dir: Path,
) -> dict:
    # Extract raw files
    _extract_object(outer_zip, object_name, raw_dir)
    raw_obj = raw_dir / object_name / f"{object_name}.obj"

    mesh = _load_mesh(raw_obj)  # mesh is in mm; MuJoCo will scale via MJCF

    # Export visual and collision meshes
    out_dir = processed_dir / object_name
    _rmtree(out_dir)
    out_dir.mkdir(parents=True)

    visual_path = out_dir / "visual.obj"
    collision_path = out_dir / "collision.obj"

    # Copy raw files so OBJ can find its MTL + JPG
    for f in (raw_dir / object_name).iterdir():
        shutil.copy2(f, out_dir / f.name)
    # Rename the main OBJ to visual.obj
    (out_dir / f"{object_name}.obj").rename(visual_path)
    # Rename MTL to match
    mtl_src = out_dir / f"{object_name}.mtl"
    if mtl_src.exists():
        # Update the OBJ's mtllib reference
        text = visual_path.read_text(encoding="utf-8", errors="replace")
        text = text.replace(f"{object_name}.mtl", "visual.mtl")
        visual_path.write_text(text, encoding="utf-8")
        mtl_src.rename(out_dir / "visual.mtl")

    try:
        hull = mesh.convex_hull
    except Exception:
        LOGGER.warning("%s: convex hull failed, using bounding box", object_name)
        extents = np.maximum(mesh.bounding_box.extents, 1e-4)
        hull = trimesh.creation.box(extents=extents)
    hull.export(collision_path)

    mass = _get_mass(object_name, mesh)
    xml_path = _write_mjcf(object_name, visual_path, collision_path, mass, mjcf_dir)

    bbox_m = (mesh.bounding_box.extents * 0.001).tolist()
    LOGGER.info("OK  %-25s  mass=%.3fkg  bbox=%s", object_name, mass, [f"{v:.3f}" for v in bbox_m])

    return {
        "object_id": object_name,
        "mass_kg": mass,
        "bbox_extents_m": bbox_m,
        "visual_mesh": str(visual_path),
        "collision_mesh": str(collision_path),
    }


def _write_mjcf(
    object_name: str,
    visual_path: Path,
    collision_path: Path,
    mass: float,
    mjcf_dir: Path,
) -> Path:
    xml_path = mjcf_dir / f"{object_name}.xml"

    visual_rel = os.path.relpath(visual_path, start=mjcf_dir)
    collision_rel = os.path.relpath(collision_path, start=mjcf_dir)

    root = ET.Element("mujoco", attrib={"model": object_name})
    ET.SubElement(root, "compiler", attrib={"angle": "radian", "inertiafromgeom": "true"})

    default = ET.SubElement(root, "default")

    vis_default = ET.SubElement(default, "default", attrib={"class": f"{object_name}_visual"})
    ET.SubElement(vis_default, "geom", attrib={
        "type": "mesh", "group": "2", "contype": "0", "conaffinity": "0", "density": "0",
    })

    col_default = ET.SubElement(default, "default", attrib={"class": f"{object_name}_collision"})
    col_geom_el = ET.SubElement(col_default, "geom", attrib={"type": "mesh", "group": "3"})
    for key, val in COLLISION_DEFAULTS.items():
        col_geom_el.set(key, val)

    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", attrib={
        "name": f"{object_name}_visual_mesh", "file": visual_rel, "scale": "0.001 0.001 0.001",
    })
    ET.SubElement(asset, "mesh", attrib={
        "name": f"{object_name}_collision_mesh", "file": collision_rel, "scale": "0.001 0.001 0.001",
    })

    worldbody = ET.SubElement(root, "worldbody")
    body = ET.SubElement(worldbody, "body", attrib={"name": object_name, "pos": "0 0 0"})
    ET.SubElement(body, "joint", attrib={
        "name": f"{object_name}_freejoint", "type": "free", "damping": "0.1",
    })
    ET.SubElement(body, "geom", attrib={
        "name": f"{object_name}_visual_geom",
        "class": f"{object_name}_visual",
        "type": "mesh",
        "mesh": f"{object_name}_visual_mesh",
    })
    col_attrib = {
        "name": f"{object_name}_collision_geom",
        "class": f"{object_name}_collision",
        "type": "mesh",
        "mesh": f"{object_name}_collision_mesh",
        "mass": f"{mass:.6g}",
    }
    col_attrib.update(COLLISION_DEFAULTS)
    ET.SubElement(body, "geom", attrib=col_attrib)

    mjcf_dir.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    return xml_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Process HOPE meshes for MuJoCo.")
    parser.add_argument("--zip", type=Path, default=None,
                        help="Path to outer HOPE zip (default: first *.zip in HOPE/)")
    parser.add_argument("--objects", type=str, default="all",
                        help="'all' or comma-separated names (e.g. AlphabetSoup,Corn)")
    parser.add_argument("--raw-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "hope" / "raw")
    parser.add_argument("--processed-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "hope" / "processed")
    parser.add_argument("--mjcf-dir", type=Path,
                        default=PROJECT_ROOT / "mjcf" / "objects" / "hope")
    parser.add_argument("--config-out", type=Path,
                        default=PROJECT_ROOT / "config" / "hope_objects.json")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    outer_zip = args.zip or _find_outer_zip(PROJECT_ROOT / "HOPE")
    LOGGER.info("Using zip: %s", outer_zip)

    all_objects = _list_objects(outer_zip)

    if args.objects.strip().lower() == "all":
        object_names = all_objects
    else:
        object_names = sorted(s.strip() for s in args.objects.split(",") if s.strip())

    LOGGER.info("Processing %d objects", len(object_names))

    results, errors = [], []
    for name in object_names:
        try:
            info = process_object(name, outer_zip, args.raw_dir, args.processed_dir, args.mjcf_dir)
            results.append(info)
        except Exception as exc:
            LOGGER.error("FAIL %-25s  %s", name, exc)
            errors.append({"object_id": name, "error": str(exc)})

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
