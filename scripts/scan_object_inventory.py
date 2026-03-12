#!/usr/bin/env python3
"""Scan dataset directories and emit mesh inventories as CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

MESH_EXTS = {".obj", ".ply", ".stl", ".dae", ".glb", ".gltf"}


def object_id_from_path(mesh_path: Path, root: Path) -> str:
    rel = mesh_path.relative_to(root)
    return rel.parts[0] if rel.parts else "unknown"


def scan(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not root.exists():
        return rows

    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in MESH_EXTS:
            rows.append(
                {
                    "object_id": object_id_from_path(path, root),
                    "mesh_path": str(path),
                    "format": path.suffix.lower().lstrip("."),
                    "size_bytes": str(path.stat().st_size),
                }
            )
    return rows


def write_csv(rows: list[dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["object_id", "mesh_path", "format", "size_bytes"]
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ycb-root", default="data/objects/ycb/raw")
    parser.add_argument("--hope-root", default="data/objects/hope/raw")
    parser.add_argument("--ycb-out", default="data/objects/ycb/ycb_inventory.csv")
    parser.add_argument("--hope-out", default="data/objects/hope/hope_inventory.csv")
    args = parser.parse_args()

    ycb_rows = scan(Path(args.ycb_root))
    hope_rows = scan(Path(args.hope_root))

    write_csv(ycb_rows, Path(args.ycb_out))
    write_csv(hope_rows, Path(args.hope_out))

    print(f"YCB meshes indexed: {len(ycb_rows)} -> {args.ycb_out}")
    print(f"HOPE meshes indexed: {len(hope_rows)} -> {args.hope_out}")


if __name__ == "__main__":
    main()
