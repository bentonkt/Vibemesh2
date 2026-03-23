"""Download YCB object meshes from the official YCB dataset server.

Downloads the mesh files for each object into models/ycb/{object_id}/.
Only downloads the files needed for MuJoCo simulation (textured OBJ + STL).

Usage:
    python scripts/download_ycb_dataset.py                  # all 77 objects
    python scripts/download_ycb_dataset.py --objects 005_tomato_soup_can,025_mug
    python scripts/download_ycb_dataset.py --list           # print available objects
"""

from __future__ import annotations

import argparse
import logging
import urllib.request
from pathlib import Path

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_URL = "https://ycb-benchmarks.s3.amazonaws.com/data/berkeley"

# Subdirectory on the server that contains the mesh files
MESH_SUBDIR = "google_16k"

# Files to download per object (relative to {object_id}/{MESH_SUBDIR}/ on the server)
DOWNLOAD_FILES = [
    "textured.obj",
    "textured.mtl",
    "nontextured.stl",
    "nontextured.ply",
    "texture_map.png",
]

# Full YCB object list (77 objects)
ALL_OBJECTS = [
    "001_chips_can",
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "011_banana",
    "012_strawberry",
    "013_apple",
    "014_lemon",
    "015_peach",
    "016_pear",
    "017_orange",
    "018_plum",
    "019_pitcher_base",
    "021_bleach_cleanser",
    "022_windex_bottle",
    "024_bowl",
    "025_mug",
    "026_sponge",
    "028_skillet_lid",
    "029_plate",
    "030_fork",
    "031_spoon",
    "032_knife",
    "033_spatula",
    "035_power_drill",
    "036_wood_block",
    "037_scissors",
    "038_padlock",
    "040_large_marker",
    "041_small_marker",
    "042_adjustable_wrench",
    "043_phillips_head_screwdriver",
    "044_flat_screwdriver",
    "048_hammer",
    "050_medium_clamp",
    "051_large_clamp",
    "052_extra_large_clamp",
    "053_mini_soccer_ball",
    "054_softball",
    "055_baseball",
    "056_tennis_ball",
    "057_racquetball",
    "058_golf_ball",
    "059_chain",
    "061_foam_brick",
    "062_dice",
    "063_a_marbles",
    "065_a_cups",
    "065_b_cups",
    "065_c_cups",
    "065_d_cups",
    "065_e_cups",
    "065_f_cups",
    "065_g_cups",
    "065_h_cups",
    "065_i_cups",
    "065_j_cups",
    "070_a_colored_wood_blocks",
    "070_b_colored_wood_blocks",
    "071_nine_hole_peg_test",
    "072_a_toy_airplane",
    "072_b_toy_airplane",
    "072_c_toy_airplane",
    "072_d_toy_airplane",
    "072_e_toy_airplane",
    "073_a_lego_duplo",
    "073_b_lego_duplo",
    "073_c_lego_duplo",
    "073_d_lego_duplo",
    "073_e_lego_duplo",
    "073_f_lego_duplo",
    "073_g_lego_duplo",
    "077_rubiks_cube",
]


def _download_file(url: str, dest: Path) -> bool:
    """Download url to dest. Returns True on success, False if 404."""
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise


def download_object(object_id: str, models_dir: Path) -> bool:
    """Download all mesh files for one object. Returns True if any file was downloaded."""
    out_dir = models_dir / object_id
    out_dir.mkdir(parents=True, exist_ok=True)

    object_url = f"{BASE_URL}/{object_id}/{MESH_SUBDIR}"
    downloaded_any = False

    for filename in DOWNLOAD_FILES:
        dest = out_dir / filename
        if dest.exists():
            LOGGER.debug("  skip (exists): %s", filename)
            continue
        url = f"{object_url}/{filename}"
        ok = _download_file(url, dest)
        if ok:
            LOGGER.info("  downloaded: %s", filename)
            downloaded_any = True
        else:
            LOGGER.debug("  not found (skipping): %s", filename)

    return downloaded_any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download YCB object meshes.")
    parser.add_argument(
        "--objects", type=str, default="all",
        help="'all' or comma-separated IDs (e.g. 005_tomato_soup_can,025_mug)",
    )
    parser.add_argument(
        "--models-dir", type=Path,
        default=PROJECT_ROOT / "models" / "ycb",
        help="Output directory (default: models/ycb/)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print all available object IDs and exit",
    )
    args = parser.parse_args(argv)

    if args.list:
        for obj in ALL_OBJECTS:
            print(obj)
        return 0

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if args.objects.strip().lower() == "all":
        object_ids = ALL_OBJECTS
    else:
        object_ids = sorted(s.strip() for s in args.objects.split(",") if s.strip())

    LOGGER.info("Downloading %d objects to %s", len(object_ids), args.models_dir)

    ok_count = 0
    for obj_id in object_ids:
        LOGGER.info("Fetching %s ...", obj_id)
        try:
            if download_object(obj_id, args.models_dir):
                ok_count += 1
            else:
                LOGGER.warning("No files downloaded for %s (object may not exist)", obj_id)
        except Exception as exc:
            LOGGER.error("FAIL %s: %s", obj_id, exc)

    LOGGER.info("Done: %d/%d objects downloaded", ok_count, len(object_ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
