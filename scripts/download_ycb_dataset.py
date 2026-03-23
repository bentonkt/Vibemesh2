"""Download YCB object meshes from the official YCB dataset server.

Downloads the google_16k mesh tarball for each object and extracts the
mesh files into models/ycb/{object_id}/ for use with process_ycb.py.

Usage:
    python scripts/download_ycb_dataset.py                  # all objects
    python scripts/download_ycb_dataset.py --objects 005_tomato_soup_can,025_mug
    python scripts/download_ycb_dataset.py --list           # print available objects
"""

from __future__ import annotations

import argparse
import logging
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_URL = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data"

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


def _already_downloaded(out_dir: Path) -> bool:
    """Return True if the object directory already has a usable mesh."""
    return any(out_dir.glob("*.obj")) or any(out_dir.glob("*.stl"))


def download_object(object_id: str, models_dir: Path) -> None:
    """Download and extract the google_16k tarball for one object."""
    out_dir = models_dir / object_id

    if _already_downloaded(out_dir):
        LOGGER.info("  already downloaded, skipping")
        return

    url = f"{BASE_URL}/google/{object_id}_google_16k.tgz"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        tgz_path = tmp_path / f"{object_id}.tgz"

        LOGGER.info("  downloading %s", url)
        try:
            urllib.request.urlretrieve(url, tgz_path)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {url}") from e

        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(tmp_path)

        # Tarball extracts to {object_id}/google_16k/
        extracted = tmp_path / object_id / "google_16k"
        if not extracted.exists():
            # Some tarballs may extract differently — find the first subdir with meshes
            candidates = [p for p in tmp_path.rglob("*.obj")]
            if not candidates:
                raise RuntimeError(f"No .obj files found after extracting {url}")
            extracted = candidates[0].parent

        out_dir.mkdir(parents=True, exist_ok=True)
        for f in extracted.iterdir():
            shutil.copy2(f, out_dir / f.name)

    LOGGER.info("  extracted to %s", out_dir)


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

    ok, failed = 0, []
    for obj_id in object_ids:
        LOGGER.info("Fetching %s ...", obj_id)
        try:
            download_object(obj_id, args.models_dir)
            ok += 1
        except Exception as exc:
            LOGGER.error("FAIL %s: %s", obj_id, exc)
            failed.append(obj_id)

    LOGGER.info("Done: %d/%d objects downloaded", ok, len(object_ids))
    if failed:
        LOGGER.error("Failed: %s", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
