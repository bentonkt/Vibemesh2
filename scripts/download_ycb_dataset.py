# Copyright 2015 Yale University - Grablab
# MIT License — see https://github.com/sea-bass/ycb-tools
# Modified by Sebastian Castro (2020) for Python 3.
# Adapted for Vibemesh2: cross-platform tarfile extraction, CLI args.

"""Download YCB object meshes from the official YCB dataset server.

Downloads berkeley_processed (textured meshes) and google_16k meshes for
each object into models/ycb/{object_id}/.

Usage:
    python scripts/download_ycb_dataset.py           # all objects
    python scripts/download_ycb_dataset.py --objects 005_tomato_soup_can,025_mug
    python scripts/download_ycb_dataset.py --types google_16k
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tarfile
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_URL = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/"
OBJECTS_URL = "https://ycb-benchmarks.s3.amazonaws.com/data/objects.json"

ALL_TYPES = ["berkeley_processed", "google_16k"]


def fetch_objects() -> list[str]:
    """Fetch the canonical object list from the YCB S3 bucket."""
    response = urlopen(OBJECTS_URL)
    return json.loads(response.read())["objects"]


def tgz_url(object_id: str, file_type: str) -> str:
    if file_type == "berkeley_processed":
        return BASE_URL + f"berkeley/{object_id}/{object_id}_berkeley_meshes.tgz"
    else:
        return BASE_URL + f"google/{object_id}_{file_type}.tgz"


def check_url(url: str) -> bool:
    try:
        req = Request(url)
        req.get_method = lambda: "HEAD"
        urlopen(req)
        return True
    except URLError:
        return False


def download_and_extract(url: str, dest_dir: Path) -> None:
    """Download a .tgz and extract it into dest_dir."""
    with tempfile.NamedTemporaryFile(suffix=".tgz", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        LOGGER.info("  downloading %s", url)
        response = urlopen(url)
        size = int(response.getheader("Content-Length", 0))
        downloaded = 0
        with open(tmp_path, "wb") as f:
            while True:
                chunk = response.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if size:
                    pct = downloaded * 100 // size
                    print(f"\r  {downloaded/1e6:.1f} MB / {size/1e6:.1f} MB ({pct}%)", end="", flush=True)
        if size:
            print()

        with tarfile.open(tmp_path, "r:gz") as tar:
            tar.extractall(dest_dir)
    finally:
        tmp_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download YCB object meshes.")
    parser.add_argument("--objects", type=str, default="all",
                        help="'all' or comma-separated IDs")
    parser.add_argument("--types", type=str, default=",".join(ALL_TYPES),
                        help=f"comma-separated types (default: {','.join(ALL_TYPES)})")
    parser.add_argument("--models-dir", type=Path,
                        default=PROJECT_ROOT / "models" / "ycb")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    LOGGER.info("Fetching object list from YCB S3...")
    all_objects = fetch_objects()

    if args.objects.strip().lower() == "all":
        object_ids = all_objects
    else:
        object_ids = [s.strip() for s in args.objects.split(",") if s.strip()]

    file_types = [s.strip() for s in args.types.split(",") if s.strip()]

    args.models_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading %d objects (%s) to %s",
                len(object_ids), ", ".join(file_types), args.models_dir)

    ok, skipped, failed = 0, 0, 0
    for obj_id in object_ids:
        for file_type in file_types:
            url = tgz_url(obj_id, file_type)
            if not check_url(url):
                LOGGER.debug("skip (not available): %s / %s", obj_id, file_type)
                skipped += 1
                continue
            LOGGER.info("%s / %s", obj_id, file_type)
            try:
                download_and_extract(url, args.models_dir)
                ok += 1
            except Exception as exc:
                LOGGER.error("FAIL %s / %s: %s", obj_id, file_type, exc)
                failed += 1

    LOGGER.info("Done: %d downloaded, %d skipped (not available), %d failed",
                ok, skipped, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
