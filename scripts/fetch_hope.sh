#!/usr/bin/env bash
set -euo pipefail

# Prepare HOPE dataset directory and capture source metadata.
# By default this script does not auto-download from Google Drive because
# access patterns vary. If gdown is installed and a file/folder ID is provided,
# it will attempt an automated download.
#
# Usage:
#   scripts/fetch_hope.sh [output_dir] [drive_url_or_id]

OUT_DIR="${1:-data/objects/hope/raw}"
DRIVE_REF="${2:-https://drive.google.com/drive/folders/1Hj5K9RIdcNxBFiU8qG0-oL3Ryd9f2gOY}"
META_PATH="${OUT_DIR}/download_metadata.txt"

mkdir -p "${OUT_DIR}"

cat > "${META_PATH}" <<META
source=${DRIVE_REF}
downloaded_at_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
notes=Place HOPE archives or extracted object folders under this directory.
META

if command -v gdown >/dev/null 2>&1; then
  echo "gdown detected. Attempting HOPE download (best effort)."
  # gdown folder support may require --folder and direct URL
  set +e
  gdown --folder "${DRIVE_REF}" -O "${OUT_DIR}"
  GDOWN_STATUS=$?
  set -e
  if [[ ${GDOWN_STATUS} -ne 0 ]]; then
    echo "gdown download did not complete. You can manually download to ${OUT_DIR}."
  fi
else
  echo "gdown not found. Please manually download HOPE assets into ${OUT_DIR}."
fi

echo "HOPE dataset directory prepared. Metadata written to ${META_PATH}"
