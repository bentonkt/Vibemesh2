#!/usr/bin/env bash
set -euo pipefail

# Download YCB meshes using the official ycb-tools downloader script.
# Usage:
#   scripts/fetch_ycb.sh [output_dir]
# Example:
#   scripts/fetch_ycb.sh data/objects/ycb/raw

OUT_DIR="${1:-data/objects/ycb/raw}"
TOOLS_DIR="${OUT_DIR}/_ycb_tools"
SCRIPT_PATH="${TOOLS_DIR}/download_ycb_dataset.py"
META_PATH="${OUT_DIR}/download_metadata.txt"

mkdir -p "${OUT_DIR}" "${TOOLS_DIR}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but not found." >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required but not found." >&2
  exit 1
fi

YCB_SCRIPT_URL="https://raw.githubusercontent.com/sea-bass/ycb-tools/main/download_ycb_dataset.py"

echo "Fetching ycb-tools downloader from ${YCB_SCRIPT_URL}"
curl -fsSL "${YCB_SCRIPT_URL}" -o "${SCRIPT_PATH}"

# Optional: try to record current HEAD commit for reproducibility
YCB_COMMIT="unknown"
if command -v git >/dev/null 2>&1; then
  YCB_COMMIT="$(git ls-remote https://github.com/sea-bass/ycb-tools.git HEAD | awk '{print $1}' || true)"
  if [[ -z "${YCB_COMMIT}" ]]; then
    YCB_COMMIT="unknown"
  fi
fi

# Run downloader inside output directory so dataset lands there.
pushd "${OUT_DIR}" >/dev/null
python3 "${SCRIPT_PATH}"
popd >/dev/null

cat > "${META_PATH}" <<META
download_source=${YCB_SCRIPT_URL}
downloaded_at_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
head_commit=${YCB_COMMIT}
out_dir=${OUT_DIR}
META

echo "YCB download finished. Metadata written to ${META_PATH}"
