#!/usr/bin/env bash
# setup_raft.sh — Clone the RAFT repository and download pretrained weights.
#
# Run from any directory; the script locates ai-services/ automatically.
#
#   bash ai-services/scripts/setup_raft.sh
#   # or from ai-services/:
#   bash scripts/setup_raft.sh
#
# What this does:
#   1. Clone https://github.com/princeton-vl/RAFT into ai-services/RAFT/
#   2. Download raft-sintel.pth from HuggingFace (primary) or Dropbox (fallback)
#   3. Verify folder structure
#   4. Print success message
#
# Idempotent — safe to run multiple times; will never duplicate downloads.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICES_DIR="$(dirname "$SCRIPT_DIR")"
RAFT_DIR="${AI_SERVICES_DIR}/RAFT"
MODELS_DIR="${RAFT_DIR}/models"
WEIGHTS_FILE="${MODELS_DIR}/raft-sintel.pth"

# Primary: HuggingFace (community mirror — reliable, no auth required)
HF_URL="https://huggingface.co/ddrfan/RAFT/resolve/main/raft-sintel.pth?download=true"
# Fallback: official Dropbox (original source, may be rate-limited)
DROPBOX_URL="https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip"

echo "========================================================"
echo " RAFT Setup Script"
echo " ai-services: ${AI_SERVICES_DIR}"
echo " RAFT target: ${RAFT_DIR}"
echo "========================================================"

# ── Step 1: Clone RAFT repository ─────────────────────────────────────────────
if [ -d "${RAFT_DIR}" ] && [ -d "${RAFT_DIR}/core" ]; then
    echo "[skip] RAFT repository already cloned — skipping clone."
else
    echo "[clone] Cloning RAFT from GitHub…"
    # Remove partial clone if it exists but core/ is missing
    if [ -d "${RAFT_DIR}" ] && [ ! -d "${RAFT_DIR}/core" ]; then
        echo "[warn]  Incomplete RAFT directory found — removing and re-cloning."
        rm -rf "${RAFT_DIR}"
    fi
    git clone -q https://github.com/princeton-vl/RAFT.git "${RAFT_DIR}"
    echo "[clone] Done."
fi

# Verify core directory
if [ ! -d "${RAFT_DIR}/core" ]; then
    echo "[error] RAFT/core not found after clone. Something went wrong."
    exit 1
fi
echo "[ok]    RAFT/core directory present."

# ── Step 2: Create models directory ───────────────────────────────────────────
mkdir -p "${MODELS_DIR}"

# ── Step 3: Download raft-sintel.pth ──────────────────────────────────────────
if [ -f "${WEIGHTS_FILE}" ]; then
    echo "[skip]  raft-sintel.pth already present — skipping download."
else
    echo "[download] Downloading raft-sintel.pth from HuggingFace…"
    if wget -q --show-progress -O "${WEIGHTS_FILE}" "${HF_URL}"; then
        echo "[download] Done (HuggingFace)."
    else
        echo "[warn]  HuggingFace download failed. Trying official Dropbox source…"
        # Dropbox distributes a zip containing all models
        TMP_ZIP="${AI_SERVICES_DIR}/_raft_models_tmp.zip"
        if wget -q --show-progress -O "${TMP_ZIP}" "${DROPBOX_URL}"; then
            echo "[unzip]  Extracting models.zip…"
            # Unzip into a temp dir, then move raft-sintel.pth
            TMP_EXTRACT="${AI_SERVICES_DIR}/_raft_extract_tmp"
            mkdir -p "${TMP_EXTRACT}"
            unzip -q "${TMP_ZIP}" -d "${TMP_EXTRACT}"
            # The zip puts files under models/ at its root
            find "${TMP_EXTRACT}" -name "raft-sintel.pth" -exec cp {} "${WEIGHTS_FILE}" \;
            rm -rf "${TMP_ZIP}" "${TMP_EXTRACT}"
            echo "[download] Done (Dropbox fallback)."
        else
            rm -f "${TMP_ZIP}"
            echo "[error]  Both download sources failed."
            echo "         Please download raft-sintel.pth manually and place it at:"
            echo "           ${WEIGHTS_FILE}"
            exit 1
        fi
    fi
fi

# ── Step 4: Verify ────────────────────────────────────────────────────────────
echo ""
echo "Verification:"
echo "  RAFT/core:            $([ -d "${RAFT_DIR}/core" ] && echo OK || echo MISSING)"
echo "  RAFT/core/raft.py:    $([ -f "${RAFT_DIR}/core/raft.py" ] && echo OK || echo MISSING)"
echo "  raft-sintel.pth:      $([ -f "${WEIGHTS_FILE}" ] && echo "OK ($(du -h "${WEIGHTS_FILE}" | cut -f1))" || echo MISSING)"
echo ""

echo "========================================================"
echo " RAFT setup complete."
echo ""
echo " Resolved paths (absolute):"
echo "   RAFT_REPO_PATH  → ${RAFT_DIR}"
echo "   RAFT_MODEL_PATH → ${WEIGHTS_FILE}"
echo ""
echo " Your .env is already configured with relative paths:"
echo "   RAFT_REPO_PATH=RAFT"
echo "   RAFT_MODEL_PATH=RAFT/models/raft-sintel.pth"
echo " (settings.py resolves these to absolute paths automatically)"
echo "========================================================"
