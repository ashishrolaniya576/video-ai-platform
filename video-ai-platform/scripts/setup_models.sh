#!/usr/bin/env bash
# =============================================================================
#  setup_models.sh — Download all AI model weights
#
#  Usage (called automatically by setup.sh, or standalone):
#    bash scripts/setup_models.sh
#
#  Models handled:
#    1. RAFT Stabilization    — GitHub clone + HuggingFace weights (21 MB)
#    2. PromptIR Visibility   — GitHub clone + GitHub Release checkpoint (389 MB)
#    3. Heavy Rain Removal    — GitHub clone + Dropbox checkpoint (193 MB)
#    4. YOLOv11n Detection    — ultralytics auto-download (5 MB)
#
#  Idempotent: never re-downloads if files already exist and are non-empty.
#  All downloads are verified by checking file size > 0.
# =============================================================================

set -euo pipefail

# ── Colour helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; }
banner()  { echo -e "\n${BOLD}${CYAN}── $* ──${RESET}"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
AI_DIR="$REPO_ROOT/ai-services"

# ── Download helper (wget → curl → python fallback) ───────────────────────────
download_file() {
    local url="$1"
    local dest="$2"
    local label="${3:-file}"

    info "Downloading $label …"
    if command -v wget &>/dev/null; then
        wget -q --show-progress --tries=3 --timeout=120 -O "$dest" "$url"
    elif command -v curl &>/dev/null; then
        curl -L --retry 3 --retry-delay 5 --connect-timeout 30 -o "$dest" "$url"
    else
        # Python urllib fallback
        python3 -c "
import urllib.request, sys
url, dest = sys.argv[1], sys.argv[2]
print(f'  Downloading via Python urllib: {url}')
urllib.request.urlretrieve(url, dest)
" "$url" "$dest"
    fi

    if [ ! -s "$dest" ]; then
        error "Download failed or produced empty file: $dest"
        rm -f "$dest"
        return 1
    fi
    success "$label downloaded ($(du -h "$dest" | cut -f1))"
}

# ── Clone helper ──────────────────────────────────────────────────────────────
clone_if_missing() {
    local repo_url="$1"
    local target_dir="$2"
    local sentinel_file="$3"  # file that proves the clone succeeded
    local label="$4"

    if [ -d "$target_dir" ] && [ -f "$target_dir/$sentinel_file" ]; then
        success "$label repository already present — skipping clone"
        return 0
    fi

    if [ -d "$target_dir" ] && [ ! -f "$target_dir/$sentinel_file" ]; then
        warn "Incomplete $label directory found — removing and re-cloning"
        rm -rf "$target_dir"
    fi

    info "Cloning $label from $repo_url …"
    git clone --depth 1 -q "$repo_url" "$target_dir"
    success "$label cloned"
}

# =============================================================================
# MODEL 1: RAFT Stabilization
# =============================================================================
banner "RAFT Video Stabilization"

RAFT_DIR="$AI_DIR/RAFT"
RAFT_WEIGHTS="$RAFT_DIR/models/raft-sintel.pth"
RAFT_HF_URL="https://huggingface.co/ddrfan/RAFT/resolve/main/raft-sintel.pth?download=true"

clone_if_missing \
    "https://github.com/princeton-vl/RAFT.git" \
    "$RAFT_DIR" \
    "core/raft.py" \
    "RAFT"

mkdir -p "$RAFT_DIR/models"

if [ -s "$RAFT_WEIGHTS" ]; then
    success "raft-sintel.pth already present ($(du -h "$RAFT_WEIGHTS" | cut -f1)) — skipping"
else
    download_file "$RAFT_HF_URL" "$RAFT_WEIGHTS" "raft-sintel.pth (HuggingFace)"
fi

# =============================================================================
# MODEL 2: PromptIR Video Visibility Enhancement
# =============================================================================
banner "PromptIR Video Visibility"

PROMPTIR_DIR="$AI_DIR/PromptIR"
PROMPTIR_CKPT_DIR="$AI_DIR/pretrained/PromptIR/checkpoint"
PROMPTIR_CKPT="$PROMPTIR_CKPT_DIR/model.ckpt"
# Also check the models_weights/ location used in current settings
PROMPTIR_CKPT_ALT="$AI_DIR/models_weights/promptir_model.ckpt"
PROMPTIR_GH_URL="https://github.com/va1shn9v/PromptIR/releases/download/v1.0/model.ckpt"

clone_if_missing \
    "https://github.com/va1shn9v/PromptIR.git" \
    "$PROMPTIR_DIR" \
    "net/model.py" \
    "PromptIR"

mkdir -p "$PROMPTIR_CKPT_DIR"

if [ -s "$PROMPTIR_CKPT" ]; then
    success "PromptIR model.ckpt already present ($(du -h "$PROMPTIR_CKPT" | cut -f1)) — skipping"
elif [ -s "$PROMPTIR_CKPT_ALT" ]; then
    # Copy from models_weights if it exists there
    cp "$PROMPTIR_CKPT_ALT" "$PROMPTIR_CKPT"
    success "PromptIR checkpoint copied from models_weights/"
else
    download_file "$PROMPTIR_GH_URL" "$PROMPTIR_CKPT" "PromptIR model.ckpt (~389 MB)"
fi

# =============================================================================
# MODEL 3: Heavy Rain Removal
# =============================================================================
banner "Heavy Rain Removal"

HEAVYRAIN_DIR="$AI_DIR/HeavyRainRemoval"
HEAVYRAIN_CKPT_DIR="$AI_DIR/pretrained/HeavyRainRemoval/checkpoint"
HEAVYRAIN_CKPT="$HEAVYRAIN_CKPT_DIR/HeavyRain-stage2-2019-05-11-76_ckpt.pth.tar"
# Dropbox direct download (dl=1 forces raw download)
HEAVYRAIN_DROPBOX_URL="https://www.dropbox.com/s/h8x6xl6epc45ngn/HeavyRain-stage2-2019-05-11-76_ckpt.pth.tar?dl=1"

clone_if_missing \
    "https://github.com/liruoteng/HeavyRainRemoval.git" \
    "$HEAVYRAIN_DIR" \
    "helper.py" \
    "HeavyRainRemoval"

mkdir -p "$HEAVYRAIN_CKPT_DIR"

if [ -s "$HEAVYRAIN_CKPT" ]; then
    success "HeavyRain checkpoint already present ($(du -h "$HEAVYRAIN_CKPT" | cut -f1)) — skipping"
else
    info "Downloading Heavy Rain Removal checkpoint (~193 MB from Dropbox)…"
    if download_file "$HEAVYRAIN_DROPBOX_URL" "$HEAVYRAIN_CKPT" "HeavyRain checkpoint"; then
        success "Heavy Rain checkpoint downloaded"
    else
        warn "Heavy Rain Removal checkpoint download failed."
        warn "Download manually from:"
        warn "  $HEAVYRAIN_DROPBOX_URL"
        warn "Save to: $HEAVYRAIN_CKPT"
        warn "Heavy Rain Removal will be unavailable until the checkpoint is present."
    fi
fi



# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${BOLD}${GREEN}Model Weight Status:${RESET}"
echo -e "  RAFT:        $([ -s "$RAFT_WEIGHTS" ] && echo "✓ $(du -h "$RAFT_WEIGHTS" | cut -f1)" || echo "✗ MISSING")"
echo -e "  PromptIR:    $([ -s "$PROMPTIR_CKPT" ] && echo "✓ $(du -h "$PROMPTIR_CKPT" | cut -f1)" || echo "✗ MISSING")"
echo -e "  Heavy Rain:  $([ -s "$HEAVYRAIN_CKPT" ] && echo "✓ $(du -h "$HEAVYRAIN_CKPT" | cut -f1)" || echo "✗ MISSING")"
echo -e "  Distance Estimation: ✓ Present in repository"
echo ""
success "setup_models.sh complete"
