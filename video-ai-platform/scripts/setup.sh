#!/usr/bin/env bash
# =============================================================================
#  setup.sh — Full installation script for VideoAI Platform
#
#  Usage:
#    cd video-ai-platform
#    bash scripts/setup.sh
#
#  What this does:
#    1. Validates system requirements (Python 3.10+, Node 18+, npm, git)
#    2. Creates the Python virtual environment inside ai-services/venv/
#    3. Upgrades pip and installs all Python dependencies
#    4. Installs Node.js dependencies for both backend and frontend
#    5. Creates required runtime directories (output/, temp/, logs/)
#    6. Generates .env files from .env.example if they don't exist
#    7. Calls setup_models.sh to download all AI model weights
#    8. Runs verify_installation.py for a full health report
#
#  Idempotent: safe to run multiple times — will never break an existing install.
#  Platform:   Linux / macOS (Lightning AI compatible)
# =============================================================================

set -euo pipefail

# ── Colour helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; }
banner()  { echo -e "\n${BOLD}${CYAN}$*${RESET}\n"; }

# ── Resolve repo root (works regardless of CWD) ───────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
AI_DIR="$REPO_ROOT/ai-services"
BACKEND_DIR="$REPO_ROOT/backend"
FRONTEND_DIR="$REPO_ROOT/frontend"
VENV_DIR="$AI_DIR/venv"
LOGS_DIR="$REPO_ROOT/logs"

banner "╔══════════════════════════════════════════════════════╗"
banner "║         VideoAI Platform — Setup Script              ║"
banner "║         $(date '+%Y-%m-%d %H:%M:%S')                         ║"
banner "╚══════════════════════════════════════════════════════╝"

# =============================================================================
# PHASE 1 — System requirement checks
# =============================================================================
banner "Phase 1: Checking System Requirements"

# ── Python ────────────────────────────────────────────────────────────────────
PYTHON_BIN=""
for candidate in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || true)
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "${major:-0}" -ge 3 ] && [ "${minor:-0}" -ge 10 ]; then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    error "Python 3.10 or higher is required but was not found."
    error "Install with: sudo apt install python3.12 python3.12-venv python3.12-dev"
    exit 1
fi
PYTHON_VER=$("$PYTHON_BIN" -c "import sys; print(sys.version.split()[0])")
success "Python $PYTHON_VER found at $(command -v $PYTHON_BIN)"

# ── ffmpeg ────────────────────────────────────────────────────────────────────
if ! command -v ffmpeg &>/dev/null; then
    warn "ffmpeg is not installed (required for video processing)."
    if command -v apt-get &>/dev/null; then
        info "Attempting to install ffmpeg via apt-get..."
        sudo apt-get update && sudo apt-get install -y ffmpeg || warn "Failed to install ffmpeg automatically. Please install it manually."
    elif command -v brew &>/dev/null; then
        info "Attempting to install ffmpeg via Homebrew..."
        brew install ffmpeg || warn "Failed to install ffmpeg automatically. Please install it manually."
    else
        warn "Please install ffmpeg manually (e.g. 'sudo apt install ffmpeg' or 'brew install ffmpeg')."
    fi
else
    success "ffmpeg found"
fi

# ── Node.js ───────────────────────────────────────────────────────────────────
if ! command -v node &>/dev/null; then
    error "Node.js is required but was not found."
    error "Install with: curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash - && sudo apt install -y nodejs"
    exit 1
fi
NODE_VER=$(node --version)
NODE_MAJOR=$(echo "$NODE_VER" | tr -d 'v' | cut -d. -f1)
if [ "$NODE_MAJOR" -lt 18 ]; then
    warn "Node.js $NODE_VER detected. Version 18+ is recommended."
else
    success "Node.js $NODE_VER found"
fi

# ── npm ───────────────────────────────────────────────────────────────────────
if ! command -v npm &>/dev/null; then
    error "npm is required but was not found."
    exit 1
fi
success "npm $(npm --version) found"

# ── git ───────────────────────────────────────────────────────────────────────
if ! command -v git &>/dev/null; then
    error "git is required but was not found."
    error "Install with: sudo apt install git"
    exit 1
fi
success "git $(git --version | awk '{print $3}') found"

# ── wget / curl ───────────────────────────────────────────────────────────────
if command -v wget &>/dev/null; then
    success "wget found"
elif command -v curl &>/dev/null; then
    warn "wget not found, but curl is available. Downloads will use curl."
else
    warn "Neither wget nor curl found. Model downloads may fail."
fi

# =============================================================================
# PHASE 2 — Python virtual environment
# =============================================================================
banner "Phase 2: Python Virtual Environment"

if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    success "Virtual environment already exists at $VENV_DIR"
else
    info "Creating virtual environment at $VENV_DIR …"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    success "Virtual environment created"
fi

# Activate
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
success "Virtual environment activated ($(python --version))"

# =============================================================================
# PHASE 3 — Python dependencies
# =============================================================================
banner "Phase 3: Python Dependencies"

info "Upgrading pip …"
pip install --upgrade pip --quiet
success "pip $(pip --version | awk '{print $2}') ready"

info "Installing Python requirements from $AI_DIR/requirements.txt …"
pip install -r "$AI_DIR/requirements.txt" --quiet
success "All Python packages installed"

# =============================================================================
# PHASE 4 — Node.js dependencies
# =============================================================================
banner "Phase 4: Node.js Dependencies"

info "Installing backend Node modules …"
cd "$BACKEND_DIR"
npm install --silent
success "Backend node_modules ready"

info "Installing frontend Node modules …"
cd "$FRONTEND_DIR"
npm install --silent
success "Frontend node_modules ready"

cd "$REPO_ROOT"

# =============================================================================
# PHASE 5 — Create runtime directories
# =============================================================================
banner "Phase 5: Runtime Directories"

for dir in \
    "$AI_DIR/output" \
    "$AI_DIR/temp" \
    "$AI_DIR/models_weights" \
    "$AI_DIR/pretrained/HeavyRainRemoval/checkpoint" \
    "$AI_DIR/pretrained/PromptIR/checkpoint" \
    "$LOGS_DIR"; do
    mkdir -p "$dir"
    success "Directory ready: ${dir#$REPO_ROOT/}"
done

# =============================================================================
# PHASE 6 — Environment files
# =============================================================================
banner "Phase 6: Environment Configuration"

# AI service .env
AI_ENV="$AI_DIR/.env"
AI_ENV_EXAMPLE="$AI_DIR/.env.example"
if [ -f "$AI_ENV" ]; then
    success "AI service .env already exists — skipping"
else
    if [ -f "$AI_ENV_EXAMPLE" ]; then
        cp "$AI_ENV_EXAMPLE" "$AI_ENV"
        success "Created ai-services/.env from .env.example"
    else
        warn "ai-services/.env.example not found — creating minimal .env"
        cat > "$AI_ENV" << 'ENVEOF'
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info
DEVICE=auto
OUTPUT_DIR=output
TEMP_DIR=temp
MODELS_DIR=models_weights
FRAME_BUFFER_SIZE=32
RAFT_REPO_PATH=RAFT
RAFT_MODEL_PATH=RAFT/models/raft-sintel.pth
RAFT_SMOOTHING_RADIUS=30
RAFT_CROP_RATIO=0.07
RAFT_FLOW_WIDTH=960
RAFT_FLOW_HEIGHT=540
RAFT_ITERS=20
HEAVY_RAIN_REPO_PATH=HeavyRainRemoval
HEAVY_RAIN_CHECKPOINT=pretrained/HeavyRainRemoval/checkpoint/HeavyRain-stage2-2019-05-11-76_ckpt.pth.tar
PROMPTIR_REPO_PATH=PromptIR
PROMPTIR_CHECKPOINT=pretrained/PromptIR/checkpoint/model.ckpt
PROMPTIR_TILE_SIZE=512
PROMPTIR_TILE_OVERLAP=32
PROMPTIR_CONTRAST_ALPHA=1.3
PROMPTIR_CONTRAST_BETA=10.0
PROMPTIR_CLAHE_CLIP=1.5
DISTANCE_WEIGHTS_PATH=../distanceEstimation_d2/best.pth
DISTANCE_YAML_PATH=../distanceEstimation_d2/data.yaml
DISTANCE_CONFIDENCE_THRESHOLD=0.3
ENVEOF
    fi
fi

# Backend .env
BACKEND_ENV="$BACKEND_DIR/.env"
BACKEND_ENV_EXAMPLE="$BACKEND_DIR/.env.example"
if [ -f "$BACKEND_ENV" ]; then
    success "Backend .env already exists — skipping"
else
    if [ -f "$BACKEND_ENV_EXAMPLE" ]; then
        cp "$BACKEND_ENV_EXAMPLE" "$BACKEND_ENV"
        success "Created backend/.env from .env.example"
    else
        cat > "$BACKEND_ENV" << 'ENVEOF'
PORT=5000
AI_SERVICE_URL=http://localhost:8000
NODE_ENV=development
ENVEOF
        success "Created backend/.env with default values"
    fi
fi

# =============================================================================
# PHASE 7 — AI model weights
# =============================================================================
banner "Phase 7: AI Model Weights"

bash "$SCRIPT_DIR/setup_models.sh"

# =============================================================================
# PHASE 8 — Verify installation
# =============================================================================
banner "Phase 8: Verification"

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
python "$SCRIPT_DIR/verify_installation.py" --repo-root "$REPO_ROOT"

# =============================================================================
# Summary
# =============================================================================
banner "╔══════════════════════════════════════════════════════╗"
echo -e "  ${GREEN}${BOLD}Setup Complete!${RESET}"
echo ""
echo -e "  ${CYAN}To start all services:${RESET}"
echo -e "    bash scripts/start_all.sh"
echo ""
echo -e "  ${CYAN}To verify health:${RESET}"
echo -e "    bash scripts/health_check.sh"
echo ""
echo -e "  ${CYAN}URLs (after starting):${RESET}"
echo -e "    Frontend:   http://localhost:5173"
echo -e "    Backend:    http://localhost:5000"
echo -e "    AI Service: http://localhost:8000"
echo -e "    AI Docs:    http://localhost:8000/docs"
banner "╚══════════════════════════════════════════════════════╝"
