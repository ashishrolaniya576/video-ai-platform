#!/usr/bin/env bash
# =============================================================================
#  stop_all.sh — Stops the VideoAI Platform services
#
#  Usage:
#    bash scripts/stop_all.sh
# =============================================================================

set -euo pipefail

# ── Colour helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_DIR="$REPO_ROOT/logs/pids"

banner()  { echo -e "\n${BOLD}${CYAN}── $* ──${RESET}"; }
banner "Stopping VideoAI Platform"

stop_service() {
    local name=$1
    local pid_file="$PID_DIR/$2.pid"
    local port=$3

    info "Stopping $name..."
    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID" || true
            success "$name (PID $PID) stopped."
        else
            info "$name PID file found, but process not running."
        fi
        rm -f "$pid_file"
    else
        info "No PID file for $name."
    fi

    # Fallback checking port
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null; then
        local port_pid
        port_pid=$(lsof -Pi :$port -sTCP:LISTEN -t)
        kill -9 "$port_pid" || true
        success "Forcefully stopped process on port $port."
    fi
}

stop_service "Frontend" "frontend" 5173
stop_service "Backend" "backend" 5000
stop_service "AI Service" "ai" 8000

# Additional cleanup for lingering node or python processes spawned via npm
pkill -f "vite" || true
pkill -f "uvicorn app.main:create_app" || true

success "All services stopped successfully."
