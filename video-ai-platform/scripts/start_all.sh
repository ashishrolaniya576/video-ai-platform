#!/usr/bin/env bash
# =============================================================================
#  start_all.sh — Starts the VideoAI Platform services
#
#  Usage:
#    bash scripts/start_all.sh
#
#  What this does:
#    1. Checks if ports are free.
#    2. Starts the FastAPI AI Service (port 8000)
#    3. Waits for AI Service health check to pass.
#    4. Starts the Node.js Express Backend (port 5000)
#    5. Waits for Backend health check to pass.
#    6. Starts the React Vite Frontend (port 5173)
#    7. Saves PIDs for stop_all.sh and logs to logs/.
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
LOGS_DIR="$REPO_ROOT/logs"
PID_DIR="$LOGS_DIR/pids"

mkdir -p "$LOGS_DIR" "$PID_DIR"

banner "Starting VideoAI Platform"

# ── 1. FastAPI AI Service ─────────────────────────────────────────────────────
info "Starting AI Service (FastAPI) on port 8000..."
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null; then
    warn "Port 8000 is already in use. AI Service might already be running."
else
    cd "$REPO_ROOT/ai-services"
    # shellcheck source=/dev/null
    source venv/bin/activate
    nohup uvicorn app.main:create_app --host 0.0.0.0 --port 8000 --factory > "$LOGS_DIR/ai.log" 2>&1 &
    AI_PID=$!
    echo $AI_PID > "$PID_DIR/ai.pid"
    
    info "Waiting for AI Service to become healthy..."
    # Wait up to 60 seconds
    timeout 60 bash -c 'until curl -s http://localhost:8000/health > /dev/null; do sleep 2; done' || {
        error "AI Service failed to start in time. Check logs/ai.log"
        exit 1
    }
    success "AI Service is healthy"
fi

# ── 2. Node.js Backend ────────────────────────────────────────────────────────
info "Starting Backend Service on port 5000..."
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null; then
    warn "Port 5000 is already in use. Backend might already be running."
else
    cd "$REPO_ROOT/backend"
    nohup npm start > "$LOGS_DIR/backend.log" 2>&1 &
    BACKEND_PID=$!
    echo $BACKEND_PID > "$PID_DIR/backend.pid"

    info "Waiting for Backend to become healthy..."
    timeout 30 bash -c 'until curl -s http://localhost:5000/api/health > /dev/null; do sleep 2; done' || {
        error "Backend failed to start in time. Check logs/backend.log"
        exit 1
    }
    success "Backend is healthy"
fi

# ── 3. React Frontend ─────────────────────────────────────────────────────────
info "Starting Frontend Service on port 5173..."
if lsof -Pi :5173 -sTCP:LISTEN -t >/dev/null; then
    warn "Port 5173 is already in use. Frontend might already be running."
else
    cd "$REPO_ROOT/frontend"
    nohup npm run dev -- --port 5173 > "$LOGS_DIR/frontend.log" 2>&1 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > "$PID_DIR/frontend.pid"

    info "Waiting for Frontend..."
    sleep 3 # Vite usually starts instantly
    success "Frontend is running"
fi

banner "All Services Started!"
echo -e "  ${CYAN}Frontend:   ${RESET}http://localhost:5173"
echo -e "  ${CYAN}Backend:    ${RESET}http://localhost:5000"
echo -e "  ${CYAN}AI Service: ${RESET}http://localhost:8000"
echo -e "  ${CYAN}Logs:       ${RESET}$LOGS_DIR/"
echo -e "  To stop services, run: ${BOLD}bash scripts/stop_all.sh${RESET}\n"
