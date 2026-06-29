#!/usr/bin/env bash
# =============================================================================
#  verify_startup.sh — Live Startup Verifier for VideoAI Platform
#
#  Usage:
#    bash scripts/verify_startup.sh
#
#  What this does:
#    1. Checks if all services are running (Frontend, Backend, AI).
#    2. Tests the Backend API health endpoint.
#    3. Tests the AI API health endpoint and verifies all models loaded.
#    4. Submits a test job to the backend API to verify end-to-end functionality.
#    5. Generates STARTUP_REPORT.md with the results.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPORT_FILE="$REPO_ROOT/STARTUP_REPORT.md"

# ── Colour helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; }
banner()  { echo -e "\n${BOLD}${CYAN}── $* ──${RESET}"; }

# Arrays for reporting
declare -a RESULTS_PASS
declare -a RESULTS_FAIL

log_pass() {
    success "$1"
    RESULTS_PASS+=("$1")
}

log_fail() {
    error "$1"
    RESULTS_FAIL+=("$1")
}

banner "Automatic Startup Verification"

# 1. Frontend Build & Serve Check
info "Verifying Frontend..."
if curl -s http://localhost:5173 | grep -qi "VideoAI Platform"; then
    log_pass "Frontend is serving correctly on port 5173."
else
    log_fail "Frontend is not reachable or incorrect response on port 5173."
fi

# 2. Backend Health Check
info "Verifying Backend API..."
if curl -s http://localhost:5000/api/health | grep -q "running"; then
    log_pass "Backend API is healthy on port 5000."
else
    log_fail "Backend API failed health check on port 5000."
fi

# 3. AI Service Health & Model Loading Check
info "Verifying AI Service & Model Loading..."
AI_HEALTH=$(curl -s http://localhost:8000/health || echo "FAILED")
if [[ "$AI_HEALTH" == "FAILED" ]]; then
    log_fail "AI Service is unreachable on port 8000."
else
    if echo "$AI_HEALTH" | grep -q '"status":"running"'; then
        log_pass "AI Service API is healthy."
    else
        log_fail "AI Service API reported unhealthy status."
    fi

    # Check each model in the health JSON
    for model in "stabilization" "heavy_rain_removal" "video_visibility" "distance_estimation"; do
        if echo "$AI_HEALTH" | grep -q "\"$model\":true"; then
            log_pass "AI Model loaded: $model"
        else
            log_fail "AI Model failed to load: $model"
        fi
    done
fi

# 4. End-to-End API Test
info "Verifying End-to-End Job Submission..."
# Submit a tiny test request (we won't wait for completion here to avoid hanging the test, just verify acceptance)
JOB_RESP=$(curl -s -X POST -H "Content-Type: application/json" \
    -d '{"videoUrl": "https://www.w3schools.com/html/mov_bbb.mp4", "distanceEstimation": true}' \
    http://localhost:5000/api/process || echo "FAILED")

if echo "$JOB_RESP" | grep -q "jobId"; then
    log_pass "Backend API accepted processing job."
else
    log_fail "Backend API failed to accept processing job. Response: $JOB_RESP"
fi

# =============================================================================
# Generate STARTUP_REPORT.md
# =============================================================================
info "Generating STARTUP_REPORT.md..."

cat > "$REPORT_FILE" << EOF
# Startup Verification Report — VideoAI Platform

**Generated:** $(date '+%Y-%m-%d %H:%M:%S')

## Overall Status
EOF

if [ ${#RESULTS_FAIL[@]} -eq 0 ]; then
    echo "**Result: ✅ PASS (All services and models verified)**" >> "$REPORT_FILE"
else
    echo "**Result: ❌ FAIL (${#RESULTS_FAIL[@]} checks failed)**" >> "$REPORT_FILE"
fi

echo -e "\n## Successful Checks\n" >> "$REPORT_FILE"
for p in "${RESULTS_PASS[@]}"; do
    echo "- ✅ $p" >> "$REPORT_FILE"
done

if [ ${#RESULTS_FAIL[@]} -gt 0 ]; then
    echo -e "\n## Failed Checks\n" >> "$REPORT_FILE"
    for f in "${RESULTS_FAIL[@]}"; do
        echo "- ❌ $f" >> "$REPORT_FILE"
    done
fi

success "Verification complete. Report written to $REPORT_FILE."
