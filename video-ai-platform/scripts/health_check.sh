#!/usr/bin/env bash
# =============================================================================
#  health_check.sh — Verifies the health of the VideoAI Platform
#
#  Usage:
#    bash scripts/health_check.sh
# =============================================================================

set -euo pipefail

# ── Colour helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
banner()  { echo -e "\n${BOLD}${CYAN}── $* ──${RESET}"; }

banner "VideoAI Platform Health Check"

check_endpoint() {
    local name=$1
    local url=$2
    local expected_status=${3:-200}
    
    printf "  %-20s " "$name"
    
    if ! command -v curl &> /dev/null; then
        echo -e "${YELLOW}[WARN] curl not installed${RESET}"
        return
    fi

    # Suppress output, write out http status code
    local HTTP_CODE
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$url" || echo "failed")

    if [ "$HTTP_CODE" == "$expected_status" ]; then
        echo -e "${GREEN}[PASS] HTTP $HTTP_CODE${RESET}  ($url)"
    elif [ "$HTTP_CODE" == "failed" ]; then
        echo -e "${RED}[FAIL] Unreachable${RESET}    ($url)"
    else
        echo -e "${YELLOW}[WARN] HTTP $HTTP_CODE${RESET}  ($url)"
    fi
}

check_endpoint "Frontend" "http://localhost:5173"
check_endpoint "Backend API" "http://localhost:5000/api/health"
check_endpoint "AI Service API" "http://localhost:8000/health"

echo ""
info "Querying AI Service model status..."
if curl -s http://localhost:8000/health >/dev/null; then
    JSON_RESP=$(curl -s http://localhost:8000/health)
    
    # We can use python to parse JSON safely if jq isn't available
    python3 -c "
import sys, json
try:
    data = json.loads('''$JSON_RESP''')
    print('  AI Status:       ' + data.get('status', 'unknown'))
    print('  Compute Device:  ' + str(data.get('device', 'unknown')))
    models = data.get('models_loaded', {})
    if models:
        print('  Models Loaded:')
        for m, status in models.items():
            print(f'    - {m}: {\"\\033[0;32m[PASS]\\033[0m\" if status else \"\\033[0;31m[FAIL]\\033[0m\"}')
    else:
        print('  Models:          None loaded (or pipeline not active)')
except Exception as e:
    print('  Failed to parse AI health response')
"
else
    echo -e "  ${RED}AI Service is unreachable. Make sure it is running.${RESET}"
fi

echo ""
