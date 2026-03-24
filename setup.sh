#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# NBA Stars Model — Environment Setup
# ============================================================================
# Run once to set up the Python environment and install all dependencies.
#
# Usage:
#   ./setup.sh           # Auto-detect Python, create venv, install deps
#   ./setup.sh --no-venv # Install into current Python (no virtual env)
#   source .venv/bin/activate && ./run.sh   # Then run the app
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
header() { echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; echo -e "${CYAN}  $*${NC}"; echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }

USE_VENV=true
if [[ "${1:-}" == "--no-venv" ]]; then
    USE_VENV=false
fi

# ── Find Python ──
header "Finding Python"
PYTHON=""
for cmd in python3.12 python3.11 python3.10 python3; do
    if command -v "$cmd" &>/dev/null; then
        version=$("$cmd" --version 2>&1 | awk '{print $2}')
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if (( major == 3 && minor >= 10 )); then
            PYTHON="$cmd"
            ok "Found $cmd ($version)"
            break
        else
            warn "$cmd is version $version (need 3.10+), skipping"
        fi
    fi
done

if [[ -z "$PYTHON" ]]; then
    err "Python 3.10+ not found. Install it first:
    macOS:   brew install python@3.12
    Ubuntu:  sudo apt install python3.12 python3.12-venv
    Windows: Download from python.org"
fi

# ── Create virtual environment ──
if $USE_VENV; then
    header "Setting Up Virtual Environment"
    if [[ -d .venv ]]; then
        warn "Virtual environment .venv already exists"
        read -p "  Recreate it? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf .venv
        else
            ok "Keeping existing .venv"
        fi
    fi

    if [[ ! -d .venv ]]; then
        info "Creating virtual environment..."
        "$PYTHON" -m venv .venv
        ok "Created .venv"
    fi

    # Activate
    source .venv/bin/activate
    PYTHON="python"
    ok "Activated .venv ($(python --version))"
fi

# ── Install dependencies ──
header "Installing Dependencies"
info "Upgrading pip..."
"$PYTHON" -m pip install --upgrade pip -q

info "Installing dependencies from requirements.txt..."
"$PYTHON" -m pip install -r requirements.txt -q 2>&1 | tail -3
ok "All dependencies installed"

# ── Verify installation ──
header "Verifying Installation"
"$PYTHON" -c "
import torch; print(f'  PyTorch {torch.__version__}')
import gymnasium; print(f'  Gymnasium {gymnasium.__version__}')
import dash; print(f'  Dash {dash.__version__}')
import plotly; print(f'  Plotly {plotly.__version__}')
import numpy; print(f'  NumPy {numpy.__version__}')
from config import NUM_ACTIONS, HEDIS_MEASURES; print(f'  Config: {NUM_ACTIONS} actions, {len(HEDIS_MEASURES)} measures')
from training.cql_trainer import ActorCriticCQL; print(f'  CQL Agent: OK')
from dashboard.app import create_app; print(f'  Dashboard: OK')
print()
print('  All imports verified!')
"
ok "Installation complete"

# ── Summary ──
header "Ready!"
echo ""
if $USE_VENV; then
    echo -e "  To activate the environment:"
    echo -e "    ${GREEN}source .venv/bin/activate${NC}"
    echo ""
fi
echo -e "  To run the full simulation + dashboard:"
echo -e "    ${GREEN}./run.sh${NC}"
echo ""
echo -e "  To run just the dashboard:"
echo -e "    ${GREEN}./run.sh dashboard${NC}"
echo ""
echo -e "  To run tests:"
echo -e "    ${GREEN}python -m pytest tests/ -v${NC}"
echo ""
