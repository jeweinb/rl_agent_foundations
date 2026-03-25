#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# HEDIS STARS RL Agent — Run Script
# ============================================================================
# Usage:
#   ./run.sh              — Full run: generate data + start dashboard + simulation
#   ./run.sh start        — Same as above
#   ./run.sh stop         — Stop all running processes
#   ./run.sh restart      — Stop then start fresh
#   ./run.sh status       — Show what's running
#   ./run.sh dashboard    — Start only the dashboard
#   ./run.sh simulate     — Start only the simulation (assumes data exists)
#   ./run.sh generate     — Generate data only
#   ./run.sh clean        — Stop everything and wipe all generated data
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Find Python 3.10+ ---
PYTHON=""
if [[ -d "$SCRIPT_DIR/.venv" ]]; then
    # Prefer venv if it exists
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
elif command -v python3.12 &>/dev/null; then
    PYTHON="python3.12"
elif command -v python3.11 &>/dev/null; then
    PYTHON="python3.11"
elif command -v python3.10 &>/dev/null; then
    PYTHON="python3.10"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    echo "ERROR: Python 3.10+ not found. Run ./setup.sh first."
    exit 1
fi

# Auto-activate venv if it exists
if [[ -d "$SCRIPT_DIR/.venv" && -z "${VIRTUAL_ENV:-}" ]]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
    PYTHON="python"
fi


PID_DIR="$SCRIPT_DIR/.pids"
LOG_DIR="$SCRIPT_DIR/logs"
DASHBOARD_PID="$PID_DIR/dashboard.pid"
SIMULATION_PID="$PID_DIR/simulation.pid"
DASHBOARD_LOG="$LOG_DIR/dashboard.log"
SIMULATION_LOG="$LOG_DIR/simulation.log"
DASHBOARD_PORT=8050

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# --- Helpers ---
info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }
header() { echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; echo -e "${CYAN}  $*${NC}"; echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }

ensure_dirs() {
    mkdir -p "$PID_DIR" "$LOG_DIR" data/generated data/simulation training/checkpoints
    # Auto-clean stale PID files on every invocation
    for pidfile in "$PID_DIR"/*.pid; do
        [[ -f "$pidfile" ]] || continue
        local pid
        pid=$(cat "$pidfile" 2>/dev/null)
        if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
            rm -f "$pidfile"
        fi
    done
}

is_running() {
    local pidfile="$1"
    if [[ -f "$pidfile" ]]; then
        local pid
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        rm -f "$pidfile"
    fi
    return 1
}

kill_process() {
    local pidfile="$1"
    local name="$2"
    if is_running "$pidfile"; then
        local pid
        pid=$(cat "$pidfile")
        info "Stopping $name (PID $pid)..."
        kill "$pid" 2>/dev/null || true
        for i in {1..10}; do
            if ! kill -0 "$pid" 2>/dev/null; then
                break
            fi
            sleep 0.5
        done
        if kill -0 "$pid" 2>/dev/null; then
            warn "Force-killing $name (PID $pid)"
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$pidfile"
        ok "$name stopped"
    fi
}

kill_stale() {
    # Kill any orphaned processes from previous runs
    local stale
    stale=$(ps aux | grep -E "scripts/run_dashboard|scripts/run_simulation" | grep -v grep | awk '{print $2}' || true)
    if [[ -n "$stale" ]]; then
        warn "Found stale processes: $stale"
        echo "$stale" | xargs kill 2>/dev/null || true
        sleep 1
        echo "$stale" | xargs kill -9 2>/dev/null || true
        ok "Stale processes cleaned"
    fi
}

check_port() {
    if command -v ss &>/dev/null; then
        ss -tlnp 2>/dev/null | grep -q ":${DASHBOARD_PORT} " && return 0
    elif command -v lsof &>/dev/null; then
        lsof -i ":${DASHBOARD_PORT}" &>/dev/null && return 0
    fi
    return 1
}

wait_for_port() {
    local max_wait=15
    for ((i=1; i<=max_wait; i++)); do
        if check_port; then
            return 0
        fi
        sleep 1
    done
    return 1
}

# --- Commands ---

cmd_stop() {
    header "Stopping all processes"
    kill_process "$DASHBOARD_PID" "Dashboard"
    kill_process "$SIMULATION_PID" "Simulation"
    kill_stale
    ok "All processes stopped"
}

cmd_status() {
    header "Process Status"
    if is_running "$DASHBOARD_PID"; then
        ok "Dashboard:  running (PID $(cat "$DASHBOARD_PID")) → \e]8;;http://localhost:$DASHBOARD_PORT\e\\http://localhost:$DASHBOARD_PORT\e]8;;\e\\"
    else
        warn "Dashboard:  not running"
    fi
    if is_running "$SIMULATION_PID"; then
        ok "Simulation: running (PID $(cat "$SIMULATION_PID"))"
        # Show latest log line
        if [[ -f "$SIMULATION_LOG" ]]; then
            local last
            last=$(tail -1 "$SIMULATION_LOG" 2>/dev/null || true)
            info "  Last log: $last"
        fi
    else
        warn "Simulation: not running"
    fi
    # Show simulation progress
    if [[ -f data/simulation/cumulative_metrics.json ]]; then
        local days
        days=$(python3 -c "import json; d=json.load(open('data/simulation/cumulative_metrics.json')); print(f'Day {d[-1][\"day\"]}/30 — STARS: {d[-1][\"stars_score\"]:.2f} — Reward: {d[-1][\"cumulative_reward\"]:.2f} — Model: v{d[-1][\"model_version\"]}')" 2>/dev/null || echo "error reading metrics")
        info "  Progress: $days"
    fi
}

cmd_generate() {
    header "Generating Data"
    info "Generating 5000 patient mock dataset..."
    python3 scripts/generate_data.py --cohort-size 5000
    ok "Data generated in data/generated/"
}

cmd_dashboard() {
    header "Starting Dashboard"

    # Stop existing dashboard
    kill_process "$DASHBOARD_PID" "existing Dashboard"

    # Check port
    if check_port; then
        warn "Port $DASHBOARD_PORT already in use, killing occupant..."
        if command -v fuser &>/dev/null; then
            fuser -k "${DASHBOARD_PORT}/tcp" 2>/dev/null || true
        fi
        sleep 1
    fi

    info "Launching Dash app on port $DASHBOARD_PORT..."
    python3 scripts/run_dashboard.py --port "$DASHBOARD_PORT" \
        > "$DASHBOARD_LOG" 2>&1 &
    echo $! > "$DASHBOARD_PID"

    if wait_for_port; then
        ok "Dashboard running → \e]8;;http://localhost:$DASHBOARD_PORT\e\\http://localhost:$DASHBOARD_PORT\e]8;;\e\\"
    else
        warn "Dashboard may still be starting — check $DASHBOARD_LOG"
    fi
}

cmd_simulate() {
    header "Starting Simulation"

    # Check data exists with enough patients
    local count=0
    if [[ -f data/generated/state_features.json ]]; then
        count=$(python3 -c "import json; print(len(json.load(open('data/generated/state_features.json'))))" 2>/dev/null || echo "0")
    fi
    if (( count < 1000 )); then
        warn "No data or insufficient patients ($count). Generating 5000..."
        cmd_generate
    fi

    # Stop existing simulation
    kill_process "$SIMULATION_PID" "existing Simulation"

    # Clean previous simulation data
    info "Clearing previous simulation data..."
    rm -rf data/simulation/* training/checkpoints/*
    mkdir -p data/simulation training/checkpoints

    info "Starting 90-day simulation..."
    python3 -u scripts/run_simulation.py \
        --days 90 --bc-epochs 5 --cql-epochs 3 --eval-episodes 50 \
        > "$SIMULATION_LOG" 2>&1 &
    echo $! > "$SIMULATION_PID"

    ok "Simulation started (PID $(cat "$SIMULATION_PID"))"
    info "Logs: tail -f $SIMULATION_LOG"
}

cmd_start() {
    header "HEDIS STARS RL Agent"

    # Stop everything first
    cmd_stop

    # Generate data if needed — also check it has enough patients (not a leftover test file)
    local min_patients=1000
    local need_gen=false
    if [[ ! -f data/generated/state_features.json ]]; then
        need_gen=true
    else
        local count
        count=$(python3 -c "import json; print(len(json.load(open('data/generated/state_features.json'))))" 2>/dev/null || echo "0")
        if (( count < min_patients )); then
            warn "Data exists but only has $count patients (need $min_patients+). Regenerating..."
            need_gen=true
        else
            ok "Data exists with $count patients"
        fi
    fi
    if $need_gen; then
        cmd_generate
    fi

    # Start dashboard
    cmd_dashboard

    # Start simulation
    cmd_simulate

    echo ""
    header "All systems running!"
    echo ""
    echo -e "  ${GREEN}Dashboard${NC}:    \e]8;;http://localhost:$DASHBOARD_PORT\e\\${CYAN}http://localhost:$DASHBOARD_PORT${NC}\e]8;;\e\\"
    echo -e "  ${GREEN}Sim Logs${NC}:     tail -f $SIMULATION_LOG"
    echo -e "  ${GREEN}Dash Logs${NC}:    tail -f $DASHBOARD_LOG"
    echo -e "  ${GREEN}Stop${NC}:         ./run.sh stop"
    echo -e "  ${GREEN}Status${NC}:       ./run.sh status"
    echo ""
    info "Click the link above or open http://localhost:$DASHBOARD_PORT"
    info "90-day simulation (~60-75 min). Watch STARS climb toward 4.0!"
}

cmd_restart() {
    cmd_stop
    sleep 1
    cmd_start
}

cmd_hot() {
    header "Hot-Restarting Dashboard"
    info "Simulation keeps running, only dashboard restarts..."
    kill_process "$DASHBOARD_PID" "Dashboard"
    if check_port; then
        warn "Port $DASHBOARD_PORT still in use, waiting..."
        sleep 2
    fi
    info "Launching fresh dashboard..."
    python3 scripts/run_dashboard.py --port "$DASHBOARD_PORT" \
        > "$DASHBOARD_LOG" 2>&1 &
    echo $! > "$DASHBOARD_PID"
    if wait_for_port; then
        ok "Dashboard hot-restarted → \e]8;;http://localhost:$DASHBOARD_PORT\e\\http://localhost:$DASHBOARD_PORT\e]8;;\e\\"
    else
        warn "Dashboard may still be starting — check $DASHBOARD_LOG"
    fi
}

cmd_clean() {
    header "Cleaning Everything"
    cmd_stop
    info "Removing generated data..."
    rm -rf data/generated/* data/simulation/* training/checkpoints/* logs/*
    ok "All data cleaned. Run ./run.sh start to begin fresh."
}

cmd_logs() {
    if [[ -f "$SIMULATION_LOG" ]]; then
        tail -f "$SIMULATION_LOG"
    else
        err "No simulation log found. Is the simulation running?"
        exit 1
    fi
}

# --- Main ---
ensure_dirs

case "${1:-start}" in
    start)     cmd_start ;;
    stop)      cmd_stop ;;
    restart)   cmd_restart ;;
    hot)       cmd_hot ;;
    status)    cmd_status ;;
    dashboard) cmd_dashboard ;;
    simulate)  cmd_simulate ;;
    generate)  cmd_generate ;;
    clean)     cmd_clean ;;
    logs)      cmd_logs ;;
    *)
        echo "Usage: ./run.sh {start|stop|restart|hot|status|dashboard|simulate|generate|clean|logs}"
        exit 1
        ;;
esac
