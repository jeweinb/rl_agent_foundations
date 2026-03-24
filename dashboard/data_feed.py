"""
Data feed for the dashboard.
Reads simulation JSON files and returns DataFrames/dicts for visualization.
"""
import json
import os
import glob
from typing import Dict, Any, List, Optional

from config import SIMULATION_DATA_DIR, HEDIS_MEASURES


def load_cumulative_metrics() -> List[Dict[str, Any]]:
    """Load cumulative metrics from simulation output."""
    path = os.path.join(SIMULATION_DATA_DIR, "cumulative_metrics.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def load_day_actions(day: int) -> List[Dict[str, Any]]:
    """Load actions taken on a specific day."""
    path = os.path.join(SIMULATION_DATA_DIR, f"day_{day:02d}", "actions_taken.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def load_all_actions() -> List[Dict[str, Any]]:
    """Load all actions across all simulation days."""
    all_actions = []
    for day_dir in sorted(glob.glob(os.path.join(SIMULATION_DATA_DIR, "day_*"))):
        actions_path = os.path.join(day_dir, "actions_taken.json")
        if os.path.exists(actions_path):
            with open(actions_path) as f:
                all_actions.extend(json.load(f))
    return all_actions


def load_nightly_metrics(day: int) -> Optional[Dict[str, Any]]:
    """Load nightly training metrics for a specific day."""
    path = os.path.join(SIMULATION_DATA_DIR, f"day_{day:02d}", "nightly_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_all_nightly_metrics() -> List[Dict[str, Any]]:
    """Load all nightly metrics."""
    metrics = []
    for day_dir in sorted(glob.glob(os.path.join(SIMULATION_DATA_DIR, "day_*"))):
        path = os.path.join(day_dir, "nightly_metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                metrics.append(json.load(f))
    return metrics


def load_state_machine_data(day: int = None) -> List[Dict[str, Any]]:
    """Load state machine records. If day is None, load latest available."""
    if day is None:
        # Find latest day
        day_dirs = sorted(glob.glob(os.path.join(SIMULATION_DATA_DIR, "day_*")))
        if not day_dirs:
            return []
        day_dir = day_dirs[-1]
    else:
        day_dir = os.path.join(SIMULATION_DATA_DIR, f"day_{day:02d}")

    path = os.path.join(day_dir, "state_machine.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def load_all_state_machine_data() -> List[Dict[str, Any]]:
    """Load cumulative state machine records."""
    # Prefer the cumulative file (written each day with all records)
    cumulative_path = os.path.join(SIMULATION_DATA_DIR, "state_machine_cumulative.json")
    if os.path.exists(cumulative_path):
        with open(cumulative_path) as f:
            return json.load(f)
    # Fallback: load from latest day
    day_dirs = sorted(glob.glob(os.path.join(SIMULATION_DATA_DIR, "day_*")))
    if day_dirs:
        path = os.path.join(day_dirs[-1], "state_machine.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return []


def get_patient_journey(patient_id: str) -> List[Dict[str, Any]]:
    """Get full action history for a specific patient across all days."""
    all_actions = load_all_actions()
    return [a for a in all_actions if a.get("patient_id") == patient_id]


def get_all_patient_ids() -> List[str]:
    """Get list of all patient IDs that appear in simulation data."""
    all_actions = load_all_actions()
    return sorted(set(a.get("patient_id", "") for a in all_actions if a.get("patient_id")))


def load_sim_predictions() -> List[Dict[str, Any]]:
    """Load simulation predictions from the learned world (nightly eval rollouts)."""
    path = os.path.join(SIMULATION_DATA_DIR, "sim_predictions.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def load_simulation_logs(max_lines: int = 200) -> List[Dict[str, Any]]:
    """Load simulation log entries."""
    path = os.path.join(SIMULATION_DATA_DIR, "simulation_log.jsonl")
    if not os.path.exists(path):
        return []
    logs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return logs[-max_lines:]


def get_latest_day() -> int:
    """Get the latest simulation day number."""
    day_dirs = glob.glob(os.path.join(SIMULATION_DATA_DIR, "day_*"))
    if not day_dirs:
        return 0
    days = []
    for d in day_dirs:
        try:
            days.append(int(os.path.basename(d).split("_")[1]))
        except (ValueError, IndexError):
            pass
    return max(days) if days else 0
