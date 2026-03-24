"""
Converts JSON datasets into RLlib-compatible offline format.
Produces (obs, action, reward, next_obs, done, action_mask) tuples.
"""
import json
import os
import numpy as np
from typing import List, Dict, Any, Tuple

from config import (
    GENERATED_DATA_DIR, NUM_ACTIONS, STATE_DIM,
    HEDIS_MEASURES, REWARD_WEIGHTS, MEASURE_WEIGHTS,
)
from environment.state_space import snapshot_to_vector
from environment.action_masking import compute_action_mask


def load_datasets(data_dir: str = GENERATED_DATA_DIR) -> Dict[str, Any]:
    """Load all generated datasets."""
    datasets = {}
    for name in ["state_features", "historical_activity", "gap_closure", "action_eligibility"]:
        filepath = os.path.join(data_dir, f"{name}.json")
        with open(filepath) as f:
            datasets[name] = json.load(f)
    return datasets


def build_offline_episodes(
    state_snapshots: List[Dict],
    historical_activity: List[Dict],
    eligibility_snapshots: List[Dict],
) -> List[Dict[str, np.ndarray]]:
    """Convert raw data into offline RL episodes.

    Each episode is a dict with arrays:
        obs: (T, STATE_DIM)
        actions: (T,)
        rewards: (T,)
        action_mask: (T, NUM_ACTIONS)
        terminateds: (T,)
        truncateds: (T,)
    """
    # Build lookups
    patient_states = {}
    for snap in state_snapshots:
        vec = snapshot_to_vector(snap)
        patient_states[snap["patient_id"]] = (vec, snap)

    patient_eligibility = {}
    for elig in eligibility_snapshots:
        patient_eligibility[elig["patient_id"]] = elig

    # Group historical records by patient
    patient_records: Dict[str, List] = {}
    for record in historical_activity:
        pid = record["patient_id"]
        patient_records.setdefault(pid, []).append(record)

    # Sort each patient's records by date
    for pid in patient_records:
        patient_records[pid].sort(key=lambda r: r["date"])

    episodes = []
    for pid, records in patient_records.items():
        if pid not in patient_states:
            continue

        base_vec, snap = patient_states[pid]
        elig = patient_eligibility.get(pid, {})

        obs_list = []
        action_list = []
        reward_list = []
        mask_list = []

        state_vec = base_vec.copy()
        open_gaps = set(snap.get("open_gaps", []))
        engagement = snap.get("engagement", {})

        channel_avail = {
            "sms": engagement.get("sms_consent", False),
            "email": engagement.get("email_available", False),
            "portal": engagement.get("portal_registered", False),
            "app": engagement.get("app_installed", False),
            "ivr": True,
        }

        for i, record in enumerate(records):
            # Compute mask for current state
            mask = compute_action_mask(
                open_gaps=open_gaps,
                channel_availability=channel_avail,
            )

            action_id = record["action_id"]
            outcome = record["outcome"]
            measure = record["measure"]

            # Compute reward (matches simplified reward function)
            reward = REWARD_WEIGHTS["action_cost"]
            if outcome.get("clicked"):
                reward += REWARD_WEIGHTS.get("engagement_click", 0.05)
            if outcome.get("gap_closed_within_30d") and measure:
                mw = MEASURE_WEIGHTS.get(measure, 1)
                reward += REWARD_WEIGHTS["gap_closure"] * mw

            obs_list.append(state_vec.copy())
            action_list.append(action_id)
            reward_list.append(reward)
            mask_list.append(mask.copy())

            # Update state for next step
            if outcome.get("gap_closed_within_30d") and measure in HEDIS_MEASURES:
                open_gaps.discard(measure)
                gap_idx = 24 + HEDIS_MEASURES.index(measure)
                state_vec[gap_idx] = 0.0

            # Small state perturbation
            state_vec += np.random.normal(0, 0.005, STATE_DIM).astype(np.float32)
            state_vec = np.clip(state_vec, -5.0, 5.0)

        if len(obs_list) < 2:
            continue

        T = len(obs_list)
        episodes.append({
            "obs": np.array(obs_list, dtype=np.float32),
            "actions": np.array(action_list, dtype=np.int64),
            "rewards": np.array(reward_list, dtype=np.float32),
            "action_mask": np.array(mask_list, dtype=np.float32),
            "terminateds": np.zeros(T, dtype=np.float32),
            "truncateds": np.array([0.0] * (T - 1) + [1.0], dtype=np.float32),
        })

    return episodes


def episodes_to_sample_batches(episodes: List[Dict]) -> List[Dict]:
    """Convert episodes to flat sample batch dicts for RLlib offline input."""
    batches = []
    for ep in episodes:
        T = len(ep["obs"])
        for t in range(T - 1):
            batch = {
                "obs": {
                    "observations": ep["obs"][t].tolist(),
                    "action_mask": ep["action_mask"][t].tolist(),
                },
                "actions": int(ep["actions"][t]),
                "rewards": float(ep["rewards"][t]),
                "new_obs": {
                    "observations": ep["obs"][t + 1].tolist(),
                    "action_mask": ep["action_mask"][min(t + 1, T - 1)].tolist(),
                },
                "dones": bool(ep["terminateds"][t] or ep["truncateds"][t]),
                "truncateds": bool(ep["truncateds"][t]),
                "terminateds": bool(ep["terminateds"][t]),
            }
            batches.append(batch)
    return batches


def save_offline_data(batches: List[Dict], output_path: str):
    """Save sample batches as JSON lines file for RLlib offline input."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for batch in batches:
            f.write(json.dumps(batch) + "\n")
    print(f"Saved {len(batches)} transitions to {output_path}")
