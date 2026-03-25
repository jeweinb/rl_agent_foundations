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

        open_gaps = set(snap.get("open_gaps", []))
        closed_gaps = set(snap.get("closed_gaps", []))
        engagement = snap.get("engagement", {})

        channel_avail = {
            "sms": engagement.get("sms_consent", False),
            "email": engagement.get("email_available", False),
            "portal": engagement.get("portal_registered", False),
            "app": engagement.get("app_installed", False),
            "ivr": True,
        }

        # Track Tier 3 action context as we process historical records
        from config import ACTION_BY_ID
        messages_sent = 0
        responses = 0
        channels_used = set()
        channel_counts = {"sms": 0, "email": 0, "portal": 0, "app": 0, "ivr": 0}
        channel_last_step = {"sms": -999, "email": -999, "portal": -999, "app": -999, "ivr": -999}
        channel_success = {"sms": [0, 0], "email": [0, 0], "portal": [0, 0], "app": [0, 0], "ivr": [0, 0]}  # [successes, attempts]

        for i, record in enumerate(records):
            # Build state vector with accumulated action context
            action_info = ACTION_BY_ID.get(record["action_id"])
            response_rate = responses / max(messages_sent, 1)
            ch_success_rates = {
                ch: (s[0] / max(s[1], 1)) for ch, s in channel_success.items()
            }
            ch_recency = {
                ch: (i - d if d >= 0 else 90) for ch, d in channel_last_step.items()
            }

            state_vec = snapshot_to_vector(
                snap,
                # Tier 3: accumulated action context
                patient_messages_received=messages_sent,
                patient_response_rate=response_rate,
                patient_contacts_7d=min(messages_sent, 3),  # Approximate from history
                patient_days_since_contact=max(0, i - messages_sent) if messages_sent > 0 else 90,
                patient_channels_used=len(channels_used),
                patient_channel_success=ch_success_rates,
                patient_days_since_closure=90.0,
                patient_avg_gap_age=float(i * 7),  # Approximate: ~1 record per week
                channel_affinity_counts=channel_counts,
                channel_affinity_recency=ch_recency,
            )

            # Update gap flags in the vector to reflect closures so far
            from config import FEAT_IDX_GAP_FLAGS_START
            for m in HEDIS_MEASURES:
                gap_idx = FEAT_IDX_GAP_FLAGS_START + HEDIS_MEASURES.index(m)
                state_vec[gap_idx] = 1.0 if m in open_gaps else 0.0

            # Compute mask for current state
            mask = compute_action_mask(
                open_gaps=open_gaps,
                channel_availability=channel_avail,
            )

            action_id = record["action_id"]
            outcome = record["outcome"]
            measure = record["measure"]

            # Compute reward
            reward = 0.0
            if outcome.get("clicked"):
                reward += REWARD_WEIGHTS.get("engagement_click", 0.05)
            if outcome.get("gap_closed_within_30d") and measure:
                mw = MEASURE_WEIGHTS.get(measure, 1)
                reward += REWARD_WEIGHTS["gap_closure"] * mw

            obs_list.append(state_vec.copy())
            action_list.append(action_id)
            reward_list.append(reward)
            mask_list.append(mask.copy())

            # Update accumulated action context for next step
            if action_info and action_info.measure != "NO_ACTION":
                messages_sent += 1
                ch = action_info.channel
                channels_used.add(ch)
                channel_counts[ch] = channel_counts.get(ch, 0) + 1
                channel_last_step[ch] = i
                channel_success[ch][1] += 1  # attempt
                if outcome.get("clicked") or outcome.get("gap_closed_within_30d"):
                    responses += 1
                    channel_success[ch][0] += 1  # success

            # Update gap state
            if outcome.get("gap_closed_within_30d") and measure in HEDIS_MEASURES:
                open_gaps.discard(measure)
                closed_gaps.add(measure)

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
