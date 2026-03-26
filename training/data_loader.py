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

        # Build base vector ONCE for Tier 1 (patient state is static in historical data)
        # Then incrementally update Tier 2 (system state evolves) and Tier 3 (action context)
        from config import (
            ACTION_BY_ID, FEAT_IDX_GAP_FLAGS_START, MEASURE_WEIGHTS as MW,
            FEAT_IDX_CONTACT_HISTORY_START, FEAT_IDX_PATIENT_CHANNEL_SUCCESS_START,
            FEAT_IDX_GAP_CONTEXT_START, FEAT_IDX_PENDING_ACTIONS_START,
            FEAT_IDX_BUDGET_START, FEAT_IDX_TEMPORAL_START,
            FEAT_IDX_POPULATION_START, FEAT_IDX_STARS_PROGRESS_START,
            TIER3_START, AVG_MESSAGES_PER_PATIENT, MAX_CONTACTS_PER_WEEK,
            DAYS_SINCE_NORMALIZATION_MAX, YEAR_DAYS,
            BUDGET_WARNING_THRESHOLD, BUDGET_CRITICAL_THRESHOLD,
            compute_global_budget,
        )
        total_records = len(records)
        cohort_size = len(patient_records)
        budget_max = compute_global_budget(cohort_size)
        base_vec = snapshot_to_vector(snap)  # Build full 176-dim once
        state_vec = base_vec.copy()

        # Tier 3 accumulators
        messages_sent = 0
        responses = 0
        channels_used = set()
        ch_counts = {"sms": 0, "email": 0, "portal": 0, "app": 0, "ivr": 0}
        ch_last = {"sms": -999, "email": -999, "portal": -999, "app": -999, "ivr": -999}
        ch_success = {"sms": [0, 0], "email": [0, 0], "portal": [0, 0], "app": [0, 0], "ivr": [0, 0]}
        ch_order = ["sms", "email", "portal", "app", "ivr"]

        for i, record in enumerate(records):
            # Tier 2: System state evolves across the historical timeline
            progress = i / max(total_records - 1, 1)  # 0.0 → 1.0 across patient's history
            day_approx = int(progress * 365)
            budget_frac = max(0.0, 1.0 - progress * 0.6)  # Budget depletes ~60% over the year

            idx = FEAT_IDX_BUDGET_START
            state_vec[idx] = budget_frac; idx += 1                                         # budget_remaining_norm
            state_vec[idx] = 1.0 - budget_frac; idx += 1                                   # budget_utilization
            state_vec[idx] = 1.0 if budget_frac < BUDGET_WARNING_THRESHOLD else 0.0; idx += 1
            state_vec[idx] = 1.0 if budget_frac < BUDGET_CRITICAL_THRESHOLD else 0.0; idx += 1
            state_vec[idx] = min(0.6 / max(YEAR_DAYS, 1), 1.0); idx += 1                  # burn_rate_norm (steady)
            state_vec[idx] = budget_frac; idx += 1                                         # projected_days_left

            idx = FEAT_IDX_TEMPORAL_START
            state_vec[idx] = np.sin(2 * np.pi * day_approx / YEAR_DAYS); idx += 1
            state_vec[idx] = np.cos(2 * np.pi * day_approx / YEAR_DAYS); idx += 1
            state_vec[idx] = min(day_approx / YEAR_DAYS, 1.0); idx += 1                   # measurement_year_progress

            idx = FEAT_IDX_POPULATION_START
            state_vec[idx] = min(cohort_size / 10000.0, 1.0); idx += 1                    # cohort_size_norm
            state_vec[idx] = min(messages_sent / max(AVG_MESSAGES_PER_PATIENT, 1), 3.0) / 3.0; idx += 1

            # STARS progress approximation: increases with closures over time
            closure_rate_approx = len(closed_gaps) / max(len(open_gaps) + len(closed_gaps), 1)
            idx = FEAT_IDX_STARS_PROGRESS_START
            state_vec[idx] = min(1.0 + closure_rate_approx * 3.0, 5.0) / 5.0; idx += 1   # overall_stars_norm
            state_vec[idx] = 0.0; idx += 1                                                 # stars_7d_trend (unknown)
            state_vec[idx] = closure_rate_approx; idx += 1                                 # pct_measures_above_4
            state_vec[idx] = 0.2; idx += 1                                                 # lowest_measure_stars_norm

            # Fast incremental Tier 3 update (no full snapshot_to_vector call)
            idx = FEAT_IDX_CONTACT_HISTORY_START
            avg_msg = max(AVG_MESSAGES_PER_PATIENT, 1)
            state_vec[idx] = min(messages_sent / avg_msg, 3.0) / 3.0; idx += 1
            state_vec[idx] = min(messages_sent / avg_msg, 3.0) / 3.0; idx += 1
            state_vec[idx] = min(messages_sent, MAX_CONTACTS_PER_WEEK) / MAX_CONTACTS_PER_WEEK; idx += 1
            state_vec[idx] = min(messages_sent, 6) / 6.0; idx += 1
            state_vec[idx] = min(messages_sent, 12) / 12.0; idx += 1
            state_vec[idx] = min(max(0, i - messages_sent) if messages_sent > 0 else 90, 90) / 90.0; idx += 1
            state_vec[idx] = responses / max(messages_sent, 1); idx += 1
            state_vec[idx] = len(channels_used) / 5.0; idx += 1

            # Per-channel success rates
            idx = FEAT_IDX_PATIENT_CHANNEL_SUCCESS_START
            for ch in ch_order:
                s = ch_success[ch]
                state_vec[idx] = s[0] / max(s[1], 1); idx += 1

            # Channel affinity (volume + recency) — starts at TIER3_START + 26 = index 132
            aff_idx = TIER3_START + 26  # After contact(8) + ch_success(5) + engagement(5) + gap_ctx(5) + pending(3)
            for ch in ch_order:
                state_vec[aff_idx] = min(ch_counts[ch], 10) / 10.0; aff_idx += 1
            for ch in ch_order:
                state_vec[aff_idx] = min(i - ch_last[ch] if ch_last[ch] >= 0 else 90, 90) / 90.0; aff_idx += 1

            # Gap context
            idx = FEAT_IDX_GAP_CONTEXT_START
            num_open = len(open_gaps)
            num_closed = len(closed_gaps)
            max_w = max((MW.get(g, 1) for g in open_gaps), default=0)
            state_vec[idx] = num_open / 18.0; idx += 1
            state_vec[idx] = min(i * 7.0 / 365.0, 1.0); idx += 1
            state_vec[idx] = 1.0; idx += 1  # days_since_closure (approximate)
            state_vec[idx] = max_w / 3.0; idx += 1
            state_vec[idx] = num_closed / 18.0; idx += 1

            # Update gap flags
            for mi, m in enumerate(HEDIS_MEASURES):
                state_vec[FEAT_IDX_GAP_FLAGS_START + mi] = 1.0 if m in open_gaps else 0.0

            # Compute mask
            mask = compute_action_mask(open_gaps=open_gaps, channel_availability=channel_avail)

            action_id = record["action_id"]
            outcome = record["outcome"]
            measure = record["measure"]

            # Compute reward
            reward = 0.0
            if outcome.get("clicked"):
                reward += REWARD_WEIGHTS.get("engagement_click", 0.05)
            if outcome.get("gap_closed_within_30d") and measure:
                mw = MW.get(measure, 1)
                reward += REWARD_WEIGHTS["gap_closure"] * mw

            obs_list.append(state_vec.copy())
            action_list.append(action_id)
            reward_list.append(reward)
            mask_list.append(mask.copy())

            # Update accumulators
            action_info = ACTION_BY_ID.get(action_id)
            if action_info and action_info.measure != "NO_ACTION":
                messages_sent += 1
                ch = action_info.channel
                channels_used.add(ch)
                ch_counts[ch] = ch_counts.get(ch, 0) + 1
                ch_last[ch] = i
                ch_success[ch][1] += 1
                if outcome.get("clicked") or outcome.get("gap_closed_within_30d"):
                    responses += 1
                    ch_success[ch][0] += 1

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
