"""
Generate historical activity dataset — what the business has been doing before RL.
Each record represents a past outreach action taken on a patient.
"""
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any

from datagen.constants import (
    HISTORICAL_RECORDS, HISTORICAL_DATE_RANGE,
    BEHAVIORAL_CHANNEL_PROBS, DELIVERY_RATES, OPEN_RATES, CLICK_RATES,
    GAP_CLOSURE_BASE_RATES, OUTREACH_LIFT,
)
from config import (
    HEDIS_MEASURES, ACTION_CATALOG, ACTION_IDS_BY_MEASURE,
    MEASURE_CATEGORIES,
)


def _get_category(measure: str) -> str:
    for cat, measures in MEASURE_CATEGORIES.items():
        if measure in measures:
            return cat
    return "unknown"


def generate_historical_activity(
    patients: List[Dict[str, Any]],
    state_snapshots: List[Dict[str, Any]],
    n_records: int = HISTORICAL_RECORDS,
    rng: np.random.Generator = None,
) -> List[Dict[str, Any]]:
    """Generate historical outreach records from the behavioral policy."""
    if rng is None:
        rng = np.random.default_rng(42)

    start_date = datetime.strptime(HISTORICAL_DATE_RANGE[0], "%Y-%m-%d")
    end_date = datetime.strptime(HISTORICAL_DATE_RANGE[1], "%Y-%m-%d")
    date_range_days = (end_date - start_date).days

    # Build patient lookup
    patient_gaps = {}
    for snap in state_snapshots:
        pid = snap["patient_id"]
        patient_gaps[pid] = snap["open_gaps"] + snap["closed_gaps"]  # All eligible measures

    patient_ids = [p["patient_id"] for p in patients]
    channels = list(BEHAVIORAL_CHANNEL_PROBS.keys())
    channel_probs = list(BEHAVIORAL_CHANNEL_PROBS.values())

    records = []
    for i in range(n_records):
        record_id = f"act_{i:06d}"
        pid = rng.choice(patient_ids)

        # Pick a date
        day_offset = int(rng.integers(0, date_range_days))
        action_date = start_date + timedelta(days=day_offset)
        date_str = action_date.strftime("%Y-%m-%d")

        # Pick a measure the patient is eligible for
        eligible = patient_gaps.get(pid, ["COL", "FLU"])  # fallback
        if not eligible:
            eligible = ["COL", "FLU"]
        measure = rng.choice(eligible)

        # Pick a channel using behavioral policy distribution
        channel = rng.choice(channels, p=channel_probs)

        # Find matching action from catalog, or pick closest
        matching_actions = [
            a for a in ACTION_CATALOG
            if a.measure == measure and a.channel == channel
        ]
        if matching_actions:
            action = matching_actions[int(rng.integers(0, len(matching_actions)))]
        else:
            # Fallback: pick any action for this measure
            measure_actions = [a for a in ACTION_CATALOG if a.measure == measure]
            if measure_actions:
                action = measure_actions[int(rng.integers(0, len(measure_actions)))]
                channel = action.channel
            else:
                continue

        # Simulate outcome
        delivered = bool(rng.random() < DELIVERY_RATES.get(channel, 0.9))
        opened = bool(delivered and rng.random() < OPEN_RATES.get(channel, 0.3))
        clicked = bool(opened and rng.random() < CLICK_RATES.get(channel, 0.1))

        # Gap closure (lagged outcome)
        base_rate = GAP_CLOSURE_BASE_RATES.get(measure, 0.5)
        lift = OUTREACH_LIFT.get(channel, 1.0)
        closure_prob_30d = base_rate * lift * 0.15  # ~monthly slice of annual rate
        closure_prob_90d = base_rate * lift * 0.40

        # Engagement boosts closure probability
        if clicked:
            closure_prob_30d *= 1.8
            closure_prob_90d *= 1.5
        elif opened:
            closure_prob_30d *= 1.3
            closure_prob_90d *= 1.2

        gap_closed_30d = bool(rng.random() < min(closure_prob_30d, 0.95))
        gap_closed_90d = bool(gap_closed_30d or rng.random() < min(closure_prob_90d, 0.95))
        days_to_closure = None
        if gap_closed_30d:
            days_to_closure = int(rng.integers(1, 31))
        elif gap_closed_90d:
            days_to_closure = int(rng.integers(31, 91))

        # Context
        prior_attempts = int(rng.poisson(2))
        days_since_last = int(rng.integers(1, 90))

        record = {
            "record_id": record_id,
            "patient_id": pid,
            "date": date_str,
            "action_id": action.action_id,
            "measure": measure,
            "channel": channel,
            "variant": action.variant,
            "outcome": {
                "delivered": delivered,
                "opened": opened,
                "clicked": clicked,
                "gap_closed_within_30d": gap_closed_30d,
                "gap_closed_within_90d": gap_closed_90d,
                "days_to_closure": days_to_closure,
            },
            "context": {
                "prior_attempts_this_measure": prior_attempts,
                "days_since_last_contact": days_since_last,
                "member_tenure_months": int(rng.integers(6, 120)),
            },
        }
        records.append(record)

    # Sort by date
    records.sort(key=lambda r: r["date"])
    return records
