"""
Generate historical activity dataset — what the business has been doing before RL.

The data encodes learnable patterns that BC can pick up:
1. Channel-measure affinity: certain channels work better for certain measures
2. Patient-context matching: engaged patients respond more
3. Timing: well-spaced contacts work better than rapid-fire
4. Measure priority: acting on clinically relevant gaps drives closure
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

from config import (
    get_measure_category,
    BEST_CHANNEL_BY_CATEGORY,
    SECOND_BEST_CHANNEL_BY_CATEGORY as SECOND_BEST_CHANNEL,
)


def generate_historical_activity(
    patients: List[Dict[str, Any]],
    state_snapshots: List[Dict[str, Any]],
    n_records: int = HISTORICAL_RECORDS,
    rng: np.random.Generator = None,
) -> List[Dict[str, Any]]:
    """Generate historical outreach records with learnable patterns."""
    if rng is None:
        rng = np.random.default_rng(42)

    start_date = datetime.strptime(HISTORICAL_DATE_RANGE[0], "%Y-%m-%d")
    end_date = datetime.strptime(HISTORICAL_DATE_RANGE[1], "%Y-%m-%d")
    date_range_days = (end_date - start_date).days

    # Build patient lookup
    patient_gaps = {}
    patient_engagement = {}
    for snap in state_snapshots:
        pid = snap["patient_id"]
        patient_gaps[pid] = snap["open_gaps"] + snap["closed_gaps"]
        patient_engagement[pid] = snap.get("engagement", {})

    patient_ids = [p["patient_id"] for p in patients]
    channels = list(BEHAVIORAL_CHANNEL_PROBS.keys())
    channel_probs = list(BEHAVIORAL_CHANNEL_PROBS.values())

    # Track per-patient contact history for timing patterns
    patient_last_contact: Dict[str, int] = {}
    patient_measure_contacts: Dict[str, Dict[str, int]] = {}

    records = []
    for i in range(n_records):
        record_id = f"act_{i:06d}"
        pid = rng.choice(patient_ids)

        day_offset = int(rng.integers(0, date_range_days))
        action_date = start_date + timedelta(days=day_offset)
        date_str = action_date.strftime("%Y-%m-%d")

        eligible = patient_gaps.get(pid, ["COL", "FLU"])
        if not eligible:
            eligible = ["COL", "FLU"]
        measure = rng.choice(eligible)
        category = get_measure_category(measure)

        # --- PATTERN 1: Channel selection with measure affinity ---
        # 40% of time use best channel, 25% second-best, 35% random
        # This creates a learnable pattern: best channels → better outcomes
        roll = rng.random()
        if roll < 0.40:
            channel = BEST_CHANNEL_BY_CATEGORY.get(category, "sms")
        elif roll < 0.65:
            channel = SECOND_BEST_CHANNEL.get(category, "email")
        else:
            channel = rng.choice(channels, p=channel_probs)

        # Check channel availability
        eng = patient_engagement.get(pid, {})
        if channel == "sms" and not eng.get("sms_consent", True):
            channel = "email"
        if channel == "app" and not eng.get("app_installed", False):
            channel = "sms" if eng.get("sms_consent", True) else "email"
        if channel == "portal" and not eng.get("portal_registered", False):
            channel = "email"

        # Find matching action
        matching_actions = [
            a for a in ACTION_CATALOG
            if a.measure == measure and a.channel == channel
        ]
        if matching_actions:
            action = matching_actions[int(rng.integers(0, len(matching_actions)))]
        else:
            measure_actions = [a for a in ACTION_CATALOG if a.measure == measure]
            if measure_actions:
                action = measure_actions[int(rng.integers(0, len(measure_actions)))]
                channel = action.channel
            else:
                continue

        # --- PATTERN 2: Engagement based on patient profile ---
        # Patients with higher response rates engage more
        patient_response = eng.get(f"{channel}_response_rate",
                                   eng.get("sms_response_rate", 0.3))
        if channel == "email":
            patient_response = eng.get("email_open_rate", 0.2)
        elif channel == "app":
            patient_response = eng.get("app_engagement_rate", 0.4)

        delivered = bool(rng.random() < DELIVERY_RATES.get(channel, 0.9))
        # Open rate is boosted by patient engagement level
        base_open = OPEN_RATES.get(channel, 0.3)
        open_prob = base_open * (0.5 + patient_response)  # Engaged patients open more
        opened = bool(delivered and rng.random() < min(open_prob, 0.95))

        base_click = CLICK_RATES.get(channel, 0.1)
        click_prob = base_click * (0.5 + patient_response * 1.5)
        clicked = bool(opened and rng.random() < min(click_prob, 0.60))

        # --- PATTERN 3: Timing affects outcomes ---
        days_since_last = patient_last_contact.get(pid, 30)
        measure_contacts = patient_measure_contacts.get(pid, {}).get(measure, 0)

        # Well-spaced contacts (7+ days) work better
        timing_factor = 1.0
        if days_since_last < 3:
            timing_factor = 0.3  # Too frequent — patient ignores
        elif days_since_last < 7:
            timing_factor = 0.7
        elif days_since_last > 14:
            timing_factor = 1.2  # Patient had time to act

        # Diminishing returns on same measure
        repetition_factor = max(0.3, 1.0 - measure_contacts * 0.15)

        # --- PATTERN 4: Channel-measure match drives closure ---
        best_ch = BEST_CHANNEL_BY_CATEGORY.get(category, "sms")
        channel_match_factor = 1.0
        if channel == best_ch:
            channel_match_factor = 1.8  # Best channel → much higher closure
        elif channel == SECOND_BEST_CHANNEL.get(category, "email"):
            channel_match_factor = 1.3  # Second best → moderate boost

        # --- Compute closure probability ---
        base_rate = GAP_CLOSURE_BASE_RATES.get(measure, 0.5)
        closure_prob_30d = (base_rate * 0.25 *
                           channel_match_factor *
                           timing_factor *
                           repetition_factor)

        if clicked:
            closure_prob_30d *= 3.0  # Strong signal: click → closure
        elif opened:
            closure_prob_30d *= 1.5

        closure_prob_90d = closure_prob_30d * 2.5

        gap_closed_30d = bool(rng.random() < min(closure_prob_30d, 0.85))
        gap_closed_90d = bool(gap_closed_30d or rng.random() < min(closure_prob_90d, 0.90))
        days_to_closure = None
        if gap_closed_30d:
            days_to_closure = int(rng.integers(1, 31))
        elif gap_closed_90d:
            days_to_closure = int(rng.integers(31, 91))

        # Update tracking
        patient_last_contact[pid] = 0
        if pid not in patient_measure_contacts:
            patient_measure_contacts[pid] = {}
        patient_measure_contacts[pid][measure] = measure_contacts + 1

        # Age all contact timers
        for p in list(patient_last_contact.keys()):
            if p != pid:
                patient_last_contact[p] = patient_last_contact.get(p, 0) + 1

        prior_attempts = measure_contacts
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
                "days_since_last_contact": max(days_since_last, 1),
                "member_tenure_months": int(rng.integers(6, 120)),
            },
        }
        records.append(record)

    records.sort(key=lambda r: r["date"])
    return records
