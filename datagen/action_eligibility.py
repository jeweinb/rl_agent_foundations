"""
Generate action eligibility / constraint snapshots for each patient.
Determines which of the 125 curated actions each patient can receive.
"""
import numpy as np
from typing import List, Dict, Any

from datagen.constants import (
    OPT_OUT_RATE, GRIEVANCE_HOLD_RATE, SUPPRESSION_RATE,
)
from config import (
    ACTION_CATALOG, HEDIS_MEASURES, NUM_ACTIONS,
    MAX_CONTACTS_PER_WEEK, MIN_DAYS_BETWEEN_SAME_MEASURE,
)


def generate_action_eligibility(
    state_snapshots: List[Dict[str, Any]],
    rng: np.random.Generator = None,
) -> List[Dict[str, Any]]:
    """Generate eligibility snapshots showing which actions each patient can receive."""
    if rng is None:
        rng = np.random.default_rng(42)

    eligibility_snapshots = []

    for snap in state_snapshots:
        pid = snap["patient_id"]
        engagement = snap["engagement"]
        open_gaps = set(snap["open_gaps"])

        # Global constraints
        opt_out = bool(rng.random() < OPT_OUT_RATE)
        grievance_hold = bool(rng.random() < GRIEVANCE_HOLD_RATE)
        suppression_active = bool(rng.random() < SUPPRESSION_RATE)
        contacts_this_week = int(rng.integers(0, MAX_CONTACTS_PER_WEEK + 1))

        global_constraints = {
            "opt_out": opt_out,
            "grievance_hold": grievance_hold,
            "suppression_active": suppression_active,
            "max_contacts_per_week": MAX_CONTACTS_PER_WEEK,
            "contacts_this_week": contacts_this_week,
            "min_days_between_same_measure": MIN_DAYS_BETWEEN_SAME_MEASURE,
        }

        # Per-action eligibility
        action_mask = [False] * NUM_ACTIONS
        action_mask[0] = True  # no_action always available

        blocked_reasons = {}

        # If globally blocked, only no_action
        if opt_out or grievance_hold or suppression_active:
            reason = "opt_out" if opt_out else ("grievance_hold" if grievance_hold else "suppression")
            blocked_reasons["global"] = reason
            eligibility_snapshots.append({
                "patient_id": pid,
                "snapshot_date": snap["snapshot_date"],
                "action_mask": action_mask,
                "global_constraints": global_constraints,
                "blocked_reasons": blocked_reasons,
                "eligible_measures": [],
            })
            continue

        # If at contact limit, only no_action
        if contacts_this_week >= MAX_CONTACTS_PER_WEEK:
            blocked_reasons["global"] = "contact_limit_reached"
            eligibility_snapshots.append({
                "patient_id": pid,
                "snapshot_date": snap["snapshot_date"],
                "action_mask": action_mask,
                "global_constraints": global_constraints,
                "blocked_reasons": blocked_reasons,
                "eligible_measures": [],
            })
            continue

        eligible_measures = []

        for action in ACTION_CATALOG[1:]:  # Skip no_action
            aid = action.action_id
            measure = action.measure
            channel = action.channel

            # Must have an open gap for this measure
            if measure not in open_gaps:
                blocked_reasons[str(aid)] = "gap_not_open"
                continue

            # Channel availability
            if channel == "sms" and not engagement.get("sms_consent", False):
                blocked_reasons[str(aid)] = "no_sms_consent"
                continue
            if channel == "email" and not engagement.get("email_available", False):
                blocked_reasons[str(aid)] = "no_email"
                continue
            if channel == "portal" and not engagement.get("portal_registered", False):
                blocked_reasons[str(aid)] = "not_portal_registered"
                continue
            if channel == "app" and not engagement.get("app_installed", False):
                blocked_reasons[str(aid)] = "no_app"
                continue

            # Random per-action suppression (e.g., recently sent same measure)
            if rng.random() < 0.05:
                blocked_reasons[str(aid)] = "recent_same_measure_contact"
                continue

            action_mask[aid] = True
            if measure not in eligible_measures:
                eligible_measures.append(measure)

        eligibility_snapshots.append({
            "patient_id": pid,
            "snapshot_date": snap["snapshot_date"],
            "action_mask": action_mask,
            "global_constraints": global_constraints,
            "blocked_reasons": blocked_reasons,
            "eligible_measures": eligible_measures,
        })

    return eligibility_snapshots
