"""
Action masking logic for the HEDIS environment.
Computes which actions are valid for a patient at any given time.
"""
import numpy as np
from typing import Dict, Any, Optional, Set

from config import (
    ACTION_CATALOG, NUM_ACTIONS,
    MAX_CONTACTS_PER_WEEK, MIN_DAYS_BETWEEN_SAME_MEASURE,
)


def compute_action_mask(
    open_gaps: Set[str],
    channel_availability: Dict[str, bool],
    contacts_this_week: int = 0,
    recent_measures: Optional[Dict[str, int]] = None,
    suppressed: bool = False,
    opt_out: bool = False,
    grievance_hold: bool = False,
    budget_remaining: int = None,
    days_since_last_email: int = 999,
) -> np.ndarray:
    """Compute the action mask for a patient.

    Args:
        open_gaps: Set of HEDIS measure codes with open gaps.
        channel_availability: Dict mapping channel -> bool (available).
        contacts_this_week: Number of contacts already sent this week.
        recent_measures: Dict mapping measure -> days since last contact for that measure.
        suppressed: Whether patient is under general suppression.
        opt_out: Whether patient has opted out of all communications.
        grievance_hold: Whether patient has an active grievance hold.
        budget_remaining: Global message budget remaining. 0 = fully exhausted.
        days_since_last_email: Days since last email was sent to this patient. Email suppressed if < 30.

    Returns:
        Boolean numpy array of shape (NUM_ACTIONS,). True = action is valid.
    """
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[0] = True  # no_action always valid

    # Global blocks
    if opt_out or grievance_hold or suppressed:
        return mask

    # Global budget exhausted — no more messages for anyone
    if budget_remaining is not None and budget_remaining <= 0:
        return mask

    # Contact limit
    if contacts_this_week >= MAX_CONTACTS_PER_WEEK:
        return mask

    if recent_measures is None:
        recent_measures = {}

    for action in ACTION_CATALOG[1:]:  # Skip no_action
        measure = action.measure
        channel = action.channel

        # Must have open gap
        if measure not in open_gaps:
            continue

        # Channel must be available
        if not channel_availability.get(channel, False):
            continue

        # Email suppression: max 1 email per 30 days
        from config import MIN_DAYS_BETWEEN_EMAIL
        if channel == "email" and days_since_last_email < MIN_DAYS_BETWEEN_EMAIL:
            continue

        # Cooldown per measure
        days_since = recent_measures.get(measure, MIN_DAYS_BETWEEN_SAME_MEASURE + 1)
        if days_since < MIN_DAYS_BETWEEN_SAME_MEASURE:
            continue

        mask[action.action_id] = True

    return mask


def mask_from_eligibility_snapshot(eligibility: Dict[str, Any]) -> np.ndarray:
    """Extract the action mask directly from a pre-computed eligibility snapshot."""
    raw_mask = eligibility.get("action_mask", [False] * NUM_ACTIONS)
    return np.array(raw_mask, dtype=bool)
