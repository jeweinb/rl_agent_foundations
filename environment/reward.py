"""
Reward function for the HEDIS environment.

SIMPLIFIED: Two clear signals only.
1. Gap closure = big positive reward (weighted by STARS importance)
2. Tiny action cost = prevents mindless spam

The model learns to close gaps efficiently. Budget constraint is handled by
action masking (global budget exhaustion), not reward penalties.
"""
import numpy as np
from typing import Optional

from config import REWARD_WEIGHTS, MEASURE_WEIGHTS, MEASURE_CUT_POINTS


def compute_reward(
    measure: Optional[str],
    delivered: bool = False,
    opened: bool = False,
    clicked: bool = False,
    gap_closed: bool = False,
    contacts_this_week: int = 0,
    days_since_same_measure: int = 999,
    is_no_action: bool = False,
    budget_remaining: int = None,
    budget_max: int = None,
) -> float:
    """Compute reward. Simple: gap closure good, spam bad.

    Args:
        measure: HEDIS measure code, or None for no_action.
        clicked: Whether patient engaged (clicked/accepted).
        gap_closed: Whether the gap was closed.
        is_no_action: Whether this was the no_action action.
        Other args kept for API compatibility.

    Returns:
        Scalar reward value.
    """
    if is_no_action:
        return 0.0

    w = REWARD_WEIGHTS
    reward = 0.0

    # Gap closure — the real objective
    if gap_closed and measure:
        measure_weight = MEASURE_WEIGHTS.get(measure, 1)
        reward += w["gap_closure"] * measure_weight

    # Small engagement bonus
    if clicked:
        reward += w.get("engagement_click", 0.05)

    return reward


def measure_rate_to_stars(measure: str, rate: float) -> float:
    """Convert a single measure's performance rate to its individual star rating.

    Uses CMS cut points: each measure has specific thresholds for 2-5 stars.
    Below the 2-star cut point = 1 star.
    """
    cuts = MEASURE_CUT_POINTS.get(measure)
    if not cuts:
        if rate >= 0.80: return 5.0
        if rate >= 0.65: return 4.0
        if rate >= 0.50: return 3.0
        if rate >= 0.35: return 2.0
        return 1.0

    if rate >= cuts[5]:
        return 5.0
    elif rate >= cuts[4]:
        return 4.0 + (rate - cuts[4]) / max(cuts[5] - cuts[4], 0.01)
    elif rate >= cuts[3]:
        return 3.0 + (rate - cuts[3]) / max(cuts[4] - cuts[3], 0.01)
    elif rate >= cuts[2]:
        return 2.0 + (rate - cuts[2]) / max(cuts[3] - cuts[2], 0.01)
    else:
        return 1.0 + min(rate / max(cuts[2], 0.01), 1.0)


def compute_stars_score(
    measure_closure_rates: dict,
    measure_weights: dict = None,
) -> float:
    """Compute the overall STARS score using CMS methodology.

    1. Each measure's rate → individual 1-5 star rating via cut points
    2. Overall = weighted average of individual stars
    """
    if measure_weights is None:
        measure_weights = MEASURE_WEIGHTS

    total_weight = 0.0
    weighted_star_sum = 0.0

    for measure, rate in measure_closure_rates.items():
        w = measure_weights.get(measure, 1)
        individual_stars = measure_rate_to_stars(measure, rate)
        weighted_star_sum += individual_stars * w
        total_weight += w

    if total_weight == 0:
        return 1.0

    return weighted_star_sum / total_weight


def get_measure_stars_detail(measure_closure_rates: dict) -> dict:
    """Get individual star ratings and 4-star thresholds for each measure."""
    detail = {}
    for measure, rate in measure_closure_rates.items():
        cuts = MEASURE_CUT_POINTS.get(measure, {})
        stars = measure_rate_to_stars(measure, rate)
        threshold_4 = cuts.get(4, 0.70)
        detail[measure] = {
            "rate": rate,
            "stars": round(stars, 2),
            "threshold_4star": threshold_4,
            "at_or_above_4": stars >= 4.0,
            "gap_to_4star": max(0, threshold_4 - rate),
            "weight": MEASURE_WEIGHTS.get(measure, 1),
        }
    return detail
