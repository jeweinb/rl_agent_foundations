"""
Composite reward function for the HEDIS environment.
Combines gap closure, engagement signals, action costs, and fatigue penalties.
"""
import numpy as np
from typing import Optional

from config import (
    REWARD_WEIGHTS, MEASURE_WEIGHTS,
    BUDGET_WARNING_THRESHOLD, BUDGET_CRITICAL_THRESHOLD,
    MESSAGE_BUDGET_PER_QUARTER,
)


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
    """Compute the composite reward for a single action.

    The reward function encodes a dual objective:
    1. Close HEDIS gaps to achieve STARS ≥ 4.0
    2. Conserve message budget — every message has opportunity cost

    When budget is low, the agent is rewarded for choosing no_action (strategic
    silence) and penalized for sending low-value messages. This teaches the
    agent to wait for high-impact moments rather than spamming patients.

    Args:
        measure: HEDIS measure code, or None for no_action.
        delivered: Whether the message was successfully delivered.
        opened: Whether the patient opened/answered.
        clicked: Whether the patient clicked/engaged.
        gap_closed: Whether the gap was closed (from learned reward model or actual).
        contacts_this_week: Number of contacts already this week.
        days_since_same_measure: Days since last contact for same measure.
        is_no_action: Whether this was the no_action action.
        budget_remaining: Messages remaining in patient's budget period.
        budget_max: Max budget for the period.

    Returns:
        Scalar reward value.
    """
    if budget_max is None:
        budget_max = MESSAGE_BUDGET_PER_QUARTER
    if budget_remaining is None:
        budget_remaining = budget_max

    budget_frac = budget_remaining / max(budget_max, 1)
    w = REWARD_WEIGHTS

    # --- No-action reward: strategic silence ---
    if is_no_action:
        # Reward the agent for conserving budget when it's getting low
        if budget_frac < BUDGET_WARNING_THRESHOLD:
            return w["budget_conservation"]
        return 0.0

    reward = 0.0

    # --- Gap closure reward (weighted by STARS importance) ---
    if gap_closed and measure:
        measure_weight = MEASURE_WEIGHTS.get(measure, 1.0)
        reward += w["gap_closure"] * measure_weight

    # --- Engagement signals (immediate) ---
    if delivered:
        reward += w["engagement_deliver"]
    if clicked:
        reward += w["engagement_click"]

    # --- Action cost (every message costs something) ---
    reward += w["action_cost"]

    # --- Contact fatigue penalty ---
    if contacts_this_week >= 2:
        reward += w["fatigue"] * (contacts_this_week - 1)
    if days_since_same_measure < 7:
        reward += w["fatigue"]

    # --- Budget-aware penalties ---
    # When budget is running low, penalize messages that don't produce engagement
    if budget_frac < BUDGET_CRITICAL_THRESHOLD:
        # Critical: heavy penalty for any message (save remaining budget for emergencies)
        reward += w["budget_critical_penalty"]
    elif budget_frac < BUDGET_WARNING_THRESHOLD:
        # Warning: penalize messages that don't get clicks (wasteful)
        if not clicked:
            reward += w["budget_waste"]

    return reward


def measure_rate_to_stars(measure: str, rate: float) -> float:
    """Convert a single measure's performance rate to its individual star rating.

    Uses CMS cut points: each measure has specific thresholds for 2-5 stars.
    Below the 2-star cut point = 1 star.

    Args:
        measure: HEDIS measure code.
        rate: Performance rate (0-1).

    Returns:
        Individual measure star rating (1.0-5.0).
    """
    from config import MEASURE_CUT_POINTS
    cuts = MEASURE_CUT_POINTS.get(measure)
    if not cuts:
        # Fallback for unknown measures
        if rate >= 0.80: return 5.0
        if rate >= 0.65: return 4.0
        if rate >= 0.50: return 3.0
        if rate >= 0.35: return 2.0
        return 1.0

    if rate >= cuts[5]:
        return 5.0
    elif rate >= cuts[4]:
        # Interpolate between 4 and 5
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

    CMS methodology (2-step process):
    1. Each measure's performance rate is converted to an individual 1-5 star
       rating using measure-specific cut points.
    2. The overall rating is the weighted average of individual measure stars.

    Args:
        measure_closure_rates: Dict mapping measure -> performance rate (0-1).
        measure_weights: Dict mapping measure -> weight. Defaults to MEASURE_WEIGHTS.

    Returns:
        Overall STARS score (1.0-5.0).
    """
    if measure_weights is None:
        measure_weights = MEASURE_WEIGHTS

    total_weight = 0.0
    weighted_star_sum = 0.0

    for measure, rate in measure_closure_rates.items():
        w = measure_weights.get(measure, 1.0)
        individual_stars = measure_rate_to_stars(measure, rate)
        weighted_star_sum += individual_stars * w
        total_weight += w

    if total_weight == 0:
        return 1.0

    return weighted_star_sum / total_weight


def get_measure_stars_detail(measure_closure_rates: dict) -> dict:
    """Get individual star ratings and 4-star thresholds for each measure.

    Returns dict mapping measure -> {rate, stars, threshold_4star, at_or_above_4}.
    Used by the dashboard to show per-measure performance vs target.
    """
    from config import MEASURE_CUT_POINTS, MEASURE_WEIGHTS
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
