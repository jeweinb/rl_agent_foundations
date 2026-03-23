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


def compute_stars_score(
    measure_closure_rates: dict,
    measure_weights: dict = None,
) -> float:
    """Compute an approximate STARS score from measure-level gap closure rates.

    This is a simplified version of the CMS STARS calculation.
    Maps weighted average closure rate to a 1-5 star scale.

    Args:
        measure_closure_rates: Dict mapping measure -> closure rate (0-1).
        measure_weights: Dict mapping measure -> weight. Defaults to MEASURE_WEIGHTS.

    Returns:
        Approximate STARS score (1.0-5.0).
    """
    if measure_weights is None:
        measure_weights = MEASURE_WEIGHTS

    total_weight = 0.0
    weighted_sum = 0.0

    for measure, rate in measure_closure_rates.items():
        w = measure_weights.get(measure, 1.0)
        weighted_sum += rate * w
        total_weight += w

    if total_weight == 0:
        return 1.0

    weighted_avg = weighted_sum / total_weight

    # Map weighted average to star scale
    # Approximate CMS cut points (simplified):
    # <0.40 = 1 star, 0.40-0.55 = 2, 0.55-0.68 = 3, 0.68-0.80 = 4, >0.80 = 5
    if weighted_avg >= 0.80:
        return 4.5 + (weighted_avg - 0.80) / 0.20 * 0.5  # 4.5-5.0
    elif weighted_avg >= 0.68:
        return 3.5 + (weighted_avg - 0.68) / 0.12 * 1.0  # 3.5-4.5
    elif weighted_avg >= 0.55:
        return 2.5 + (weighted_avg - 0.55) / 0.13 * 1.0  # 2.5-3.5
    elif weighted_avg >= 0.40:
        return 1.5 + (weighted_avg - 0.40) / 0.15 * 1.0  # 1.5-2.5
    else:
        return 1.0 + weighted_avg / 0.40 * 0.5  # 1.0-1.5
