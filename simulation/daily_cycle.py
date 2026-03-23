"""
Daily simulation cycle.
The champion model selects actions for all patients, interacts with the world,
and stores experience data. The state machine tracks action lifecycles.
"""
import json
import os
import numpy as np
from typing import Dict, Any, List

import config as cfg
from config import (
    NUM_ACTIONS, HEDIS_MEASURES,
    ACTION_BY_ID, MEASURE_WEIGHTS, REWARD_WEIGHTS,
)
from environment.state_space import snapshot_to_vector
from environment.action_masking import compute_action_mask
from environment.reward import compute_reward
from simulation.action_state_machine import ActionLifecycleTracker
from simulation.lagged_rewards import LaggedRewardQueue


def run_daily_cycle(
    day: int,
    agent,
    patient_snapshots: List[Dict[str, Any]],
    eligibility_snapshots: List[Dict[str, Any]],
    state_machine: ActionLifecycleTracker,
    lagged_queue: LaggedRewardQueue,
    rng: np.random.Generator = None,
    patient_budgets: Dict[str, Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Run one simulated day of the RL agent interacting with patients.

    Args:
        day: Simulation day number (1-30).
        agent: Trained policy with get_action_greedy(obs, mask) method.
        patient_snapshots: Current state of all patients.
        eligibility_snapshots: Current eligibility for all patients.
        state_machine: Action lifecycle tracker.
        lagged_queue: Delayed reward queue.
        rng: Random number generator.

    Returns:
        Dict with daily results: actions taken, rewards, experiences, etc.
    """
    if rng is None:
        rng = np.random.default_rng()

    from config import MESSAGE_BUDGET_PER_QUARTER, BUDGET_REPLENISH_INTERVAL_DAYS

    # Initialize patient budgets if not provided
    if patient_budgets is None:
        patient_budgets = {}
    for snap in patient_snapshots:
        pid = snap["patient_id"]
        if pid not in patient_budgets:
            patient_budgets[pid] = {
                "remaining": MESSAGE_BUDGET_PER_QUARTER,
                "max": MESSAGE_BUDGET_PER_QUARTER,
                "total_sent": 0,
                "quarter_start_day": 1,
            }
        # Quarterly replenishment
        budget = patient_budgets[pid]
        if day - budget["quarter_start_day"] >= BUDGET_REPLENISH_INTERVAL_DAYS:
            budget["remaining"] = MESSAGE_BUDGET_PER_QUARTER
            budget["quarter_start_day"] = day

    eligibility_map = {e["patient_id"]: e for e in eligibility_snapshots}

    daily_actions = []
    daily_rewards = []
    daily_experiences = []
    gap_closures = {m: 0 for m in HEDIS_MEASURES}
    total_patients = {m: 0 for m in HEDIS_MEASURES}
    action_counts: Dict[str, int] = {}
    budget_exhausted_count = 0

    for snap in patient_snapshots:
        pid = snap["patient_id"]
        open_gaps = set(snap.get("open_gaps", []))

        # Count patients per measure
        for m in open_gaps:
            total_patients[m] = total_patients.get(m, 0) + 1

        # Check if patient has pending actions (from state machine)
        pending = state_machine.get_pending_actions(pid)
        pending_measures = {p["measure"] for p in pending}

        # Get patient budget
        budget = patient_budgets[pid]
        budget_remaining = budget["remaining"]

        # Build state vector with budget info
        state_vec = snapshot_to_vector(
            snap, day_of_year=day * 12,
            budget_remaining=budget_remaining,
            budget_max=budget["max"],
        )

        # Compute action mask (budget-aware)
        engagement = snap.get("engagement", {})
        channel_avail = {
            "sms": engagement.get("sms_consent", False),
            "email": engagement.get("email_available", False),
            "portal": engagement.get("portal_registered", False),
            "app": engagement.get("app_installed", False),
            "ivr": True,
        }

        # Additional masking: block measures with pending actions
        recent_measures = {m: 0 for m in pending_measures}  # Treat as "just contacted"

        mask = compute_action_mask(
            open_gaps=open_gaps,
            channel_availability=channel_avail,
            contacts_this_week=len(pending),
            recent_measures=recent_measures,
            budget_remaining=budget_remaining,
        )

        if budget_remaining <= 0:
            budget_exhausted_count += 1

        # Agent selects action
        if hasattr(agent, "get_action_greedy"):
            action_id = agent.get_action_greedy(state_vec, mask)
        elif hasattr(agent, "get_action"):
            action_id = agent.get_action(state_vec, mask)
        else:
            valid = np.where(mask)[0]
            action_id = int(rng.choice(valid)) if len(valid) > 0 else 0

        action_info = ACTION_BY_ID.get(action_id)
        is_no_act = action_id == 0

        # Track action in state machine
        if not is_no_act and action_info:
            tracking_id = f"day{day:02d}_{pid}_{action_id}"
            state_machine.create_action(
                tracking_id=tracking_id,
                patient_id=pid,
                action_id=action_id,
                measure=action_info.measure,
                channel=action_info.channel,
                variant=action_info.variant,
                day=day,
            )

            # Advance action through initial states (CREATED → QUEUED only)
            # Further transitions happen one-per-day via advance_all()
            state_machine.advance(tracking_id, day)  # → QUEUED

            # Get engagement signals from state machine (will be minimal on creation day)
            signals = state_machine.get_engagement_signals(tracking_id)

            # Schedule potential lagged reward
            # Closure probability is based on measure base rate + channel lift
            # Engagement signals improve probability but action itself has baseline value
            from datagen.constants import GAP_CLOSURE_BASE_RATES, OUTREACH_LIFT
            base_rate = GAP_CLOSURE_BASE_RATES.get(action_info.measure, 0.5)
            lift = OUTREACH_LIFT.get(action_info.channel, 1.0)
            # Per-interaction closure probability (annualized rate / ~30 interactions per year)
            closure_prob = base_rate * lift * 0.08
            if signals.get("clicked"):
                closure_prob *= 3.0
            elif signals.get("opened"):
                closure_prob *= 1.5

            lagged_queue.schedule(
                current_day=day,
                patient_id=pid,
                measure=action_info.measure,
                action_id=action_id,
                closure_prob=min(closure_prob, 0.5),
            )

            # Compute immediate reward (budget-aware)
            reward = compute_reward(
                measure=action_info.measure,
                delivered=signals["delivered"],
                opened=signals["opened"],
                clicked=signals["clicked"],
                gap_closed=False,  # Gap closure is lagged
                is_no_action=False,
                budget_remaining=budget_remaining,
                budget_max=budget["max"],
            )

            # Decrement patient budget
            budget["remaining"] = max(0, budget["remaining"] - 1)
            budget["total_sent"] += 1

            # Track
            act_key = f"{action_info.measure}_{action_info.channel}"
            action_counts[act_key] = action_counts.get(act_key, 0) + 1
        else:
            # No action taken — compute budget conservation reward
            reward = compute_reward(
                measure=None, is_no_action=True,
                budget_remaining=budget_remaining,
                budget_max=budget["max"],
            )
            signals = {"delivered": False, "opened": False, "clicked": False}

        daily_rewards.append(reward)
        daily_actions.append({
            "patient_id": pid,
            "action_id": action_id,
            "measure": action_info.measure if action_info else None,
            "channel": action_info.channel if action_info else None,
            "variant": action_info.variant if action_info else None,
            "reward": reward,
            "day": day,
            "engagement": signals if not is_no_act else {},
            "budget_remaining": budget["remaining"],
            "budget_max": budget["max"],
        })

        # Store experience for retraining
        daily_experiences.append({
            "obs": state_vec.tolist(),
            "action": action_id,
            "reward": reward,
            "mask": mask.tolist(),
            "patient_id": pid,
        })

    # Advance all pending actions in state machine
    state_machine.advance_all(day)

    # Process lagged rewards arriving today
    resolved = lagged_queue.collect(day)
    for r in resolved:
        if r["will_close"]:
            gap_closures[r["measure"]] = gap_closures.get(r["measure"], 0) + 1

    # Save daily data
    day_dir = os.path.join(cfg.SIMULATION_DATA_DIR, f"day_{day:02d}")
    os.makedirs(day_dir, exist_ok=True)

    with open(os.path.join(day_dir, "actions_taken.json"), "w") as f:
        json.dump(daily_actions, f, indent=2, default=str)

    with open(os.path.join(day_dir, "experience_buffer.json"), "w") as f:
        json.dump(daily_experiences, f, default=str)

    with open(os.path.join(day_dir, "state_machine.json"), "w") as f:
        json.dump(state_machine.to_records()[-len(patient_snapshots):], f, indent=2, default=str)

    return {
        "day": day,
        "total_reward": sum(daily_rewards),
        "num_actions": sum(1 for a in daily_actions if a["action_id"] != 0),
        "gap_closures": gap_closures,
        "total_patients": total_patients,
        "action_distribution": action_counts,
        "resolved_rewards": len(resolved),
        "pending_rewards": lagged_queue.get_pending_count(),
        "state_machine_funnel": state_machine.get_funnel_stats(),
        "budget_exhausted_count": budget_exhausted_count,
        "avg_budget_remaining": np.mean([b["remaining"] for b in patient_budgets.values()]),
        "patient_budgets": patient_budgets,
    }
