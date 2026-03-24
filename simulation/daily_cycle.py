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
    global_budget: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Run one simulated day of the RL agent interacting with patients.

    Args:
        day: Simulation day number.
        agent: Trained policy with get_action_greedy(obs, mask) method.
        patient_snapshots: Current state of all patients.
        eligibility_snapshots: Current eligibility for all patients.
        state_machine: Action lifecycle tracker.
        lagged_queue: Delayed reward queue.
        rng: Random number generator.
        global_budget: Shared budget dict with keys: remaining, max, patient_stats.
    """
    if rng is None:
        rng = np.random.default_rng()

    from config import compute_global_budget, MAX_CONTACTS_PER_WEEK

    # Initialize global budget if not provided
    if global_budget is None:
        total = compute_global_budget(len(patient_snapshots))
        global_budget = {
            "remaining": total,
            "max": total,
            "total_sent": 0,
            "patient_stats": {},  # per-patient: messages_sent, channels_used, responses, last_closure_day
        }

    # Initialize per-patient stats for any new patients
    for snap in patient_snapshots:
        pid = snap["patient_id"]
        if pid not in global_budget["patient_stats"]:
            global_budget["patient_stats"][pid] = {
                "messages_sent": 0,
                "contacts_this_week": 0,
                "week_start_day": 1,
                "channels_used": set(),
                "responses": 0,     # clicks/accepts
                "last_closure_day": 0,
            }
        ps = global_budget["patient_stats"][pid]
        # Weekly contact counter reset
        if day - ps.get("week_start_day", 1) >= 7:
            ps["contacts_this_week"] = 0
            ps["week_start_day"] = day

    # Compute cohort-level stats for state features
    all_patient_msgs = [ps["messages_sent"] for ps in global_budget["patient_stats"].values()]
    cohort_avg_messages = np.mean(all_patient_msgs) if all_patient_msgs else 0.0

    eligibility_map = {e["patient_id"]: e for e in eligibility_snapshots}

    daily_actions = []
    daily_rewards = []
    daily_experiences = []
    gap_closures = {m: 0 for m in HEDIS_MEASURES}
    total_patients = {m: 0 for m in HEDIS_MEASURES}
    action_counts: Dict[str, int] = {}

    for snap in patient_snapshots:
        pid = snap["patient_id"]
        open_gaps = set(snap.get("open_gaps", []))

        # Count patients per measure
        for m in open_gaps:
            total_patients[m] = total_patients.get(m, 0) + 1

        # Check if patient has pending actions (from state machine)
        pending = state_machine.get_pending_actions(pid)
        pending_measures = {p["measure"] for p in pending}

        # Per-patient stats (live features updated each day)
        ps = global_budget["patient_stats"][pid]

        # Build state vector with LIVE features
        state_vec = snapshot_to_vector(
            snap, day_of_year=day * 4,  # ~4 days per sim day to spread across year
            budget_remaining=global_budget["remaining"],
            budget_max=global_budget["max"],
            patient_messages_received=ps["messages_sent"],
            cohort_avg_messages=cohort_avg_messages,
            patient_response_rate=ps["responses"] / max(ps["messages_sent"], 1),
            patient_avg_gap_age=day * 4.0,  # Approximate gap age
            patient_days_since_closure=max(0, day - ps.get("last_closure_day", 0)) if ps.get("last_closure_day", 0) > 0 else 90.0,
            patient_channels_used=len(ps.get("channels_used", set())),
        )

        # Compute action mask (global budget-aware)
        engagement = snap.get("engagement", {})
        channel_avail = {
            "sms": engagement.get("sms_consent", False),
            "email": engagement.get("email_available", False),
            "portal": engagement.get("portal_registered", False),
            "app": engagement.get("app_installed", False),
            "ivr": True,
        }

        recent_measures = {m: 0 for m in pending_measures}

        mask = compute_action_mask(
            open_gaps=open_gaps,
            channel_availability=channel_avail,
            contacts_this_week=ps.get("contacts_this_week", 0),
            recent_measures=recent_measures,
            budget_remaining=global_budget["remaining"],
        )

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
            # Mirrors the same patterns as historical data:
            # best channel → higher closure, engagement → higher closure
            from datagen.constants import GAP_CLOSURE_BASE_RATES
            from config import (
                get_measure_category, BEST_CHANNEL_BY_CATEGORY,
                CLOSURE_BASE_MULTIPLIER, CLOSURE_BEST_CHANNEL_FACTOR,
                CLOSURE_CLICKED_FACTOR, CLOSURE_OPENED_FACTOR,
                CLOSURE_DELIVERED_FACTOR,
            )
            base_rate = GAP_CLOSURE_BASE_RATES.get(action_info.measure, 0.5)
            category = get_measure_category(action_info.measure)
            best_ch = BEST_CHANNEL_BY_CATEGORY.get(category, "sms")

            ch_factor = CLOSURE_BEST_CHANNEL_FACTOR if action_info.channel == best_ch else 1.0
            closure_prob = base_rate * CLOSURE_BASE_MULTIPLIER * ch_factor
            if signals.get("clicked"):
                closure_prob *= CLOSURE_CLICKED_FACTOR
            elif signals.get("opened"):
                closure_prob *= CLOSURE_OPENED_FACTOR
            elif signals.get("delivered"):
                closure_prob *= CLOSURE_DELIVERED_FACTOR

            lagged_queue.schedule(
                current_day=day,
                patient_id=pid,
                measure=action_info.measure,
                action_id=action_id,
                closure_prob=min(closure_prob, cfg.CLOSURE_PROB_CAP),
            )

            # Compute immediate reward (global budget-aware)
            reward = compute_reward(
                measure=action_info.measure,
                delivered=signals["delivered"],
                opened=signals["opened"],
                clicked=signals["clicked"],
                gap_closed=False,
                is_no_action=False,
                budget_remaining=global_budget["remaining"],
                budget_max=global_budget["max"],
            )

            # Decrement GLOBAL budget + update per-patient stats
            global_budget["remaining"] = max(0, global_budget["remaining"] - 1)
            global_budget["total_sent"] += 1
            ps["messages_sent"] += 1
            ps["contacts_this_week"] = ps.get("contacts_this_week", 0) + 1
            if isinstance(ps.get("channels_used"), set):
                ps["channels_used"].add(action_info.channel)
            else:
                ps["channels_used"] = {action_info.channel}

            # Track
            act_key = f"{action_info.measure}_{action_info.channel}"
            action_counts[act_key] = action_counts.get(act_key, 0) + 1
        else:
            # No action taken — compute budget conservation reward
            reward = compute_reward(
                measure=None, is_no_action=True,
                budget_remaining=global_budget["remaining"],
                budget_max=global_budget["max"],
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
            "budget_remaining": global_budget["remaining"],
            "budget_max": global_budget["max"],
            "patient_messages": ps["messages_sent"],
        })

        # Store experience for retraining
        daily_experiences.append({
            "obs": state_vec.tolist(),
            "action": action_id,
            "reward": reward,
            "mask": mask.tolist(),
            "patient_id": pid,
        })

    # Advance all pending actions in state machine (one transition per day)
    state_machine.advance_all(day)

    # Update action records with latest engagement signals from state machine
    # (actions created today were only at QUEUED when signals were first captured;
    #  now after advance_all they may be at PRESENTED/VIEWED/etc.)
    for action_record in daily_actions:
        if action_record["action_id"] == 0:
            continue
        pid = action_record["patient_id"]
        aid = action_record["action_id"]
        tracking_id = f"day{day:02d}_{pid}_{aid}"
        updated_signals = state_machine.get_engagement_signals(tracking_id)
        if updated_signals.get("delivered") or updated_signals.get("opened"):
            action_record["engagement"] = updated_signals

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

    # Save ALL state machine records (cumulative) so Sankey sees full lifecycle
    all_sm_records = state_machine.to_records()
    with open(os.path.join(day_dir, "state_machine.json"), "w") as f:
        json.dump(all_sm_records, f, indent=2, default=str)

    # Also save a cumulative state machine snapshot for the dashboard
    cumulative_sm_path = os.path.join(cfg.SIMULATION_DATA_DIR, "state_machine_cumulative.json")
    with open(cumulative_sm_path, "w") as f:
        json.dump(all_sm_records, f, default=str)

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
        "global_budget_remaining": global_budget["remaining"],
        "global_budget_max": global_budget["max"],
        "global_budget_used": global_budget["total_sent"],
        "global_budget": global_budget,
    }
