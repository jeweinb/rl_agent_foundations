"""
Daily simulation cycle.
The agent interacts with the world simulator to select actions for all patients.
All business rules, suppression logic, and outcome generation live in world.py.
"""
import json
import os
import numpy as np
from typing import Dict, Any, List

import config as cfg
from config import NUM_ACTIONS, HEDIS_MEASURES, ACTION_BY_ID
from simulation.world import WorldSimulator
from simulation.logger import get_logger


def run_daily_cycle(
    day: int,
    agent,
    world: WorldSimulator,
    rng: np.random.Generator = None,
) -> Dict[str, Any]:
    """Run one simulated day.

    Args:
        day: Simulation day number.
        agent: Trained policy with get_action_greedy(obs, mask) method.
        world: WorldSimulator instance (owns all patient state + business rules).
        rng: Random number generator.

    Returns:
        Dict with daily results.
    """
    if rng is None:
        rng = np.random.default_rng()

    world.day = day
    daily_actions = []
    daily_rewards = []
    daily_experiences = []
    action_counts: Dict[str, int] = {}

    log = get_logger()

    for pid in world.patients:
        try:
            # 1. Get patient context from world (state vector + mask + metadata)
            ctx = world.get_patient_context(pid)
            state_vec = ctx["state_vec"]
            mask = ctx["mask"]

            # 2. Agent selects action
            if hasattr(agent, "get_action_greedy"):
                action_id = agent.get_action_greedy(state_vec, mask.astype(np.float32))
            elif hasattr(agent, "get_action"):
                action_id = agent.get_action(state_vec, mask.astype(np.float32))
            else:
                valid = np.where(mask)[0]
                action_id = int(rng.choice(valid)) if len(valid) > 0 else 0

            # 3. Execute action in the world (handles all business rules)
            outcome = world.execute_action(pid, action_id)

            daily_rewards.append(outcome["reward"])
            daily_actions.append({
                "patient_id": pid,
                "day": day,
                **outcome,
            })
        except Exception as e:
            log.error(f"Error processing patient {pid} on day {day}: {e}")
            # Default to no-action for this patient
            daily_rewards.append(0.0)
            daily_actions.append({
                "patient_id": pid, "day": day, "action_id": 0,
                "measure": None, "channel": None, "variant": None,
                "reward": 0.0, "is_no_action": True, "engagement": {},
            })
            continue

        # Track for retraining
        daily_experiences.append({
            "obs": state_vec.tolist(),
            "action": action_id,
            "reward": outcome["reward"],
            "mask": mask.tolist(),
            "patient_id": pid,
        })

        if not outcome["is_no_action"]:
            key = f"{outcome['measure']}_{outcome['channel']}"
            action_counts[key] = action_counts.get(key, 0) + 1

    # 4. Advance day in world (state machine, lagged rewards, rolling windows)
    day_summary = world.advance_day()

    # Update engagement signals for today's actions after state machine advance
    for action_record in daily_actions:
        if action_record["is_no_action"]:
            continue
        pid = action_record["patient_id"]
        aid = action_record["action_id"]
        tracking_id = f"day{day:02d}_{pid}_{aid}"
        updated = world.state_machine.get_engagement_signals(tracking_id)
        if updated.get("delivered") or updated.get("opened"):
            action_record["engagement"] = updated

    # 5. Save daily data
    day_dir = os.path.join(cfg.SIMULATION_DATA_DIR, f"day_{day:02d}")
    os.makedirs(day_dir, exist_ok=True)

    with open(os.path.join(day_dir, "actions_taken.json"), "w") as f:
        json.dump(daily_actions, f, indent=2, default=str)

    with open(os.path.join(day_dir, "experience_buffer.json"), "w") as f:
        json.dump(daily_experiences, f, default=str)

    # Save cumulative state machine
    all_sm_records = world.state_machine.to_records()
    with open(os.path.join(day_dir, "state_machine.json"), "w") as f:
        json.dump(all_sm_records, f, indent=2, default=str)
    cumulative_sm_path = os.path.join(cfg.SIMULATION_DATA_DIR, "state_machine_cumulative.json")
    with open(cumulative_sm_path, "w") as f:
        json.dump(all_sm_records, f, default=str)

    return {
        "day": day,
        "total_reward": sum(daily_rewards),
        "num_actions": sum(1 for a in daily_actions if not a["is_no_action"]),
        "action_distribution": action_counts,
        **day_summary,
    }
