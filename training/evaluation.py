"""
Champion vs challenger model evaluation.
Runs both models on the same simulated episodes and compares performance.
Captures detailed simulation rollout data for dashboard visualization.
"""
import numpy as np
from typing import Dict, Any, List, Tuple
from collections import Counter, defaultdict
from scipy import stats

from config import NUM_ACTIONS, ACTION_BY_ID, HEDIS_MEASURES, MEASURE_DESCRIPTIONS
from environment.hedis_env import HEDISEnv


def evaluate_agent(
    agent,
    env: HEDISEnv,
    n_episodes: int = 500,
    seed: int = 42,
) -> Dict[str, float]:
    """Evaluate an agent on the environment.

    Args:
        agent: Policy with get_action(obs, mask) or get_action_greedy(obs, mask) method.
        env: HEDISEnv instance.
        n_episodes: Number of episodes to evaluate.
        seed: Random seed for reproducibility.

    Returns:
        Dict with evaluation metrics.
    """
    episode_rewards = []
    episode_lengths = []
    gaps_closed = []
    actions_taken = []

    for ep_idx in range(n_episodes):
        obs, info = env.reset(seed=seed + ep_idx, options={"patient_idx": ep_idx % len(env.patient_snapshots)})
        total_reward = 0.0
        length = 0
        initial_gaps = len(info["open_gaps"])
        ep_actions = []

        done = False
        while not done:
            state = obs["observations"]
            mask = obs["action_mask"]

            if hasattr(agent, "get_action_greedy"):
                action = agent.get_action_greedy(state, mask)
            elif hasattr(agent, "get_action"):
                action = agent.get_action(state, mask)
            else:
                # Fallback: random valid action
                valid = np.where(mask)[0]
                action = np.random.choice(valid) if len(valid) > 0 else 0

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            length += 1
            ep_actions.append(action)
            done = terminated or truncated

        final_gaps = len(info["open_gaps"])
        gaps_closed.append(initial_gaps - final_gaps)
        episode_rewards.append(total_reward)
        episode_lengths.append(length)
        actions_taken.append(ep_actions)

    # Compute metrics
    rewards_arr = np.array(episode_rewards)
    gaps_arr = np.array(gaps_closed)
    no_action_rate = np.mean([
        sum(1 for a in ep if a == 0) / max(len(ep), 1) for ep in actions_taken
    ])

    return {
        "mean_reward": float(rewards_arr.mean()),
        "std_reward": float(rewards_arr.std()),
        "median_reward": float(np.median(rewards_arr)),
        "mean_gaps_closed": float(gaps_arr.mean()),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "no_action_rate": float(no_action_rate),
        "n_episodes": n_episodes,
    }


def compare_models(
    champion_metrics: Dict[str, float],
    challenger_metrics: Dict[str, float],
    improvement_threshold: float = 0.05,
) -> Dict[str, Any]:
    """Compare champion and challenger evaluation results.

    Args:
        champion_metrics: Evaluation metrics for champion.
        challenger_metrics: Evaluation metrics for challenger.
        improvement_threshold: Minimum relative improvement to promote.

    Returns:
        Dict with comparison results.
    """
    champ_reward = champion_metrics["mean_reward"]
    chall_reward = challenger_metrics["mean_reward"]

    if champ_reward == 0:
        relative_improvement = float("inf") if chall_reward > 0 else 0.0
    else:
        relative_improvement = (chall_reward - champ_reward) / abs(champ_reward)

    promote = relative_improvement > improvement_threshold

    return {
        "champion_mean_reward": champ_reward,
        "challenger_mean_reward": chall_reward,
        "relative_improvement": relative_improvement,
        "promote_challenger": promote,
        "champion_gaps_closed": champion_metrics["mean_gaps_closed"],
        "challenger_gaps_closed": challenger_metrics["mean_gaps_closed"],
    }


def evaluate_agent_detailed(
    agent,
    env_or_snapshots,
    n_episodes: int = 1000,
    seed: int = 42,
    eligibility_snapshots=None,
) -> Dict[str, Any]:
    """Simulate a full 90-day quarter using WorldSimulator (ground truth).

    Runs the agent across a cohort of patients for 90 simulated days with
    shared global budget, weekly suppression, archetype-driven closures,
    and lagged rewards. Tracks STARS trajectory as it evolves.

    This is the "gold standard" evaluation — what actually happens when
    we deploy this model for a full quarter.
    """
    from environment.reward import compute_stars_score
    from simulation.world import WorldSimulator

    # Accept either HEDISEnv or raw snapshots
    if hasattr(env_or_snapshots, 'patient_snapshots'):
        patient_snapshots = env_or_snapshots.patient_snapshots
        elig = eligibility_snapshots or [{"patient_id": s["patient_id"], "action_mask": [True] * NUM_ACTIONS}
                                         for s in patient_snapshots]
    else:
        patient_snapshots = env_or_snapshots
        elig = eligibility_snapshots or []

    # Sample patients for simulation
    rng = np.random.default_rng(seed)
    n_patients = min(n_episodes, len(patient_snapshots))
    sampled = rng.choice(len(patient_snapshots), size=n_patients, replace=False)
    sim_snapshots = [patient_snapshots[i] for i in sampled]
    sim_elig = elig[:n_patients] if elig else []

    # Create a fresh WorldSimulator for this evaluation
    world = WorldSimulator(sim_snapshots, sim_elig, rng=rng)

    sim_days = 90
    measure_attempts = Counter()
    channel_actions = Counter()
    no_action_count = 0
    total_actions = 0
    total_reward = 0.0

    # STARS trajectory: snapshot every 5 days
    stars_trajectory = []

    for day in range(1, sim_days + 1):
        world.day = day
        daily_actions_count = 0

        for pid in world.patients:
            try:
                ctx = world.get_patient_context(pid)
                state_vec = ctx["state_vec"]
                mask = ctx["mask"]

                if hasattr(agent, "get_action_greedy"):
                    action = agent.get_action_greedy(state_vec, mask.astype(np.float32))
                else:
                    valid = np.where(mask)[0]
                    action = int(rng.choice(valid)) if len(valid) > 0 else 0

                outcome = world.execute_action(pid, action)
                total_actions += 1

                if action == 0:
                    no_action_count += 1
                else:
                    act = ACTION_BY_ID.get(action)
                    if act:
                        measure_attempts[act.measure] += 1
                        channel_actions[act.channel] += 1
                    daily_actions_count += 1
            except Exception:
                continue

        # Advance day (process lagged rewards, state machine, rolling windows)
        day_summary = world.advance_day()
        closure_reward = day_summary.get("closure_reward", 0)
        total_reward += closure_reward

        # Snapshot STARS every 5 days
        if day % 5 == 0 or day == sim_days:
            gap_closures = day_summary.get("gap_closures", {})
            total_patients = day_summary.get("total_patients", {})

            # Compute cumulative closure rates from the world's metrics tracker
            # Use the world's own gap closure accumulation
            closure_rates = {}
            for m in HEDIS_MEASURES:
                # Denominator: initial patients with this gap
                denom = sum(1 for ps in world.patients.values() if m in ps.snapshot.get("open_gaps", []) or m in ps.snapshot.get("closed_gaps", []))
                # Numerator: patients who have responded (closure day set)
                closed = sum(1 for ps in world.patients.values() if ps.last_closure_day > 0 and m in ps.snapshot.get("open_gaps", []))
                closure_rates[m] = closed / max(denom, 1)

            stars = compute_stars_score(closure_rates)
            stars_trajectory.append({
                "day": day,
                "stars": stars,
                "avg_closure": sum(closure_rates.values()) / max(len(closure_rates), 1),
                "daily_actions": daily_actions_count,
                "daily_closures": sum(gap_closures.values()),
                "budget_remaining": world.budget_remaining,
            })

    # Final closure rates
    sim_closure_rates = {}
    for m in HEDIS_MEASURES:
        denom = sum(1 for ps in world.patients.values() if m in ps.snapshot.get("open_gaps", []) or m in ps.snapshot.get("closed_gaps", []))
        closed = sum(1 for ps in world.patients.values() if ps.last_closure_day > 0 and m in ps.snapshot.get("open_gaps", []))
        sim_closure_rates[m] = closed / max(denom, 1)

    sim_channel_rates = {}
    for ch in ["sms", "email", "portal", "app", "ivr"]:
        ch_acts = channel_actions.get(ch, 0)
        sim_channel_rates[ch] = ch_acts / max(total_actions - no_action_count, 1)

    action_dist_by_measure = dict(measure_attempts)
    action_dist_by_channel = dict(channel_actions)

    final_stars = compute_stars_score(sim_closure_rates)

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "total_actions": total_actions,
        "no_action_count": no_action_count,
        "no_action_rate": no_action_count / max(total_actions, 1),
        "sim_closure_rates": sim_closure_rates,
        "sim_channel_rates": sim_channel_rates,
        "action_dist_by_measure": action_dist_by_measure,
        "action_dist_by_channel": action_dist_by_channel,
        "n_episodes": n_patients,
        "sim_days": sim_days,
        "final_stars": final_stars,
        "stars_trajectory": stars_trajectory,
    }
