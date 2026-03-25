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
    env: HEDISEnv,
    n_episodes: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Simulate a full 90-day quarter across the cohort.

    Each patient gets a 90-step episode (one quarter). Tracks STARS trajectory
    over time so we can see gaps closing and STARS climbing toward 4.0.

    n_episodes: number of patients to simulate (use full cohort size for best results)
    """
    from environment.reward import compute_stars_score

    n_patients = min(n_episodes, len(env.patient_snapshots))
    sim_days = 90  # Full quarter

    # Per-patient tracking
    measure_patients_with_gap = Counter()
    measure_patients_closed = Counter()
    measure_attempts = Counter()
    channel_actions = Counter()
    channel_closures = Counter()
    no_action_count = 0
    total_actions = 0
    episode_rewards = []

    # STARS trajectory: snapshot closure rates every 10 days
    stars_trajectory = []

    for ep_idx in range(n_patients):
        # Override max_steps to 90 for full quarter simulation
        env.max_steps = sim_days
        obs, info = env.reset(seed=seed + ep_idx,
                             options={"patient_idx": ep_idx % len(env.patient_snapshots)})
        total_reward = 0.0

        initial_gaps = set(info.get("open_gaps", []))
        for m in initial_gaps:
            measure_patients_with_gap[m] += 1
        closed_this_episode = set()

        done = False
        step = 0
        while not done:
            state = obs["observations"]
            mask = obs["action_mask"]

            if hasattr(agent, "get_action_greedy"):
                action = agent.get_action_greedy(state, mask)
            else:
                valid = np.where(mask)[0]
                action = int(np.random.choice(valid)) if len(valid) > 0 else 0

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_actions += 1
            step += 1

            if action == 0:
                no_action_count += 1
            else:
                act = ACTION_BY_ID.get(action)
                if act:
                    measure_attempts[act.measure] += 1
                    channel_actions[act.channel] += 1
                    if info.get("gap_closed"):
                        closed_this_episode.add(act.measure)
                        channel_closures[act.channel] += 1

            done = terminated or truncated

        for m in closed_this_episode:
            measure_patients_closed[m] += 1
        episode_rewards.append(total_reward)

        # Snapshot STARS every 100 patients
        if (ep_idx + 1) % max(n_patients // 10, 1) == 0:
            snapshot_rates = {}
            for m in HEDIS_MEASURES:
                pw = measure_patients_with_gap.get(m, 0)
                pc = measure_patients_closed.get(m, 0)
                snapshot_rates[m] = pc / max(pw, 1) if pw > 0 else 0.0
            stars_trajectory.append({
                "patients_processed": ep_idx + 1,
                "stars": compute_stars_score(snapshot_rates),
                "avg_closure": sum(snapshot_rates.values()) / max(len(snapshot_rates), 1),
            })

    # Final closure rates
    sim_closure_rates = {}
    for m in HEDIS_MEASURES:
        pw = measure_patients_with_gap.get(m, 0)
        pc = measure_patients_closed.get(m, 0)
        sim_closure_rates[m] = pc / max(pw, 1) if pw > 0 else 0.0

    sim_channel_rates = {}
    for ch in ["sms", "email", "portal", "app", "ivr"]:
        ch_acts = channel_actions.get(ch, 0)
        ch_close = channel_closures.get(ch, 0)
        sim_channel_rates[ch] = ch_close / max(ch_acts, 1) if ch_acts > 0 else 0.0

    action_dist_by_measure = dict(measure_attempts)
    action_dist_by_channel = dict(channel_actions)

    # Final STARS from the full simulation
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
