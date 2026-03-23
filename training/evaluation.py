"""
Champion vs challenger model evaluation.
Runs both models on the same simulated episodes and compares performance.
"""
import numpy as np
from typing import Dict, Any, List, Tuple
from scipy import stats

from config import NUM_ACTIONS
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
    improvement_threshold: float = 0.02,
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
