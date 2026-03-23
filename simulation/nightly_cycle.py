"""
Nightly simulation cycle.
Retrains a challenger model on all accumulated data,
evaluates champion vs challenger, and promotes if better.
"""
import json
import os
import torch
import numpy as np
from typing import Dict, Any, List, Optional

from config import (
    SIMULATION_DATA_DIR, CHECKPOINTS_DIR, STATE_DIM, NUM_ACTIONS,
    GENERATED_DATA_DIR,
)
from training.data_loader import load_datasets, build_offline_episodes
from training.cql_trainer import train_cql, ActorCriticCQL
from training.evaluation import evaluate_agent, compare_models
from environment.hedis_env import HEDISEnv


def _load_simulation_experiences(up_to_day: int) -> List[Dict]:
    """Load all simulation experience buffers up to given day."""
    all_experiences = []
    for day in range(1, up_to_day + 1):
        exp_path = os.path.join(SIMULATION_DATA_DIR, f"day_{day:02d}", "experience_buffer.json")
        if os.path.exists(exp_path):
            with open(exp_path) as f:
                experiences = json.load(f)
                all_experiences.extend(experiences)
    return all_experiences


def _experiences_to_episodes(experiences: List[Dict]) -> List[Dict]:
    """Convert flat experiences to episode format for CQL training."""
    # Group by patient
    by_patient: Dict[str, List] = {}
    for exp in experiences:
        pid = exp.get("patient_id", "unknown")
        by_patient.setdefault(pid, []).append(exp)

    episodes = []
    for pid, exps in by_patient.items():
        if len(exps) < 2:
            continue
        obs = np.array([e["obs"] for e in exps], dtype=np.float32)
        actions = np.array([e["action"] for e in exps], dtype=np.int64)
        rewards = np.array([e["reward"] for e in exps], dtype=np.float32)
        masks = np.array([e["mask"] for e in exps], dtype=np.float32)
        T = len(exps)
        episodes.append({
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "action_mask": masks,
            "terminateds": np.zeros(T, dtype=np.float32),
            "truncateds": np.array([0.0] * (T - 1) + [1.0], dtype=np.float32),
        })
    return episodes


def run_nightly_cycle(
    day: int,
    champion: ActorCriticCQL,
    patient_snapshots: list,
    eligibility_snapshots: list,
    cql_epochs: int = 30,
    eval_episodes: int = 200,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run nightly retraining and evaluation.

    Args:
        day: Current simulation day.
        champion: Current champion Q-network.
        patient_snapshots: Patient state snapshots.
        eligibility_snapshots: Eligibility snapshots.
        cql_epochs: CQL training epochs for challenger.
        eval_episodes: Episodes for evaluation.
        verbose: Print progress.

    Returns:
        Dict with nightly results: promoted, scores, etc.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Night {day}: Retraining and Evaluation")
        print(f"{'='*60}")

    # Load historical + simulation data
    datasets = load_datasets()
    historical_episodes = build_offline_episodes(
        datasets["state_features"],
        datasets["historical_activity"],
        datasets["action_eligibility"],
    )

    # Add simulation experiences
    sim_experiences = _load_simulation_experiences(day)
    sim_episodes = _experiences_to_episodes(sim_experiences)

    all_episodes = historical_episodes + sim_episodes
    if verbose:
        print(f"  Training data: {len(historical_episodes)} historical + {len(sim_episodes)} simulation episodes")

    # Train challenger
    if verbose:
        print("  Training challenger CQL model...")
    challenger = train_cql(
        episodes=all_episodes,
        epochs=cql_epochs,
        verbose=verbose,
    )

    # Evaluate both models
    env = HEDISEnv(patient_snapshots, eligibility_snapshots)

    if verbose:
        print(f"  Evaluating champion ({eval_episodes} episodes)...")
    champion_metrics = evaluate_agent(champion, env, n_episodes=eval_episodes, seed=day * 1000)

    if verbose:
        print(f"  Evaluating challenger ({eval_episodes} episodes)...")
    challenger_metrics = evaluate_agent(challenger, env, n_episodes=eval_episodes, seed=day * 1000)

    # Compare
    comparison = compare_models(champion_metrics, challenger_metrics)

    if verbose:
        print(f"\n  Champion reward: {comparison['champion_mean_reward']:.4f}")
        print(f"  Challenger reward: {comparison['challenger_mean_reward']:.4f}")
        print(f"  Relative improvement: {comparison['relative_improvement']:.4f}")
        print(f"  Promote: {comparison['promote_challenger']}")

    # Save challenger checkpoint
    challenger_path = os.path.join(CHECKPOINTS_DIR, f"challenger_day_{day:02d}.pt")
    torch.save(challenger.state_dict(), challenger_path)

    # Promote if better
    promoted = comparison["promote_challenger"]
    if promoted:
        champion_path = os.path.join(CHECKPOINTS_DIR, "champion.pt")
        torch.save(challenger.state_dict(), champion_path)
        if verbose:
            print(f"  Challenger PROMOTED to champion! Saved to {champion_path}")
        new_champion = challenger
    else:
        new_champion = champion
        if verbose:
            print("  Champion RETAINED.")

    # Save nightly metrics
    day_dir = os.path.join(SIMULATION_DATA_DIR, f"day_{day:02d}")
    os.makedirs(day_dir, exist_ok=True)
    nightly_metrics = {
        "day": day,
        "champion_metrics": champion_metrics,
        "challenger_metrics": challenger_metrics,
        "comparison": comparison,
        "promoted": promoted,
    }
    with open(os.path.join(day_dir, "nightly_metrics.json"), "w") as f:
        json.dump(nightly_metrics, f, indent=2, default=str)

    return {
        "new_champion": new_champion,
        "promoted": promoted,
        "champion_score": champion_metrics["mean_reward"],
        "challenger_score": challenger_metrics["mean_reward"],
        "comparison": comparison,
    }
