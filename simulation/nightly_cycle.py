"""
Nightly simulation cycle — Dyna-style online update.

Instead of full retraining from scratch each night, does a quick incremental
update on the champion model using:
1. Today's new experiences (real interactions)
2. A small replay sample from historical data (prevents catastrophic forgetting)
3. A few fast CQL epochs on this combined mini-batch

This mirrors production deployment: incremental nightly model updates, not
cold retraining. Champion/challenger evaluation still runs to gate promotions.
"""
import json
import os
import copy
import torch
import numpy as np
from typing import Dict, Any, List, Optional

from config import (
    SIMULATION_DATA_DIR, CHECKPOINTS_DIR, STATE_DIM, NUM_ACTIONS,
    GENERATED_DATA_DIR,
)
from training.data_loader import load_datasets, build_offline_episodes
from training.cql_trainer import train_cql, ActorCriticCQL
from training.evaluation import evaluate_agent, compare_models, evaluate_agent_detailed
from environment.hedis_env import HEDISEnv


def _load_simulation_experiences(up_to_day: int) -> List[Dict]:
    """Load simulation experience buffers up to given day."""
    all_experiences = []
    for day in range(1, up_to_day + 1):
        exp_path = os.path.join(SIMULATION_DATA_DIR, f"day_{day:02d}", "experience_buffer.json")
        if os.path.exists(exp_path):
            with open(exp_path) as f:
                all_experiences.extend(json.load(f))
    return all_experiences


def _load_today_experiences(day: int) -> List[Dict]:
    """Load only today's experience buffer."""
    exp_path = os.path.join(SIMULATION_DATA_DIR, f"day_{day:02d}", "experience_buffer.json")
    if os.path.exists(exp_path):
        with open(exp_path) as f:
            return json.load(f)
    return []


def _experiences_to_episodes(experiences: List[Dict], rng=None) -> List[Dict]:
    """Convert flat experiences to episode format for CQL training."""
    if rng is None:
        rng = np.random.default_rng()
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


# Cache historical episodes so we only load them once
_historical_cache = None


def _get_historical_episodes():
    global _historical_cache
    if _historical_cache is None:
        datasets = load_datasets()
        _historical_cache = build_offline_episodes(
            datasets["state_features"],
            datasets["historical_activity"],
            datasets["action_eligibility"],
        )
    return _historical_cache


def run_nightly_cycle(
    day: int,
    champion: ActorCriticCQL,
    patient_snapshots: list,
    eligibility_snapshots: list,
    dynamics_model=None,
    reward_model=None,
    cql_epochs: int = 5,
    eval_episodes: int = 50,
    history_replay_frac: float = 0.05,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Dyna-style nightly update.

    1. Take today's experiences (~5000 transitions)
    2. Sample a small fraction of historical episodes for replay
    3. Clone the champion, do a quick CQL update on the mixed batch
    4. Evaluate champion vs challenger on the gym env
    5. Promote if challenger is better

    Args:
        day: Current simulation day.
        champion: Current champion agent.
        patient_snapshots: Patient state snapshots.
        eligibility_snapshots: Eligibility snapshots.
        cql_epochs: Quick CQL epochs for the update (default 5).
        eval_episodes: Episodes for evaluation (default 50).
        history_replay_frac: Fraction of historical data to replay (default 5%).
        verbose: Print progress.
    """
    if verbose:
        print(f"\n  Night {day}: Dyna-style update", flush=True)

    rng = np.random.default_rng(day)

    # --- Build training batch from 3 tiers of experience ---
    # In production, all of this is one experience store. The tiers are:
    #   1. TODAY — full batch, freshest signal
    #   2. RECENT (last 14 days) — has retroactively updated closure rewards
    #   3. HISTORICAL — sample from older data to prevent catastrophic forgetting

    # Tier 1: Today's experiences (all of them)
    today_experiences = _load_today_experiences(day)
    today_episodes = _experiences_to_episodes(today_experiences, rng)

    # Tier 2: Recent simulation experiences (last 14 days, all of them)
    recent_experiences = []
    for recent_day in range(max(1, day - 14), day):
        recent_experiences.extend(_load_today_experiences(recent_day))
    recent_episodes = _experiences_to_episodes(recent_experiences, rng)

    # Tier 3: Historical sample (50% of pre-simulation data)
    historical = _get_historical_episodes()
    n_replay = max(1, int(len(historical) * 0.50))
    replay_indices = rng.choice(len(historical), size=min(n_replay, len(historical)), replace=False)
    replay_episodes = [historical[i] for i in replay_indices]

    # Combine all tiers
    training_episodes = today_episodes + recent_episodes + replay_episodes
    total_transitions = sum(len(ep["obs"]) - 1 for ep in training_episodes if len(ep["obs"]) > 1)

    if verbose:
        print(f"    Training batch: {len(today_episodes)} today + "
              f"{len(recent_episodes)} recent + {len(replay_episodes)} replay "
              f"= {total_transitions} transitions", flush=True)

    # --- 4. Online update of world models (dynamics + reward) ---
    # Same data, same tiers as CQL — world models train side-by-side with the agent.
    from simulation.logger import get_logger
    log = get_logger()

    try:
        import torch

        if dynamics_model is not None and total_transitions > 10:
            dynamics_model.train()
            opt_d = torch.optim.Adam(dynamics_model.parameters(), lr=1e-4)
            # Train on ALL episodes (same batch as CQL)
            for ep in training_episodes:
                if len(ep["obs"]) < 2:
                    continue
                states = torch.FloatTensor(ep["obs"][:-1])
                actions = torch.LongTensor(ep["actions"][:-1])
                next_states = torch.FloatTensor(ep["obs"][1:])
                opt_d.zero_grad()
                loss = dynamics_model.compute_loss(states, actions, next_states)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dynamics_model.parameters(), 1.0)
                opt_d.step()

        if reward_model is not None and total_transitions > 10:
            reward_model.train()
            opt_r = torch.optim.Adam(reward_model.parameters(), lr=1e-4)
            # Train on ALL episodes (same batch as CQL)
            for ep in training_episodes:
                if len(ep["obs"]) < 2:
                    continue
                states = torch.FloatTensor(ep["obs"])
                actions = torch.LongTensor(ep["actions"])
                rewards = torch.FloatTensor(ep["rewards"])
                # Positive reward → gap likely closed
                labels = (rewards > 0.1).float()
                days = torch.ones(len(states)) * 30.0  # Train at 30-day horizon
                opt_r.zero_grad()
                loss = reward_model.compute_loss(states, actions, days, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(reward_model.parameters(), 1.0)
                opt_r.step()
    except Exception as e:
        log.error(f"World model online update failed: {e}")

    # --- 5. Clone champion and do quick CQL update ---
    try:
        challenger = ActorCriticCQL()
        challenger.load_state_dict(copy.deepcopy(champion.state_dict()))

        if total_transitions > 10:
            challenger = train_cql(
                episodes=training_episodes,
                agent=challenger,
                epochs=cql_epochs,
                batch_size=1024,
                verbose=False,
            )
    except Exception as e:
        log.error(f"CQL training failed: {e}")
        challenger = champion  # Fall back to champion

    # --- 6. Evaluate champion vs challenger on LEARNED world ---
    try:
        env = HEDISEnv(
            patient_snapshots, eligibility_snapshots,
            dynamics_model=dynamics_model,
            reward_model=reward_model,
        )

        champion_metrics = evaluate_agent(champion, env, n_episodes=eval_episodes, seed=day * 1000)
        challenger_metrics = evaluate_agent(challenger, env, n_episodes=eval_episodes, seed=day * 1000)

        comparison = compare_models(champion_metrics, challenger_metrics)
    except Exception as e:
        log.error(f"Evaluation failed: {e}")
        champion_metrics = {"mean_reward": 0.0, "mean_gaps_closed": 0.0, "no_action_rate": 0.0, "n_episodes": 0, "std_reward": 0.0, "mean_episode_length": 0.0}
        challenger_metrics = champion_metrics.copy()
        comparison = {"champion_mean_reward": 0.0, "challenger_mean_reward": 0.0, "relative_improvement": 0.0, "promote_challenger": False, "champion_gaps_closed": 0.0, "challenger_gaps_closed": 0.0}

    if verbose:
        print(f"    Champion: {comparison['champion_mean_reward']:.4f} | "
              f"Challenger: {comparison['challenger_mean_reward']:.4f} | "
              f"{'PROMOTE' if comparison['promote_challenger'] else 'retain'}", flush=True)

    # Save challenger checkpoint
    challenger_path = os.path.join(CHECKPOINTS_DIR, f"challenger_day_{day:02d}.pt")
    torch.save(challenger.state_dict(), challenger_path)

    promoted = comparison["promote_challenger"]
    if promoted:
        champion_path = os.path.join(CHECKPOINTS_DIR, "champion.pt")
        torch.save(challenger.state_dict(), champion_path)
        new_champion = challenger
    else:
        new_champion = champion

    # --- 7. Detailed simulation rollout on the learned world ---
    winner = new_champion
    try:
        # 90-day quarter simulation on 1000 patients using WorldSimulator (ground truth)
        sim_detail = evaluate_agent_detailed(
            winner, patient_snapshots,
            n_episodes=1000, seed=day * 2000,
            eligibility_snapshots=eligibility_snapshots,
        )
    except Exception as e:
        log.error(f"Detailed evaluation failed: {e}")
        sim_detail = {"mean_reward": 0.0, "std_reward": 0.0, "total_actions": 0,
                      "no_action_count": 0, "no_action_rate": 0.0,
                      "sim_closure_rates": {}, "sim_channel_rates": {},
                      "action_dist_by_measure": {}, "action_dist_by_channel": {},
                      "n_episodes": 0}

    # Save nightly metrics + simulation predictions
    day_dir = os.path.join(SIMULATION_DATA_DIR, f"day_{day:02d}")
    os.makedirs(day_dir, exist_ok=True)
    # Extract training debug metrics
    training_debug = {}
    if hasattr(challenger, '_training_history') and challenger._training_history:
        history = challenger._training_history
        training_debug = {
            "final_critic_loss": history[-1].get("critic_loss", 0),
            "final_td_loss": history[-1].get("td_loss", 0),
            "final_actor_loss": history[-1].get("actor_loss", 0),
            "final_cql_penalty": history[-1].get("cql_penalty", 0),
            "final_alpha": history[-1].get("alpha", 0),
            "final_entropy": history[-1].get("entropy", 0),
            "loss_history": [
                {"epoch": h["epoch"], "critic": h["critic_loss"], "td": h["td_loss"],
                 "actor": h["actor_loss"], "cql": h["cql_penalty"],
                 "alpha": h["alpha"], "entropy": h["entropy"]}
                for h in history
            ],
            "step_history": getattr(challenger, '_step_history', []),
        }

    # Compute Q-value stats for debugging
    try:
        import torch
        champion.critic.eval()
        with torch.no_grad():
            sample = torch.randn(100, STATE_DIM)
            q_min = champion.critic.q_min(sample)
            training_debug["q_mean"] = float(q_min.mean())
            training_debug["q_std"] = float(q_min.std())
            training_debug["q_min"] = float(q_min.min())
            training_debug["q_max"] = float(q_min.max())
    except Exception:
        pass

    nightly_metrics = {
        "day": day,
        "champion_metrics": champion_metrics,
        "challenger_metrics": challenger_metrics,
        "comparison": comparison,
        "promoted": promoted,
        "training_transitions": total_transitions,
        "replay_episodes": len(replay_episodes),
        "sim_detail": sim_detail,
        "training_debug": training_debug,
    }
    with open(os.path.join(day_dir, "nightly_metrics.json"), "w") as f:
        json.dump(nightly_metrics, f, indent=2, default=str)

    # Also save cumulative simulation predictions for dashboard
    sim_predictions_path = os.path.join(SIMULATION_DATA_DIR, "sim_predictions.json")
    # Append to existing predictions
    all_predictions = []
    if os.path.exists(sim_predictions_path):
        try:
            with open(sim_predictions_path) as f:
                all_predictions = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    all_predictions.append({"day": day, **sim_detail})
    with open(sim_predictions_path, "w") as f:
        json.dump(all_predictions, f, indent=2, default=str)

    # Save cumulative training debug for dashboard
    debug_path = os.path.join(SIMULATION_DATA_DIR, "training_debug.json")
    all_debug = []
    if os.path.exists(debug_path):
        try:
            with open(debug_path) as f:
                all_debug = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    all_debug.append({"day": day, **training_debug})
    with open(debug_path, "w") as f:
        json.dump(all_debug, f, indent=2, default=str)

    return {
        "new_champion": new_champion,
        "promoted": promoted,
        "champion_score": champion_metrics["mean_reward"],
        "challenger_score": challenger_metrics["mean_reward"],
        "comparison": comparison,
        "sim_detail": sim_detail,
    }
