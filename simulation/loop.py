"""
30-day simulation orchestrator.
Coordinates daily and nightly cycles, tracks metrics, and feeds the dashboard.
"""
import json
import os
import sys
import torch
import numpy as np
from typing import Optional

from config import (
    SIMULATION_DAYS, SIMULATION_DATA_DIR, CHECKPOINTS_DIR,
    GENERATED_DATA_DIR, COHORT_SIZE,
)
from training.data_loader import load_datasets, build_offline_episodes
from training.behavior_cloning import train_behavior_cloning, ActionMaskedPolicy
from training.cql_trainer import train_cql, ActorCriticCQL
from simulation.daily_cycle import run_daily_cycle
from simulation.nightly_cycle import run_nightly_cycle
from simulation.lagged_rewards import LaggedRewardQueue
from simulation.action_state_machine import ActionLifecycleTracker
from simulation.metrics import MetricsTracker
from simulation.logger import init_logger


def run_simulation(
    n_days: int = SIMULATION_DAYS,
    bc_epochs: int = 50,
    cql_epochs: int = 30,
    eval_episodes: int = 200,
    seed: int = 42,
    verbose: bool = True,
):
    """Run the full 30-day simulation."""
    # Force unbuffered stdout
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

    rng = np.random.default_rng(seed)
    os.makedirs(SIMULATION_DATA_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    log = init_logger()
    log.phase("SIMULATION STARTING", n_days=n_days, bc_epochs=bc_epochs, cql_epochs=cql_epochs)

    # =========================================================================
    # DAY 0: Initialize
    # =========================================================================
    log.phase("DAY 0: Initialization")

    log.info("Loading datasets...")
    datasets = load_datasets()
    patient_snapshots = datasets["state_features"]
    eligibility_snapshots = datasets["action_eligibility"]
    log.info(f"Loaded {len(patient_snapshots)} patients")

    log.info("Building offline episodes...")
    episodes = build_offline_episodes(
        datasets["state_features"],
        datasets["historical_activity"],
        datasets["action_eligibility"],
    )
    log.info(f"Built {len(episodes)} episodes")

    # Phase 1: Behavior Cloning
    log.phase("Phase 1: Behavior Cloning Training", epochs=bc_epochs)
    bc_policy = train_behavior_cloning(episodes=episodes, epochs=bc_epochs, verbose=verbose)
    bc_path = os.path.join(CHECKPOINTS_DIR, "bc_policy.pt")
    torch.save(bc_policy.state_dict(), bc_path)
    log.info(f"BC policy saved to {bc_path}")

    # Phase 2: Initial CQL
    log.phase("Phase 2: Initial CQL Fine-Tuning", epochs=min(cql_epochs, 30))
    champion = train_cql(
        episodes=episodes,
        bc_policy=bc_policy,
        epochs=min(cql_epochs, 30),
        verbose=verbose,
    )
    champion_path = os.path.join(CHECKPOINTS_DIR, "champion.pt")
    torch.save(champion.state_dict(), champion_path)
    log.phase("v1 model deployed", checkpoint=champion_path)

    # Initialize tracking systems
    state_machine = ActionLifecycleTracker(rng=rng)
    lagged_queue = LaggedRewardQueue(rng=rng)
    metrics = MetricsTracker()
    model_version = 1

    # Save init metrics
    init_metrics = {
        "day": 0, "phase": "initialization", "model_version": model_version,
        "bc_epochs": bc_epochs, "cql_epochs": cql_epochs,
        "cohort_size": len(patient_snapshots),
    }
    with open(os.path.join(SIMULATION_DATA_DIR, "init_metrics.json"), "w") as f:
        json.dump(init_metrics, f, indent=2)

    # Patient budgets persist across days
    patient_budgets = {}

    # =========================================================================
    # DAYS 1-30: Simulation Loop
    # =========================================================================
    for day in range(1, n_days + 1):
        log.phase(f"DAY {day}/{n_days}", model_version=model_version)

        # --- DAY PHASE ---
        log.info(f"Day phase: Agent interacting with {len(patient_snapshots)} patients...")

        day_results = run_daily_cycle(
            day=day,
            agent=champion,
            patient_snapshots=patient_snapshots,
            eligibility_snapshots=eligibility_snapshots,
            state_machine=state_machine,
            lagged_queue=lagged_queue,
            rng=rng,
            patient_budgets=patient_budgets,
        )

        # Persist budgets for next day
        patient_budgets = day_results.get("patient_budgets", patient_budgets)

        gap_closures_total = sum(day_results["gap_closures"].values())
        log.metric(
            f"Day {day} results: {day_results['num_actions']} actions, "
            f"reward={day_results['total_reward']:.2f}, "
            f"closures={gap_closures_total}, "
            f"pending_rewards={day_results['pending_rewards']}",
            day=day,
            actions=day_results["num_actions"],
            reward=day_results["total_reward"],
            gap_closures=gap_closures_total,
            pending=day_results["pending_rewards"],
        )

        funnel = day_results.get("state_machine_funnel", {})
        log.info(f"State machine: {funnel}")
        log.info(
            f"Budget: avg_remaining={day_results.get('avg_budget_remaining', 0):.1f}, "
            f"exhausted_patients={day_results.get('budget_exhausted_count', 0)}"
        )

        # --- NIGHT PHASE ---
        log.info(f"Night phase: Retraining challenger...")

        night_results = run_nightly_cycle(
            day=day,
            champion=champion,
            patient_snapshots=patient_snapshots,
            eligibility_snapshots=eligibility_snapshots,
            cql_epochs=cql_epochs,
            eval_episodes=eval_episodes,
            verbose=verbose,
        )

        if night_results["promoted"]:
            champion = night_results["new_champion"]
            model_version += 1
            log.metric(
                f"CHALLENGER PROMOTED to v{model_version}! "
                f"Score: {night_results['challenger_score']:.4f} > {night_results['champion_score']:.4f}",
                promoted=True, model_version=model_version,
            )
        else:
            log.info(
                f"Champion retained. "
                f"Champion={night_results['champion_score']:.4f}, "
                f"Challenger={night_results['challenger_score']:.4f}",
            )

        # Record metrics
        day_metrics = metrics.record_day(
            day=day,
            daily_reward=day_results["total_reward"],
            daily_actions=day_results["num_actions"],
            daily_gap_closures=day_results["gap_closures"],
            daily_total_patients=day_results["total_patients"],
            champion_score=night_results["champion_score"],
            challenger_score=night_results["challenger_score"],
            model_promoted=night_results["promoted"],
            model_version=model_version,
            action_distribution=day_results.get("action_distribution"),
            state_machine_funnel=day_results.get("state_machine_funnel"),
        )

        # Save cumulative metrics for dashboard
        with open(os.path.join(SIMULATION_DATA_DIR, "cumulative_metrics.json"), "w") as f:
            json.dump(metrics.to_records(), f, indent=2, default=str)

        log.metric(
            f"STARS Score: {day_metrics['stars_score']:.2f} | "
            f"Cumulative Reward: {day_metrics['cumulative_reward']:.2f} | "
            f"Model: v{model_version}",
            stars=day_metrics["stars_score"],
            cumulative_reward=day_metrics["cumulative_reward"],
            above_threshold=day_metrics["above_bonus_threshold"],
        )

        if day_metrics["above_bonus_threshold"]:
            log.phase("*** ABOVE 4.0 BONUS THRESHOLD! ***")

    # =========================================================================
    # Summary
    # =========================================================================
    final = metrics.get_latest()
    log.phase(
        "SIMULATION COMPLETE",
        final_stars=final.get("stars_score"),
        final_model=model_version,
        total_reward=final.get("cumulative_reward"),
        bonus_achieved=final.get("above_bonus_threshold"),
    )

    return metrics
