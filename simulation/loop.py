"""
90-day simulation orchestrator.
Coordinates daily and nightly cycles, tracks metrics, and feeds the dashboard.
Uses WorldSimulator to encapsulate all business rules.
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
from simulation.world import WorldSimulator
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
    """Run the full simulation."""
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

    # Enrich snapshots with archetype behavioral data
    patients_path = os.path.join(GENERATED_DATA_DIR, "patients.json")
    if os.path.exists(patients_path):
        with open(patients_path) as f:
            patients_data = json.load(f)
        patient_lookup = {p["patient_id"]: p for p in patients_data}
        archetype_fields = ["channel_affinity", "channel_engagement", "overall_responsiveness",
                           "timing_optimal_days", "timing_decay", "gap_closure_boost", "variant_boost",
                           "archetype"]
        for snap in patient_snapshots:
            p = patient_lookup.get(snap["patient_id"], {})
            for field in archetype_fields:
                if field in p:
                    snap[field] = p[field]
        log.info(f"Loaded {len(patient_snapshots)} patients (enriched with archetype data)")
    else:
        log.info(f"Loaded {len(patient_snapshots)} patients")

    # Build offline episodes for training
    log.info("Building offline episodes...")
    episodes = build_offline_episodes(
        datasets["state_features"],
        datasets["historical_activity"],
        datasets["action_eligibility"],
    )
    log.info(f"Built {len(episodes)} episodes")

    # Phase 1: Train World Models (dynamics + reward)
    log.phase("Phase 1: Training World Models (dynamics + reward)")
    from models.train_dynamics import train_dynamics_model
    from models.train_reward import train_reward_model
    dynamics_model = train_dynamics_model(
        state_snapshots=datasets["state_features"],
        historical_activity=datasets["historical_activity"],
        epochs=20, verbose=verbose,
    )
    reward_model = train_reward_model(
        state_snapshots=datasets["state_features"],
        historical_activity=datasets["historical_activity"],
        gap_closure=datasets.get("gap_closure"),
        epochs=20, verbose=verbose,
    )
    dynamics_path = os.path.join(CHECKPOINTS_DIR, "dynamics_model.pt")
    reward_path = os.path.join(CHECKPOINTS_DIR, "reward_model.pt")
    torch.save(dynamics_model.state_dict(), dynamics_path)
    torch.save(reward_model.state_dict(), reward_path)
    log.info(f"World models saved: {dynamics_path}, {reward_path}")

    # Phase 2: Behavior Cloning
    log.phase("Phase 2: Behavior Cloning Training", epochs=bc_epochs)
    bc_policy = train_behavior_cloning(episodes=episodes, epochs=bc_epochs, verbose=verbose)
    bc_path = os.path.join(CHECKPOINTS_DIR, "bc_policy.pt")
    torch.save(bc_policy.state_dict(), bc_path)
    log.info(f"BC policy saved to {bc_path}")

    # Phase 3: Initial CQL
    log.phase("Phase 3: Initial CQL Fine-Tuning", epochs=min(cql_epochs, 30))
    champion = train_cql(
        episodes=episodes,
        bc_policy=bc_policy,
        epochs=min(cql_epochs, 30),
        verbose=verbose,
    )
    champion_path = os.path.join(CHECKPOINTS_DIR, "champion.pt")
    torch.save(champion.state_dict(), champion_path)
    log.phase("v1 model deployed", checkpoint=champion_path)

    # =========================================================================
    # Create World Simulator (owns all business rules and patient state)
    # =========================================================================
    world = WorldSimulator(patient_snapshots, eligibility_snapshots, rng=rng)

    # Warm start patients mid-flight
    log.info("Warm-starting patient cohort...")
    warm = world.warm_start(rng)
    budget_pct = world.budget_remaining / max(world.budget_total, 1) * 100
    log.info(
        f"Warm start: {warm['stats']} | "
        f"Budget: {world.budget_remaining:,}/{world.budget_total:,} ({budget_pct:.0f}%) | "
        f"Pending rewards: {warm['pending_rewards']} | "
        f"Day-0 closures: {warm['day0_closures']}"
    )

    metrics = MetricsTracker()
    model_version = 1

    # Save init metrics
    init_metrics = {
        "day": 0, "phase": "initialization", "model_version": model_version,
        "bc_epochs": bc_epochs, "cql_epochs": cql_epochs,
        "cohort_size": len(patient_snapshots),
        "warm_start": warm["stats"],
    }
    with open(os.path.join(SIMULATION_DATA_DIR, "init_metrics.json"), "w") as f:
        json.dump(init_metrics, f, indent=2)

    # =========================================================================
    # DAYS 1-N: Simulation Loop
    # =========================================================================
    for day in range(1, n_days + 1):
        log.phase(f"DAY {day}/{n_days}", model_version=model_version)

        # --- DAY PHASE ---
        try:
            log.info(f"Day phase: Agent interacting with {len(patient_snapshots)} patients...")

            day_results = run_daily_cycle(
                day=day,
                agent=champion,
                world=world,
                rng=rng,
            )

            gap_closures_total = sum(day_results["gap_closures"].values())
            closure_reward = day_results.get("closure_reward", 0)
            immediate_reward = day_results.get("immediate_reward", 0)
            log.metric(
                f"Day {day}: {day_results['num_actions']} actions sent, "
                f"{gap_closures_total} gaps closed (reward: +{closure_reward:.0f}), "
                f"action cost: {immediate_reward:.1f}, "
                f"net reward: {day_results['total_reward']:.1f}",
                day=day,
                actions=day_results["num_actions"],
                reward=day_results["total_reward"],
                gap_closures=gap_closures_total,
                closure_reward=closure_reward,
            )

            budget_pct = world.budget_remaining / max(world.budget_total, 1) * 100
            log.info(
                f"Budget: {world.budget_remaining:,}/{world.budget_total:,} ({budget_pct:.0f}%) | "
                f"Pending closures: {day_results['pending_rewards']:,}"
            )
        except Exception as e:
            log.exception(f"DAY PHASE FAILED on day {day}", exc=e)
            raise

        # --- NIGHT PHASE ---
        try:
            log.info("Night phase: Dyna-style update...")

            night_results = run_nightly_cycle(
                day=day,
                champion=champion,
                patient_snapshots=patient_snapshots,
                eligibility_snapshots=eligibility_snapshots,
                dynamics_model=dynamics_model,
                reward_model=reward_model,
                cql_epochs=cql_epochs,
                eval_episodes=eval_episodes,
                verbose=verbose,
            )
        except Exception as e:
            log.exception(f"NIGHT PHASE FAILED on day {day}", exc=e)
            raise

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
                f"Champ={night_results['champion_score']:.4f}, "
                f"Chall={night_results['challenger_score']:.4f}",
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
            avg_budget_remaining=world.budget_remaining,
            budget_exhausted_count=0,
        )

        # Save cumulative metrics for dashboard
        with open(os.path.join(SIMULATION_DATA_DIR, "cumulative_metrics.json"), "w") as f:
            json.dump(metrics.to_records(), f, indent=2, default=str)

        # Count measures at/above 4★
        detail = day_metrics.get("measure_detail", {})
        at_4star = sum(1 for d in detail.values() if d.get("at_or_above_4", False))
        total_measures = len(detail) if detail else len(HEDIS_MEASURES)

        stars = day_metrics['stars_score']
        gap_to_bonus = 4.0 - stars
        bonus_status = "BONUS!" if day_metrics['above_bonus_threshold'] else f"{gap_to_bonus:.2f} to bonus"
        log.metric(
            f"STARS: {stars:.2f} ({bonus_status}) | "
            f"Measures at 4★: {at_4star}/{total_measures} | "
            f"Total gaps closed: {day_metrics['cumulative_reward']:.0f} | "
            f"Model: v{model_version}",
            stars=day_metrics["stars_score"],
            cumulative_reward=day_metrics["cumulative_reward"],
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
