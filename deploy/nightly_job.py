# Databricks notebook source
"""
NBA Stars Model — Nightly Retraining Job

Runs as a scheduled Databricks job. Reads from Unity Catalog, trains
challenger CQL model, evaluates on learned world, promotes if better.

Schedule: Daily at 2:00 AM ET
Cluster: ML Runtime, GPU optional (CPU works for this model size)

Unity Catalog tables used:
  - nba_stars.experiences.daily_actions     (today's action experiences)
  - nba_stars.outcomes.gap_closures         (resolved closures from claims)
  - nba_stars.experiences.historical_replay (random sample for Dyna replay)
  - nba_stars.models.checkpoints            (model checkpoint storage)
  - nba_stars.metrics.training_results      (nightly training metrics)
  - nba_stars.metrics.sim_predictions       (learned world predictions)
"""

# COMMAND ----------

# MAGIC %pip install torch gymnasium numpy scipy scikit-learn

# COMMAND ----------

import os
import sys
import json
import copy
import numpy as np
import torch
from datetime import datetime, timedelta

# Add project modules to path (deployed as a wheel or mounted volume)
sys.path.insert(0, "/Workspace/Repos/nba-stars-model/rl_agent_foundations")

from config import NUM_ACTIONS, STATE_DIM, SIMULATION_DAYS
from training.cql_trainer import train_cql, ActorCriticCQL
from training.evaluation import evaluate_agent, evaluate_agent_detailed, compare_models
from environment.hedis_env import HEDISEnv
from models.dynamics_model import DynamicsModel
from models.reward_model import RewardModel

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Today's Data from Unity Catalog

# COMMAND ----------

def load_todays_experiences(spark):
    """Load today's action experiences from Unity Catalog."""
    today = datetime.now().strftime("%Y-%m-%d")
    df = spark.sql(f"""
        SELECT patient_id, obs, action, reward, mask
        FROM nba_stars.experiences.daily_actions
        WHERE action_date = '{today}'
    """)
    experiences = []
    for row in df.collect():
        experiences.append({
            "patient_id": row.patient_id,
            "obs": json.loads(row.obs),
            "action": row.action,
            "reward": row.reward,
            "mask": json.loads(row.mask),
        })
    return experiences


def load_resolved_closures(spark, lookback_days=90):
    """Load gap closures that resolved since last training."""
    cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    df = spark.sql(f"""
        SELECT patient_id, measure, action_date, closure_date,
               DATEDIFF(closure_date, action_date) as days_to_closure
        FROM nba_stars.outcomes.gap_closures
        WHERE closure_date >= '{cutoff}'
        AND NOT already_applied
    """)
    closures = []
    for row in df.collect():
        from config import MEASURE_WEIGHTS
        weight = MEASURE_WEIGHTS.get(row.measure, 1)
        closures.append({
            "patient_id": row.patient_id,
            "measure": row.measure,
            "action_day": row.action_date,
            "closure_day": row.closure_date,
            "reward_delta": 1.0 * weight,
        })
    return closures


def load_historical_replay(spark, sample_frac=0.05):
    """Load random sample of historical experiences for Dyna replay."""
    df = spark.sql(f"""
        SELECT patient_id, obs, action, reward, mask
        FROM nba_stars.experiences.historical_replay
        TABLESAMPLE ({sample_frac * 100} PERCENT)
    """)
    episodes = []
    for row in df.collect():
        episodes.append({
            "obs": json.loads(row.obs),
            "action": row.action,
            "reward": row.reward,
            "mask": json.loads(row.mask),
            "patient_id": row.patient_id,
        })
    return episodes


def load_patient_snapshots(spark):
    """Load current patient state snapshots for evaluation."""
    df = spark.sql("""
        SELECT *
        FROM nba_stars.patients.current_state
        LIMIT 1000
    """)
    # Convert to the dict format expected by HEDISEnv
    snapshots = []
    for row in df.collect():
        snapshots.append(json.loads(row.snapshot_json))
    return snapshots

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Current Champion Model

# COMMAND ----------

def load_champion(spark):
    """Load champion model checkpoint from Unity Catalog volumes."""
    checkpoint_path = "/Volumes/nba_stars/models/checkpoints/champion.pt"
    agent = ActorCriticCQL()
    if os.path.exists(checkpoint_path):
        agent.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    return agent, checkpoint_path


def load_world_models(spark):
    """Load learned dynamics and reward models."""
    dyn_path = "/Volumes/nba_stars/models/checkpoints/dynamics_model.pt"
    rew_path = "/Volumes/nba_stars/models/checkpoints/reward_model.pt"

    dynamics = DynamicsModel()
    reward = RewardModel()
    if os.path.exists(dyn_path):
        dynamics.load_state_dict(torch.load(dyn_path, weights_only=True))
    if os.path.exists(rew_path):
        reward.load_state_dict(torch.load(rew_path, weights_only=True))
    return dynamics, reward

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Apply Retroactive Reward Updates

# COMMAND ----------

def apply_retroactive_rewards(spark, closures):
    """Update past experiences with resolved closure rewards."""
    if not closures:
        return 0

    updates = [(c["patient_id"], c["action_day"], c["reward_delta"]) for c in closures]

    # Batch update in Unity Catalog
    for patient_id, action_day, reward_delta in updates:
        spark.sql(f"""
            UPDATE nba_stars.experiences.daily_actions
            SET reward = reward + {reward_delta},
                closure_applied = true
            WHERE patient_id = '{patient_id}'
            AND action_date = '{action_day}'
            AND NOT closure_applied
        """)

    # Mark closures as applied
    spark.sql("""
        UPDATE nba_stars.outcomes.gap_closures
        SET already_applied = true
        WHERE NOT already_applied
    """)

    return len(updates)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Build Training Episodes

# COMMAND ----------

def experiences_to_episodes(experiences):
    """Convert flat experiences to episode format for CQL."""
    from collections import defaultdict
    by_patient = defaultdict(list)
    for exp in experiences:
        by_patient[exp.get("patient_id", "unknown")].append(exp)

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
            "obs": obs, "actions": actions, "rewards": rewards,
            "action_mask": masks,
            "terminateds": np.zeros(T, dtype=np.float32),
            "truncateds": np.array([0.0] * (T - 1) + [1.0], dtype=np.float32),
        })
    return episodes

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Run Nightly Training

# COMMAND ----------

# Load everything
champion, champion_path = load_champion(spark)
dynamics_model, reward_model = load_world_models(spark)
today_exp = load_todays_experiences(spark)
closures = load_resolved_closures(spark)
replay_exp = load_historical_replay(spark)
patient_snapshots = load_patient_snapshots(spark)

print(f"Today's experiences: {len(today_exp)}")
print(f"Resolved closures: {len(closures)}")
print(f"Historical replay: {len(replay_exp)}")

# COMMAND ----------

# Apply retroactive rewards
n_updated = apply_retroactive_rewards(spark, closures)
print(f"Retroactive reward updates applied: {n_updated}")

# COMMAND ----------

# Build training episodes
today_episodes = experiences_to_episodes(today_exp)
replay_episodes = experiences_to_episodes(replay_exp)

# Recent 3 days (load from Unity Catalog)
recent_exp = []
for days_ago in range(1, 4):
    date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    df = spark.sql(f"""
        SELECT patient_id, obs, action, reward, mask
        FROM nba_stars.experiences.daily_actions
        WHERE action_date = '{date}'
    """)
    for row in df.collect():
        recent_exp.append({
            "patient_id": row.patient_id,
            "obs": json.loads(row.obs),
            "action": row.action,
            "reward": row.reward,
            "mask": json.loads(row.mask),
        })
recent_episodes = experiences_to_episodes(recent_exp)

all_episodes = today_episodes + recent_episodes + replay_episodes
total_transitions = sum(len(ep["obs"]) - 1 for ep in all_episodes if len(ep["obs"]) > 1)
print(f"Training batch: {len(today_episodes)} today + {len(recent_episodes)} recent + {len(replay_episodes)} replay = {total_transitions} transitions")

# COMMAND ----------

# Online update dynamics + reward models
if dynamics_model and total_transitions > 10:
    dynamics_model.train()
    opt_d = torch.optim.Adam(dynamics_model.parameters(), lr=5e-4)
    for ep in today_episodes[:50]:
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
    print("Dynamics model updated")

if reward_model and total_transitions > 10:
    reward_model.train()
    opt_r = torch.optim.Adam(reward_model.parameters(), lr=5e-4)
    for ep in today_episodes[:50]:
        if len(ep["obs"]) < 2:
            continue
        states = torch.FloatTensor(ep["obs"])
        actions = torch.LongTensor(ep["actions"])
        rewards = torch.FloatTensor(ep["rewards"])
        labels = (rewards / 3.05).clamp(0, 1)  # Preserve measure weight signal
        days = torch.full((len(states),), 7.0)  # Use actual sim day when available
        opt_r.zero_grad()
        loss = reward_model.compute_loss(states, actions, days, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reward_model.parameters(), 1.0)
        opt_r.step()
    print("Reward model updated")

# COMMAND ----------

# Train challenger CQL
challenger = ActorCriticCQL()
challenger.load_state_dict(copy.deepcopy(champion.state_dict()))

if total_transitions > 10:
    challenger = train_cql(
        episodes=all_episodes,
        agent=challenger,
        epochs=20,
        batch_size=1024,
        verbose=True,
    )
print("Challenger trained")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Evaluate Champion vs Challenger

# COMMAND ----------

# Load eligibility for evaluation
eligibility = [{"patient_id": s["patient_id"], "action_mask": [True] * NUM_ACTIONS}
               for s in patient_snapshots]

env = HEDISEnv(
    patient_snapshots, eligibility,
    dynamics_model=dynamics_model,
    reward_model=reward_model,
)

champion_metrics = evaluate_agent(champion, env, n_episodes=500, seed=42)
challenger_metrics = evaluate_agent(challenger, env, n_episodes=500, seed=42)
comparison = compare_models(champion_metrics, challenger_metrics)

print(f"Champion:   {comparison['champion_mean_reward']:.4f}")
print(f"Challenger: {comparison['challenger_mean_reward']:.4f}")
print(f"Improve:    {comparison['relative_improvement']:.1%}")
print(f"Promote:    {comparison['promote_challenger']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Promote if Better + Save Results

# COMMAND ----------

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

# Set MLflow to use Unity Catalog model registry
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# Detailed simulation predictions
winner = challenger if comparison["promote_challenger"] else champion
sim_detail = evaluate_agent_detailed(winner, env, n_episodes=1000, seed=42)

# ── Register all 3 models in Unity Catalog via MLflow ──
with mlflow.start_run(run_name=f"nightly_{datetime.now().strftime('%Y%m%d')}") as run:

    # Log training metrics
    mlflow.log_metric("champion_reward", comparison["champion_mean_reward"])
    mlflow.log_metric("challenger_reward", comparison["challenger_mean_reward"])
    mlflow.log_metric("improvement", comparison["relative_improvement"])
    mlflow.log_metric("promoted", int(comparison["promote_challenger"]))
    mlflow.log_metric("transitions_trained", total_transitions)
    mlflow.log_metric("closures_resolved", len(closures))
    mlflow.log_metric("sim_no_action_rate", sim_detail.get("no_action_rate", 0))
    mlflow.log_metric("sim_mean_reward", sim_detail.get("mean_reward", 0))

    # Log per-measure closure rates from simulation
    for measure, rate in sim_detail.get("sim_closure_rates", {}).items():
        mlflow.log_metric(f"sim_closure_{measure}", rate)

    # Log params
    mlflow.log_param("cql_epochs", 20)
    mlflow.log_param("replay_frac", 0.05)
    mlflow.log_param("eval_episodes", 500)

    # ── Save CQL Agent (Actor + Twin Critics + Alpha) ──
    cql_model = challenger if comparison["promote_challenger"] else champion
    mlflow.pytorch.log_model(
        pytorch_model=cql_model.actor,
        artifact_path="cql_actor",
    )
    # Save full agent state dict as artifact
    agent_path = "/tmp/cql_agent.pt"
    torch.save(cql_model.state_dict(), agent_path)
    mlflow.log_artifact(agent_path, "cql_agent")

    # ── Save Dynamics Model ──
    dyn_path = "/tmp/dynamics_model.pt"
    torch.save(dynamics_model.state_dict(), dyn_path)
    mlflow.log_artifact(dyn_path, "dynamics_model")

    # ── Save Reward Model ──
    rew_path = "/tmp/reward_model.pt"
    torch.save(reward_model.state_dict(), rew_path)
    mlflow.log_artifact(rew_path, "reward_model")

    # ── Save simulation predictions as artifact ──
    pred_path = "/tmp/sim_predictions.json"
    with open(pred_path, "w") as f:
        json.dump(sim_detail, f, default=str)
    mlflow.log_artifact(pred_path, "predictions")

# ── Register models in Unity Catalog ──
run_id = run.info.run_id

# CQL Agent → nba_stars.models.cql_agent
model_uri = f"runs:/{run_id}/cql_agent"
if comparison["promote_challenger"]:
    mv = mlflow.register_model(model_uri, "nba_stars.models.cql_agent")
    # Transition to Production stage
    client.set_registered_model_alias("nba_stars.models.cql_agent", "champion", mv.version)
    print(f"CQL Agent registered as champion v{mv.version}")

    # Also update Volumes checkpoint for inference server
    torch.save(cql_model.state_dict(), "/Volumes/nba_stars/models/checkpoints/champion.pt")
else:
    print("Champion retained — no model registration")

# Always register dynamics + reward (they update every night)
dyn_uri = f"runs:/{run_id}/dynamics_model"
rew_uri = f"runs:/{run_id}/reward_model"
dyn_mv = mlflow.register_model(dyn_uri, "nba_stars.models.dynamics_model")
rew_mv = mlflow.register_model(rew_uri, "nba_stars.models.reward_model")
client.set_registered_model_alias("nba_stars.models.dynamics_model", "latest", dyn_mv.version)
client.set_registered_model_alias("nba_stars.models.reward_model", "latest", rew_mv.version)

# Save to Volumes for inference server
torch.save(dynamics_model.state_dict(), "/Volumes/nba_stars/models/checkpoints/dynamics_model.pt")
torch.save(reward_model.state_dict(), "/Volumes/nba_stars/models/checkpoints/reward_model.pt")

print(f"Dynamics model v{dyn_mv.version}, Reward model v{rew_mv.version} registered")

# ── Save training results to Unity Catalog table ──
result = {
    "date": datetime.now().isoformat(),
    "mlflow_run_id": run_id,
    "champion_score": comparison["champion_mean_reward"],
    "challenger_score": comparison["challenger_mean_reward"],
    "promoted": comparison["promote_challenger"],
    "transitions_trained": total_transitions,
    "closures_resolved": len(closures),
}

result_df = spark.createDataFrame([{
    "training_date": datetime.now().strftime("%Y-%m-%d"),
    "mlflow_run_id": run_id,
    "promoted": comparison["promote_challenger"],
    "champion_score": float(comparison["champion_mean_reward"]),
    "challenger_score": float(comparison["challenger_mean_reward"]),
    "transitions_trained": total_transitions,
    "result_json": json.dumps(result, default=str),
}])
result_df.write.mode("append").saveAsTable("nba_stars.metrics.training_results")

print("Nightly job complete!")
