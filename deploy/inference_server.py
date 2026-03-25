"""
NBA Stars Model — Inference Server

Lightweight Flask/REST endpoint that loads the champion model and serves
next-best-action predictions. Deployed as a Docker container.

Endpoints:
    POST /predict       — Single patient action recommendation
    POST /predict/batch — Batch recommendations
    GET  /health        — Health check with model version
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from flask import Flask, request, jsonify

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    NUM_ACTIONS, STATE_DIM, ACTION_BY_ID, MEASURE_DESCRIPTIONS,
    AVG_MESSAGES_PER_PATIENT,
)
from environment.state_space import snapshot_to_vector
from environment.action_masking import compute_action_mask
from training.cql_trainer import ActorCriticCQL

app = Flask(__name__)

# Global model state
_model = None
_model_version = 0
_start_time = time.time()


def load_model(checkpoint_path: str = "checkpoints/champion.pt"):
    """Load the champion model from checkpoint."""
    global _model, _model_version
    _model = ActorCriticCQL()
    if os.path.exists(checkpoint_path):
        _model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        _model_version += 1
    _model.actor.eval()
    _model.critic.eval()
    return _model


def predict_single(patient_data: dict, system_context: dict = None) -> dict:
    """Generate action recommendation for a single patient."""
    ctx = system_context or {}
    action_ctx = patient_data.get("action_context", {})

    # Build state vector from 3-tier request data
    state_vec = snapshot_to_vector(
        patient_data,
        # Tier 2: System state
        day_of_year=ctx.get("day_of_year", 15),
        budget_remaining=ctx.get("global_budget_remaining", patient_data.get("global_budget_remaining")),
        budget_max=ctx.get("global_budget_max", patient_data.get("global_budget_max")),
        budget_daily_spend=ctx.get("budget_daily_spend", 0.0),
        cohort_size=ctx.get("cohort_size", 5000),
        cohort_avg_messages=ctx.get("cohort_avg_messages", float(AVG_MESSAGES_PER_PATIENT)),
        stars_score=ctx.get("stars_score", 1.0),
        stars_7d_trend=ctx.get("stars_7d_trend", 0.0),
        pct_measures_above_4=ctx.get("pct_measures_above_4", 0.0),
        lowest_measure_stars=ctx.get("lowest_measure_stars", 1.0),
        cohort_channel_rates=ctx.get("cohort_channel_rates"),
        # Tier 3: Action context
        patient_messages_received=action_ctx.get("messages_received", patient_data.get("patient_messages_received", 0)),
        patient_response_rate=action_ctx.get("response_rate", patient_data.get("patient_response_rate", 0.0)),
        patient_contacts_7d=action_ctx.get("contacts_7d", patient_data.get("contacts_this_week", 0)),
        patient_contacts_14d=action_ctx.get("contacts_14d", 0),
        patient_contacts_30d=action_ctx.get("contacts_30d", 0),
        patient_days_since_contact=action_ctx.get("days_since_contact", 90),
        patient_channels_used=action_ctx.get("channels_used", patient_data.get("patient_channels_used", 0)),
        patient_channel_success=action_ctx.get("channel_success_rates"),
        patient_days_since_closure=action_ctx.get("days_since_closure", patient_data.get("patient_days_since_closure", 90.0)),
        patient_avg_gap_age=action_ctx.get("avg_gap_age", patient_data.get("patient_avg_gap_age", 0.0)),
        num_pending_actions=action_ctx.get("num_pending_actions", 0),
        num_in_flight_measures=action_ctx.get("num_in_flight_measures", 0),
    )

    # Build action mask
    engagement = patient_data.get("engagement", {})
    open_gaps = set(patient_data.get("open_gaps", []))
    mask = compute_action_mask(
        open_gaps=open_gaps,
        channel_availability={
            "sms": engagement.get("sms_consent", False),
            "email": engagement.get("email_available", False),
            "portal": engagement.get("portal_registered", False),
            "app": engagement.get("app_installed", False),
            "ivr": True,
        },
        contacts_this_week=patient_data.get("contacts_this_week", 0),
        budget_remaining=patient_data.get("global_budget_remaining"),
    )

    # Get Q-values and select action
    with torch.no_grad():
        state_t = torch.FloatTensor(state_vec).unsqueeze(0)
        q_min = _model.critic.q_min(state_t).squeeze().numpy()

    # Apply mask
    masked_q = q_min.copy()
    masked_q[~mask] = float("-inf")

    action_id = int(np.argmax(masked_q))
    act = ACTION_BY_ID.get(action_id)

    # Build response
    is_no_action = action_id == 0
    response = {
        "patient_id": patient_data.get("patient_id", ""),
        "recommended_action": _action_detail(act, float(masked_q[action_id])),
        "is_no_action": is_no_action,
        "q_value": float(masked_q[action_id]),
    }

    # Top 5 alternatives
    valid_indices = np.where(mask)[0]
    sorted_valid = sorted(valid_indices, key=lambda i: masked_q[i], reverse=True)
    response["alternative_actions"] = [
        _action_detail(ACTION_BY_ID.get(int(i)), float(masked_q[i]))
        for i in sorted_valid[1:6]
    ]

    # Mask summary
    blocked = {"gap_not_open": 0, "channel_blocked": 0, "budget_exhausted": 0}
    response["action_mask_summary"] = {
        "total_valid_actions": int(mask.sum()),
        "blocked_reasons": blocked,
    }

    return response


def _action_detail(act, q_val: float) -> dict:
    if act is None or act.measure == "NO_ACTION":
        return {"action_id": 0, "measure": "NO_ACTION", "channel": "none",
                "variant": "none", "q_value": q_val}
    return {
        "action_id": act.action_id,
        "measure": act.measure,
        "measure_description": MEASURE_DESCRIPTIONS.get(act.measure, act.measure),
        "channel": act.channel,
        "variant": act.variant,
        "variant_description": act.description,
        "q_value": q_val,
    }


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    result = predict_single(data)
    return jsonify(result)


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json()
    patients = data.get("patients", [])
    system_context = {
        "global_budget_remaining": data.get("global_budget_remaining"),
        "global_budget_max": data.get("global_budget_max"),
        "day_of_year": data.get("day_of_year", 15),
        "cohort_size": data.get("cohort_size", len(patients)),
        "cohort_avg_messages": data.get("cohort_avg_messages", float(AVG_MESSAGES_PER_PATIENT)),
        "stars_score": data.get("stars_score", 1.0),
        "stars_7d_trend": data.get("stars_7d_trend", 0.0),
        "pct_measures_above_4": data.get("pct_measures_above_4", 0.0),
        "lowest_measure_stars": data.get("lowest_measure_stars", 1.0),
        "cohort_channel_rates": data.get("cohort_channel_rates"),
        "budget_daily_spend": data.get("budget_daily_spend", 0.0),
    }
    results = [predict_single(p, system_context) for p in patients]
    budget_used = sum(1 for r in results if not r["is_no_action"])
    return jsonify({"predictions": results, "budget_used": budget_used})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_version": _model_version,
        "model_checkpoint": "champion.pt",
        "uptime_seconds": time.time() - _start_time,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--checkpoint", default="checkpoints/champion.pt")
    args = parser.parse_args()

    load_model(args.checkpoint)
    print(f"NBA Stars Model Inference Server — port {args.port}")
    app.run(host="0.0.0.0", port=args.port)
