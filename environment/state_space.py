"""
State space definition and feature vector construction.
Flattens patient state snapshots into fixed-size numpy vectors.
"""
import numpy as np
from typing import Dict, Any, List

from config import HEDIS_MEASURES, STATE_DIM, NUM_ACTIONS

# Feature indices for interpretability
FEATURE_NAMES: List[str] = []


def _build_feature_names():
    """Build ordered list of feature names matching the vector layout."""
    names = []
    # Demographics (6)
    names.extend(["age_norm", "sex_male", "sex_female", "dual_eligible", "lis_status", "snp_flag"])
    # Clinical vitals (6)
    names.extend(["bp_systolic_norm", "bp_diastolic_norm", "a1c_norm", "bmi_norm", "ckd_stage_norm", "phq9_norm"])
    # Condition flags (8)
    names.extend([f"cond_{c}" for c in [
        "diabetes", "hypertension", "hyperlipidemia", "depression",
        "ckd", "chd", "copd", "chf"
    ]])
    # Med fill rates (4)
    names.extend(["fill_statin", "fill_ace_arb", "fill_diabetes_oral", "fill_antidepressant"])
    # Open gap flags (18)
    names.extend([f"gap_{m}" for m in HEDIS_MEASURES])
    # Engagement (11)
    names.extend([
        "sms_consent", "email_available", "portal_registered", "app_installed",
        "sms_response_rate", "email_open_rate", "portal_engagement_rate",
        "app_engagement_rate", "ivr_completion_rate",
        "total_contacts_90d_norm", "days_since_last_contact_norm",
    ])
    # Risk scores (4)
    names.extend(["readmission_risk", "disenrollment_risk", "non_compliance_risk", "composite_acuity_norm"])
    # Global budget + patient contact context (10)
    names.extend([
        "global_budget_remaining_norm",     # Global pool remaining / total (0=exhausted, 1=full)
        "global_budget_utilization",        # Fraction of global budget used so far
        "global_budget_is_warning",         # 1.0 if global budget < 25%
        "global_budget_is_critical",        # 1.0 if global budget < 10%
        "patient_messages_received_norm",   # Messages THIS patient received / avg per patient
        "patient_messages_vs_avg",          # This patient's messages / cohort avg (>1 = above avg)
        "patient_historical_response_rate", # Has patient ever clicked/accepted? (0-1)
        "patient_avg_gap_age_norm",         # Avg days gaps have been open / 365 (urgency signal)
        "patient_days_since_last_closure",  # Days since this patient last closed any gap / 90
        "patient_channel_diversity",        # How many distinct channels used / 5 (saturation signal)
    ])
    # Temporal (3)
    names.extend(["day_of_year_sin", "day_of_year_cos", "days_remaining_norm"])
    # Action history (10) - last 5 actions encoded as (measure_idx, channel_idx)
    for i in range(5):
        names.extend([f"last_action_{i}_measure", f"last_action_{i}_channel"])
    # Gap-specific (10) - top 5 open gaps: days since last attempt, attempt count
    for i in range(5):
        names.extend([f"gap_{i}_days_since_attempt", f"gap_{i}_attempt_count"])
    # Padding
    while len(names) < STATE_DIM:
        names.append(f"pad_{len(names)}")
    return names[:STATE_DIM]


FEATURE_NAMES = _build_feature_names()

# Channel index mapping
CHANNEL_TO_IDX = {"sms": 0, "email": 1, "portal": 2, "app": 3, "ivr": 4, "none": -1}
MEASURE_TO_IDX = {m: i for i, m in enumerate(HEDIS_MEASURES)}


def snapshot_to_vector(
    snapshot: Dict[str, Any],
    action_history: List[int] = None,
    gap_attempt_info: Dict[str, Dict] = None,
    day_of_year: int = 15,
    budget_remaining: int = None,
    budget_max: int = None,
    patient_messages_received: int = 0,
    cohort_avg_messages: float = 0.0,
    patient_response_rate: float = 0.0,
    patient_avg_gap_age: float = 0.0,
    patient_days_since_closure: float = 90.0,
    patient_channels_used: int = 0,
) -> np.ndarray:
    """Convert a patient state snapshot dict to a fixed-size feature vector.

    Args:
        snapshot: Patient state snapshot from state_features dataset.
        action_history: List of recent action IDs (most recent first), up to 5.
        gap_attempt_info: Dict mapping measure -> {"days_since": int, "count": int}.
        day_of_year: Current day of the measurement year (1-365).
        budget_remaining: Global budget remaining.
        budget_max: Total global budget.
        patient_messages_received: Messages this patient has received so far.
        cohort_avg_messages: Average messages per patient across cohort.
        patient_response_rate: Historical click/accept rate for this patient (0-1).
        patient_avg_gap_age: Average days this patient's gaps have been open.
        patient_days_since_closure: Days since this patient last closed any gap.
        patient_channels_used: Number of distinct channels used for this patient.

    Returns:
        numpy array of shape (STATE_DIM,).
    """
    vec = np.zeros(STATE_DIM, dtype=np.float32)
    idx = 0

    demo = snapshot["demographics"]
    clinical = snapshot["clinical"]
    engagement = snapshot["engagement"]
    risk = snapshot["risk_scores"]
    med_fills = snapshot["medication_fill_rates"]

    # Demographics (6)
    vec[idx] = demo["age"] / 100.0; idx += 1
    vec[idx] = 1.0 if demo["sex"] == "M" else 0.0; idx += 1
    vec[idx] = 1.0 if demo["sex"] == "F" else 0.0; idx += 1
    vec[idx] = float(demo["dual_eligible"]); idx += 1
    vec[idx] = float(demo["lis_status"]); idx += 1
    vec[idx] = float(demo["snp_flag"]); idx += 1

    # Clinical vitals (6)
    vec[idx] = clinical["bp_systolic_last"] / 200.0; idx += 1
    vec[idx] = clinical["bp_diastolic_last"] / 120.0; idx += 1
    vec[idx] = clinical["a1c_last"] / 14.0; idx += 1
    vec[idx] = clinical["bmi"] / 55.0; idx += 1
    vec[idx] = clinical["ckd_stage"] / 5.0; idx += 1
    vec[idx] = clinical["phq9_score"] / 27.0; idx += 1

    # Condition flags (8)
    conditions = clinical.get("conditions", {})
    for cond in ["diabetes", "hypertension", "hyperlipidemia", "depression",
                 "ckd", "chd", "copd", "chf"]:
        vec[idx] = float(conditions.get(cond, False)); idx += 1

    # Med fill rates (4)
    vec[idx] = med_fills.get("statin", 0.0); idx += 1
    vec[idx] = med_fills.get("ace_arb", 0.0); idx += 1
    vec[idx] = med_fills.get("diabetes_oral", 0.0); idx += 1
    vec[idx] = med_fills.get("antidepressant", 0.0); idx += 1

    # Open gap flags (18)
    open_gaps = set(snapshot.get("open_gaps", []))
    for m in HEDIS_MEASURES:
        vec[idx] = 1.0 if m in open_gaps else 0.0; idx += 1

    # Engagement (11)
    vec[idx] = float(engagement.get("sms_consent", False)); idx += 1
    vec[idx] = float(engagement.get("email_available", False)); idx += 1
    vec[idx] = float(engagement.get("portal_registered", False)); idx += 1
    vec[idx] = float(engagement.get("app_installed", False)); idx += 1
    vec[idx] = engagement.get("sms_response_rate", 0.0); idx += 1
    vec[idx] = engagement.get("email_open_rate", 0.0); idx += 1
    vec[idx] = engagement.get("portal_engagement_rate", 0.0); idx += 1
    vec[idx] = engagement.get("app_engagement_rate", 0.0); idx += 1
    vec[idx] = engagement.get("ivr_completion_rate", 0.0); idx += 1
    vec[idx] = min(engagement.get("total_contacts_90d", 0), 20) / 20.0; idx += 1
    vec[idx] = min(engagement.get("days_since_last_contact", 0), 90) / 90.0; idx += 1

    # Risk scores (4)
    vec[idx] = risk.get("readmission_risk", 0.0); idx += 1
    vec[idx] = risk.get("disenrollment_risk", 0.0); idx += 1
    vec[idx] = risk.get("non_compliance_risk", 0.0); idx += 1
    vec[idx] = risk.get("composite_acuity", 0.0) / 5.0; idx += 1

    # Global budget + patient contact context (10)
    from config import BUDGET_WARNING_THRESHOLD, BUDGET_CRITICAL_THRESHOLD, AVG_MESSAGES_PER_PATIENT
    if budget_max is None:
        budget_max = AVG_MESSAGES_PER_PATIENT * 5000  # Fallback
    if budget_remaining is None:
        budget_remaining = budget_max
    budget_frac = budget_remaining / max(budget_max, 1)
    vec[idx] = budget_frac; idx += 1                                          # global_budget_remaining_norm
    vec[idx] = 1.0 - budget_frac; idx += 1                                    # global_budget_utilization
    vec[idx] = 1.0 if budget_frac < BUDGET_WARNING_THRESHOLD else 0.0; idx += 1   # global_budget_is_warning
    vec[idx] = 1.0 if budget_frac < BUDGET_CRITICAL_THRESHOLD else 0.0; idx += 1  # global_budget_is_critical
    # Per-patient contact context
    avg_msg = max(cohort_avg_messages, 1.0)
    vec[idx] = min(patient_messages_received / max(AVG_MESSAGES_PER_PATIENT, 1), 3.0) / 3.0; idx += 1  # patient_messages_received_norm
    vec[idx] = min(patient_messages_received / avg_msg, 3.0) / 3.0 if avg_msg > 0 else 0.0; idx += 1  # patient_messages_vs_avg
    vec[idx] = patient_response_rate; idx += 1                                # patient_historical_response_rate
    vec[idx] = min(patient_avg_gap_age / 365.0, 1.0); idx += 1               # patient_avg_gap_age_norm
    vec[idx] = min(patient_days_since_closure / 90.0, 1.0); idx += 1         # patient_days_since_last_closure
    vec[idx] = patient_channels_used / 5.0; idx += 1                          # patient_channel_diversity

    # Temporal (3)
    vec[idx] = np.sin(2 * np.pi * day_of_year / 365); idx += 1
    vec[idx] = np.cos(2 * np.pi * day_of_year / 365); idx += 1
    vec[idx] = max(0, 365 - day_of_year) / 365.0; idx += 1

    # Action history (10) - last 5 actions as (measure_norm, channel_norm)
    if action_history is None:
        action_history = []
    from config import ACTION_BY_ID
    for i in range(5):
        if i < len(action_history):
            act = ACTION_BY_ID.get(action_history[i])
            if act and act.measure != "NO_ACTION":
                vec[idx] = MEASURE_TO_IDX.get(act.measure, 0) / len(HEDIS_MEASURES); idx += 1
                vec[idx] = CHANNEL_TO_IDX.get(act.channel, 0) / 5.0; idx += 1
            else:
                vec[idx] = 0.0; idx += 1
                vec[idx] = 0.0; idx += 1
        else:
            vec[idx] = 0.0; idx += 1
            vec[idx] = 0.0; idx += 1

    # Gap-specific (10) - top 5 open gaps
    if gap_attempt_info is None:
        gap_attempt_info = {}
    open_gap_list = list(open_gaps)[:5]
    for i in range(5):
        if i < len(open_gap_list):
            gap_info = gap_attempt_info.get(open_gap_list[i], {})
            vec[idx] = min(gap_info.get("days_since", 90), 90) / 90.0; idx += 1
            vec[idx] = min(gap_info.get("count", 0), 10) / 10.0; idx += 1
        else:
            vec[idx] = 0.0; idx += 1
            vec[idx] = 0.0; idx += 1

    return vec
