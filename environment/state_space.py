"""
State space definition and feature vector construction.

3-TIER FEATURE ARCHITECTURE:
  Tier 1 — Patient State (86):  Who is this patient?
    Demographics(6) + Clinical(6) + Conditions(8) + Medications(4)
    + Gap flags(18) + Risk scores(4) + Channel availability(4) + Measure clinical(36)
  Tier 2 — System State (20):   What's the state of the program?
    Budget(6) + Temporal(3) + Population(2) + STARS progress(4) + Cohort channels(5)
  Tier 3 — Action Context (70): What's been tried for THIS patient?
    Contact history(8) + Channel success(5) + Engagement(5) + Gap context(5)
    + Pending actions(3) + Action history(10) + Gap detail(10) + Measure attempts(24)

Total: 176 features.  (indices 0-175; see config.py TIER1_END/TIER2_END/TIER3_END)
"""
import numpy as np
from typing import Dict, Any, List, Optional

from config import (
    HEDIS_MEASURES, STATE_DIM, NUM_ACTIONS, MEASURE_WEIGHTS,
    TIER1_START, TIER1_END, TIER2_START, TIER2_END, TIER3_START, TIER3_END,
    CONTACT_NORMALIZATION_MAX, DAYS_SINCE_NORMALIZATION_MAX, YEAR_DAYS,
)

# ═══════════════════════════════════════════════════════════════════════════
# Per-Measure Clinical Feature Definitions
# ═══════════════════════════════════════════════════════════════════════════
# Each HEDIS measure has 2 clinical features that help the model decide
# whether this patient is a good candidate for that gap closure action.
# Feature 1: primary clinical indicator (condition flag or lab value)
# Feature 2: secondary indicator (screening history, treatment status, etc.)

MEASURE_CLINICAL_FEATURES = {
    # Screenings
    "COL": ("age_over_50",           "prior_colonoscopy"),         # Colorectal Cancer Screening
    "BCS": ("sex_female",            "prior_mammogram"),           # Breast Cancer Screening
    "EED": ("has_diabetes",          "years_since_eye_exam_norm"), # Eye Exam for Diabetes
    # Vaccines
    "FVA": ("age_over_65",           "prior_flu_vaccine"),         # Flu Vaccine (adult)
    "FVO": ("chronic_condition_count_norm", "prior_flu_vaccine"),  # Flu Vaccine (older)
    "AIS": ("age_over_65",           "prior_pneumo_vaccine"),      # Pneumonia Vaccine
    "FLU": ("immunocompromised",     "prior_flu_vaccine"),         # Annual Flu Shot
    # Chronic conditions
    "CBP": ("has_hypertension",      "bp_controlled"),             # Controlling Blood Pressure
    "BPD": ("has_diabetes_and_hypertension", "bp_controlled"),     # BP for Diabetes
    "HBD": ("has_diabetes",          "a1c_controlled"),            # Hemoglobin A1C Diabetes
    "KED": ("has_ckd",              "has_diabetes"),               # Kidney Disease Evaluation
    # Medication adherence
    "MAC": ("has_cardiovascular",    "statin_adherent"),           # Statin Therapy (Cholesterol)
    "MRA": ("has_diabetes_and_hypertension", "ras_adherent"),     # RAS Antagonist Adherence
    "MDS": ("has_diabetes_and_hyperlipidemia", "statin_adherent"),# Diabetes Statin Adherence
    # Mental health
    "DSF": ("has_depression",        "prior_mh_followup"),         # Depression Follow-Up
    "DRR": ("has_depression",        "on_antidepressant"),         # Depression Remission
    "DMC02": ("has_diabetes",        "years_since_eye_exam_norm"), # Diabetes Eye Care Monitoring
    # Care coordination
    "TRC_M": ("recent_discharge",    "has_pcp"),                   # Transitions of Care
}

# ═══════════════════════════════════════════════════════════════════════════
# Feature Name Registry
# ═══════════════════════════════════════════════════════════════════════════

def _build_feature_names():
    """Build ordered list of feature names matching the 3-tier vector layout."""
    names = []

    # ── TIER 1: Patient State (50 features) ──────────────────────────────
    # Demographics (6)
    names.extend(["age_norm", "sex_male", "sex_female", "dual_eligible", "lis_status", "snp_flag"])
    # Clinical vitals (6)
    names.extend(["bp_systolic_norm", "bp_diastolic_norm", "a1c_norm", "bmi_norm", "ckd_stage_norm", "phq9_norm"])
    # Condition flags (8)
    names.extend([f"cond_{c}" for c in [
        "diabetes", "hypertension", "hyperlipidemia", "depression",
        "ckd", "chd", "copd", "chf"
    ]])
    # Medication fill rates (4)
    names.extend(["fill_statin", "fill_ace_arb", "fill_diabetes_oral", "fill_antidepressant"])
    # Open gap flags (18)
    names.extend([f"gap_{m}" for m in HEDIS_MEASURES])
    # Risk scores (4)
    names.extend(["readmission_risk", "disenrollment_risk", "non_compliance_risk", "composite_acuity_norm"])
    # Channel availability (4)
    names.extend(["sms_consent", "email_available", "portal_registered", "app_installed"])
    # Measure-relevant clinical indicators (36 = 2 per HEDIS measure)
    for m in HEDIS_MEASURES:
        feat1, feat2 = MEASURE_CLINICAL_FEATURES[m]
        names.extend([f"mcl_{m}_{feat1}", f"mcl_{m}_{feat2}"])

    assert len(names) == TIER1_END, f"Tier 1 expected {TIER1_END} features, got {len(names)}"

    # ── TIER 2: System State (20 features) ───────────────────────────────
    # Budget (6)
    names.extend([
        "budget_remaining_norm",       # Global pool remaining / total
        "budget_utilization",          # Fraction used so far
        "budget_is_warning",           # 1.0 if < 25% remaining
        "budget_is_critical",          # 1.0 if < 10% remaining
        "budget_burn_rate_norm",       # Daily avg spend / daily budget
        "budget_projected_days_left",  # Projected days until exhaustion / total days
    ])
    # Temporal (3)
    names.extend(["day_of_year_sin", "day_of_year_cos", "measurement_year_progress"])
    # Population (2)
    names.extend(["cohort_size_norm", "cohort_avg_messages_norm"])
    # STARS progress (4)
    names.extend([
        "overall_stars_norm",          # Current STARS / 5.0
        "stars_7d_trend",              # STARS change over last 7 days (can be negative)
        "pct_measures_above_4star",    # Fraction of measures at or above 4.0
        "lowest_measure_stars_norm",   # Worst-performing measure's star rating / 5.0
    ])
    # Cohort channel effectiveness (5)
    names.extend([
        "cohort_sms_acceptance_rate",
        "cohort_email_acceptance_rate",
        "cohort_portal_acceptance_rate",
        "cohort_app_acceptance_rate",
        "cohort_ivr_acceptance_rate",
    ])

    assert len(names) == TIER2_END, f"Tier 2 expected {TIER2_END} features, got {len(names)}"

    # ── TIER 3: Action Context (58 features) ─────────────────────────────
    # Contact history (8)
    names.extend([
        "patient_messages_received_norm",  # Total messages / expected avg
        "patient_messages_vs_cohort",      # This patient's count / cohort average
        "patient_contacts_7d",             # Contacts in last 7 days / max_per_week
        "patient_contacts_14d",            # Contacts in last 14 days / (2 * max_per_week)
        "patient_contacts_30d",            # Contacts in last 30 days / (4 * max_per_week)
        "patient_days_since_last_contact", # Days since last contact / 90
        "patient_overall_response_rate",   # Click/accept rate across all channels
        "patient_channel_diversity",       # Distinct channels used / 5
    ])
    # Per-channel success rates for THIS patient (5)
    names.extend([
        "patient_sms_success_rate",
        "patient_email_success_rate",
        "patient_portal_success_rate",
        "patient_app_success_rate",
        "patient_ivr_success_rate",
    ])
    # Engagement rates from historical data (5)
    names.extend([
        "sms_response_rate", "email_open_rate", "portal_engagement_rate",
        "app_engagement_rate", "ivr_completion_rate",
    ])
    # Gap context (5)
    names.extend([
        "num_open_gaps_norm",          # Open gaps / 18
        "avg_gap_age_norm",            # Avg days gaps open / 365
        "days_since_last_closure",     # Days since any gap closed / 90
        "highest_priority_gap_weight", # Weight of highest-weight open gap / 3
        "num_closed_gaps_norm",        # Closed gaps / 18
    ])
    # Pending actions (3)
    names.extend([
        "num_pending_actions_norm",    # Actions in-flight / 5
        "num_in_flight_measures",      # Distinct measures with pending actions / 18
        "has_action_in_flight",        # 1.0 if any action is currently in-flight
    ])
    # Channel affinity (10) — lifetime usage volume + recency per channel
    names.extend([
        "lifetime_sms_count_norm",     # Total SMS actions sent to this patient / 10
        "lifetime_email_count_norm",   # Total email actions / 10
        "lifetime_portal_count_norm",  # Total portal actions / 10
        "lifetime_app_count_norm",     # Total app actions / 10
        "lifetime_ivr_count_norm",     # Total IVR actions / 10
        "days_since_last_sms_norm",    # Days since last SMS / 90
        "days_since_last_email_norm",  # Days since last email / 90
        "days_since_last_portal_norm", # Days since last portal / 90
        "days_since_last_app_norm",    # Days since last app / 90
        "days_since_last_ivr_norm",    # Days since last IVR / 90
    ])
    # Gap-specific detail (10) — top 5 open gaps
    for i in range(5):
        names.extend([f"gap_{i}_days_since_attempt", f"gap_{i}_attempt_count"])
    # Per-measure attempt summary (24) — top 3 priority open gaps × 8 features each
    for i in range(3):
        names.extend([
            f"priority_gap_{i}_weight_norm",        # CMS weight (1 or 3) / 3
            f"priority_gap_{i}_attempt_count",      # Attempts / 10
            f"priority_gap_{i}_days_since",         # Days since last attempt / 90
            f"priority_gap_{i}_best_ch_sms",        # 1.0 if best channel is SMS
            f"priority_gap_{i}_best_ch_email",      # 1.0 if best channel is email
            f"priority_gap_{i}_best_ch_portal",     # 1.0 if best channel is portal
            f"priority_gap_{i}_best_ch_app",        # 1.0 if best channel is app
            f"priority_gap_{i}_best_ch_ivr",        # 1.0 if best channel is IVR
        ])

    assert len(names) == TIER3_END, f"Tier 3 expected {TIER3_END} features, got {len(names)}"
    return names


FEATURE_NAMES = _build_feature_names()

# Channel index mapping
CHANNEL_TO_IDX = {"sms": 0, "email": 1, "portal": 2, "app": 3, "ivr": 4, "none": -1}
MEASURE_TO_IDX = {m: i for i, m in enumerate(HEDIS_MEASURES)}


# ═══════════════════════════════════════════════════════════════════════════
# Feature Vector Construction
# ═══════════════════════════════════════════════════════════════════════════

def snapshot_to_vector(
    snapshot: Dict[str, Any],
    # ── Tier 2: System context (from deployment system / WorldSimulator) ──
    day_of_year: int = 15,
    budget_remaining: int = None,
    budget_max: int = None,
    budget_daily_spend: float = 0.0,
    cohort_size: int = 5000,
    cohort_avg_messages: float = 0.0,
    stars_score: float = 1.0,
    stars_7d_trend: float = 0.0,
    pct_measures_above_4: float = 0.0,
    lowest_measure_stars: float = 1.0,
    cohort_channel_rates: Dict[str, float] = None,
    # ── Tier 3: Action context (from PatientState / deployment system) ──
    patient_messages_received: int = 0,
    patient_response_rate: float = 0.0,
    patient_contacts_7d: int = 0,
    patient_contacts_14d: int = 0,
    patient_contacts_30d: int = 0,
    patient_days_since_contact: int = 90,
    patient_channels_used: int = 0,
    patient_channel_success: Dict[str, float] = None,
    patient_days_since_closure: float = 90.0,
    patient_avg_gap_age: float = 0.0,
    num_pending_actions: int = 0,
    num_in_flight_measures: int = 0,
    channel_affinity_counts: Dict[str, int] = None,
    channel_affinity_recency: Dict[str, int] = None,
    gap_attempt_info: Dict[str, Dict] = None,
    measure_attempt_summary: List[Dict] = None,
) -> np.ndarray:
    """Convert a patient state snapshot + context into a 128-dim feature vector.

    Tier 1 (Patient State) comes from the snapshot dict.
    Tier 2 (System State) comes from the deployment system / WorldSimulator.
    Tier 3 (Action Context) comes from patient-specific action tracking.
    """
    vec = np.zeros(STATE_DIM, dtype=np.float32)
    idx = 0

    demo = snapshot.get("demographics", {})
    clinical = snapshot.get("clinical", {})
    engagement = snapshot.get("engagement", {})
    risk = snapshot.get("risk_scores", {})
    med_fills = snapshot.get("medication_fill_rates", {})

    # ═════════════════════════════════════════════════════════════════════
    # TIER 1: Patient State (50 features, indices 0-49)
    # ═════════════════════════════════════════════════════════════════════

    # Demographics (6)
    vec[idx] = demo.get("age", 65) / 100.0; idx += 1
    vec[idx] = 1.0 if demo.get("sex") == "M" else 0.0; idx += 1
    vec[idx] = 1.0 if demo.get("sex") == "F" else 0.0; idx += 1
    vec[idx] = float(demo.get("dual_eligible", False)); idx += 1
    vec[idx] = float(demo.get("lis_status", False)); idx += 1
    vec[idx] = float(demo.get("snp_flag", False)); idx += 1

    # Clinical vitals (6)
    vec[idx] = clinical.get("bp_systolic_last", 120) / 200.0; idx += 1
    vec[idx] = clinical.get("bp_diastolic_last", 80) / 120.0; idx += 1
    vec[idx] = clinical.get("a1c_last", 5.5) / 14.0; idx += 1
    vec[idx] = clinical.get("bmi", 25) / 55.0; idx += 1
    vec[idx] = clinical.get("ckd_stage", 0) / 5.0; idx += 1
    vec[idx] = clinical.get("phq9_score", 0) / 27.0; idx += 1

    # Condition flags (8)
    conditions = clinical.get("conditions", {})
    for cond in ["diabetes", "hypertension", "hyperlipidemia", "depression",
                 "ckd", "chd", "copd", "chf"]:
        vec[idx] = float(conditions.get(cond, False)); idx += 1

    # Medication fill rates (4)
    vec[idx] = med_fills.get("statin", 0.0); idx += 1
    vec[idx] = med_fills.get("ace_arb", 0.0); idx += 1
    vec[idx] = med_fills.get("diabetes_oral", 0.0); idx += 1
    vec[idx] = med_fills.get("antidepressant", 0.0); idx += 1

    # Open gap flags (18)
    open_gaps = set(snapshot.get("open_gaps", []))
    closed_gaps = set(snapshot.get("closed_gaps", []))
    for m in HEDIS_MEASURES:
        vec[idx] = 1.0 if m in open_gaps else 0.0; idx += 1

    # Risk scores (4)
    vec[idx] = risk.get("readmission_risk", 0.0); idx += 1
    vec[idx] = risk.get("disenrollment_risk", 0.0); idx += 1
    vec[idx] = risk.get("non_compliance_risk", 0.0); idx += 1
    vec[idx] = risk.get("composite_acuity", 0.0) / 5.0; idx += 1

    # Channel availability (4)
    vec[idx] = float(engagement.get("sms_consent", False)); idx += 1
    vec[idx] = float(engagement.get("email_available", False)); idx += 1
    vec[idx] = float(engagement.get("portal_registered", False)); idx += 1
    vec[idx] = float(engagement.get("app_installed", False)); idx += 1

    # Measure-relevant clinical indicators (36 = 2 per HEDIS measure)
    # These give the model per-measure clinical context about this patient.
    mcl = snapshot.get("measure_clinical", {})
    age = demo.get("age", 65)
    has_diab = float(conditions.get("diabetes", False))
    has_hyp = float(conditions.get("hypertension", False))
    has_lipid = float(conditions.get("hyperlipidemia", False))
    has_dep = float(conditions.get("depression", False))
    has_ckd_flag = float(conditions.get("ckd", False))
    has_chd = float(conditions.get("chd", False))
    chronic_count = sum(1 for c in conditions.values() if c) / 8.0

    for m in HEDIS_MEASURES:
        feat1_name, feat2_name = MEASURE_CLINICAL_FEATURES[m]
        # Feature 1: primary clinical indicator
        if feat1_name == "age_over_50":
            vec[idx] = 1.0 if age >= 50 else 0.0
        elif feat1_name == "age_over_65":
            vec[idx] = 1.0 if age >= 65 else 0.0
        elif feat1_name == "sex_female":
            vec[idx] = 1.0 if demo.get("sex") == "F" else 0.0
        elif feat1_name == "has_diabetes":
            vec[idx] = has_diab
        elif feat1_name == "has_hypertension":
            vec[idx] = has_hyp
        elif feat1_name == "has_ckd":
            vec[idx] = has_ckd_flag
        elif feat1_name == "has_depression":
            vec[idx] = has_dep
        elif feat1_name == "has_cardiovascular":
            vec[idx] = max(has_chd, has_lipid)
        elif feat1_name == "has_diabetes_and_hypertension":
            vec[idx] = min(has_diab, has_hyp)
        elif feat1_name == "has_diabetes_and_hyperlipidemia":
            vec[idx] = min(has_diab, has_lipid)
        elif feat1_name == "chronic_condition_count_norm":
            vec[idx] = chronic_count
        elif feat1_name == "immunocompromised":
            vec[idx] = mcl.get("immunocompromised", 0.0)
        elif feat1_name == "recent_discharge":
            vec[idx] = mcl.get("recent_discharge", 0.0)
        else:
            vec[idx] = mcl.get(feat1_name, 0.0)
        idx += 1

        # Feature 2: secondary indicator
        if feat2_name == "bp_controlled":
            vec[idx] = 1.0 if clinical.get("bp_systolic_last", 999) < 140 else 0.0
        elif feat2_name == "a1c_controlled":
            vec[idx] = 1.0 if clinical.get("a1c_last", 999) < 8.0 else 0.0
        elif feat2_name == "statin_adherent":
            vec[idx] = 1.0 if med_fills.get("statin", 0) >= 0.8 else 0.0
        elif feat2_name == "ras_adherent":
            vec[idx] = 1.0 if med_fills.get("ace_arb", 0) >= 0.8 else 0.0
        elif feat2_name == "on_antidepressant":
            vec[idx] = 1.0 if med_fills.get("antidepressant", 0) > 0 else 0.0
        elif feat2_name == "has_diabetes":
            vec[idx] = has_diab
        elif feat2_name == "years_since_eye_exam_norm":
            vec[idx] = min(mcl.get("years_since_eye_exam", 2) / 3.0, 1.0)
        elif feat2_name in ("prior_colonoscopy", "prior_mammogram", "prior_flu_vaccine",
                            "prior_pneumo_vaccine", "prior_mh_followup", "has_pcp"):
            vec[idx] = mcl.get(feat2_name, 0.0)
        else:
            vec[idx] = mcl.get(feat2_name, 0.0)
        idx += 1

    assert idx == TIER1_END, f"Tier 1 wrote {idx} features, expected {TIER1_END}"

    # ═════════════════════════════════════════════════════════════════════
    # TIER 2: System State (20 features, indices 50-69)
    # ═════════════════════════════════════════════════════════════════════
    from config import (
        BUDGET_WARNING_THRESHOLD, BUDGET_CRITICAL_THRESHOLD,
        AVG_MESSAGES_PER_PATIENT,
    )
    if budget_max is None:
        budget_max = AVG_MESSAGES_PER_PATIENT * cohort_size
    if budget_remaining is None:
        budget_remaining = budget_max

    # Budget (6)
    budget_frac = budget_remaining / max(budget_max, 1)
    daily_budget = budget_max / max(YEAR_DAYS, 1)
    vec[idx] = budget_frac; idx += 1                                               # budget_remaining_norm
    vec[idx] = 1.0 - budget_frac; idx += 1                                         # budget_utilization
    vec[idx] = 1.0 if budget_frac < BUDGET_WARNING_THRESHOLD else 0.0; idx += 1    # budget_is_warning
    vec[idx] = 1.0 if budget_frac < BUDGET_CRITICAL_THRESHOLD else 0.0; idx += 1   # budget_is_critical
    vec[idx] = min(budget_daily_spend / max(daily_budget, 1), 3.0) / 3.0; idx += 1 # budget_burn_rate_norm
    if budget_daily_spend > 0:
        projected_days = budget_remaining / budget_daily_spend
        vec[idx] = min(projected_days / YEAR_DAYS, 1.0)
    else:
        vec[idx] = 1.0
    idx += 1  # budget_projected_days_left

    # Temporal (3)
    vec[idx] = np.sin(2 * np.pi * day_of_year / YEAR_DAYS); idx += 1
    vec[idx] = np.cos(2 * np.pi * day_of_year / YEAR_DAYS); idx += 1
    vec[idx] = min(day_of_year / YEAR_DAYS, 1.0); idx += 1  # measurement_year_progress

    # Population (2)
    vec[idx] = min(cohort_size / 10000.0, 1.0); idx += 1            # cohort_size_norm
    vec[idx] = min(cohort_avg_messages / max(AVG_MESSAGES_PER_PATIENT, 1), 3.0) / 3.0; idx += 1  # cohort_avg_messages_norm

    # STARS progress (4)
    vec[idx] = stars_score / 5.0; idx += 1                          # overall_stars_norm
    vec[idx] = np.clip(stars_7d_trend, -1.0, 1.0); idx += 1        # stars_7d_trend
    vec[idx] = pct_measures_above_4; idx += 1                       # pct_measures_above_4star
    vec[idx] = lowest_measure_stars / 5.0; idx += 1                 # lowest_measure_stars_norm

    # Cohort channel effectiveness (5)
    ch_rates = cohort_channel_rates or {}
    vec[idx] = ch_rates.get("sms", 0.0); idx += 1
    vec[idx] = ch_rates.get("email", 0.0); idx += 1
    vec[idx] = ch_rates.get("portal", 0.0); idx += 1
    vec[idx] = ch_rates.get("app", 0.0); idx += 1
    vec[idx] = ch_rates.get("ivr", 0.0); idx += 1

    assert idx == TIER2_END, f"Tier 2 wrote {idx} features, expected {TIER2_END}"

    # ═════════════════════════════════════════════════════════════════════
    # TIER 3: Action Context (58 features, indices 70-127)
    # ═════════════════════════════════════════════════════════════════════
    from config import MAX_CONTACTS_PER_WEEK

    # Contact history (8)
    avg_msg = max(cohort_avg_messages, 1.0)
    vec[idx] = min(patient_messages_received / max(AVG_MESSAGES_PER_PATIENT, 1), 3.0) / 3.0; idx += 1
    vec[idx] = min(patient_messages_received / avg_msg, 3.0) / 3.0 if avg_msg > 0 else 0.0; idx += 1
    vec[idx] = patient_contacts_7d / max(MAX_CONTACTS_PER_WEEK, 1); idx += 1
    vec[idx] = patient_contacts_14d / max(2 * MAX_CONTACTS_PER_WEEK, 1); idx += 1
    vec[idx] = patient_contacts_30d / max(4 * MAX_CONTACTS_PER_WEEK, 1); idx += 1
    vec[idx] = min(patient_days_since_contact, DAYS_SINCE_NORMALIZATION_MAX) / DAYS_SINCE_NORMALIZATION_MAX; idx += 1
    vec[idx] = patient_response_rate; idx += 1
    vec[idx] = patient_channels_used / 5.0; idx += 1

    # Per-channel success rates for THIS patient (5)
    ch_success = patient_channel_success or {}
    vec[idx] = ch_success.get("sms", 0.0); idx += 1
    vec[idx] = ch_success.get("email", 0.0); idx += 1
    vec[idx] = ch_success.get("portal", 0.0); idx += 1
    vec[idx] = ch_success.get("app", 0.0); idx += 1
    vec[idx] = ch_success.get("ivr", 0.0); idx += 1

    # Engagement rates from historical data (5)
    vec[idx] = engagement.get("sms_response_rate", 0.0); idx += 1
    vec[idx] = engagement.get("email_open_rate", 0.0); idx += 1
    vec[idx] = engagement.get("portal_engagement_rate", 0.0); idx += 1
    vec[idx] = engagement.get("app_engagement_rate", 0.0); idx += 1
    vec[idx] = engagement.get("ivr_completion_rate", 0.0); idx += 1

    # Gap context (5)
    num_open = len(open_gaps)
    num_closed = len(closed_gaps)
    # Find highest-weight open gap
    from config import MEASURE_WEIGHTS
    max_weight = 0
    for g in open_gaps:
        max_weight = max(max_weight, MEASURE_WEIGHTS.get(g, 1))

    vec[idx] = num_open / len(HEDIS_MEASURES); idx += 1              # num_open_gaps_norm
    vec[idx] = min(patient_avg_gap_age / YEAR_DAYS, 1.0); idx += 1  # avg_gap_age_norm
    vec[idx] = min(patient_days_since_closure / DAYS_SINCE_NORMALIZATION_MAX, 1.0); idx += 1
    vec[idx] = max_weight / 3.0; idx += 1                           # highest_priority_gap_weight
    vec[idx] = num_closed / len(HEDIS_MEASURES); idx += 1            # num_closed_gaps_norm

    # Pending actions (3)
    vec[idx] = min(num_pending_actions / 5.0, 1.0); idx += 1
    vec[idx] = min(num_in_flight_measures / len(HEDIS_MEASURES), 1.0); idx += 1
    vec[idx] = 1.0 if num_pending_actions > 0 else 0.0; idx += 1

    # Channel affinity (10) — lifetime usage volume + recency per channel
    # Volume: how many times each channel has been used for this patient
    # Recency: how many days since last use of each channel
    # Combined with per-channel success rates (indices 114-118), this gives
    # the model a complete channel affinity profile: volume + recency + effectiveness
    ch_counts = channel_affinity_counts or {}
    ch_recency = channel_affinity_recency or {}
    vec[idx] = min(ch_counts.get("sms", 0), 10) / 10.0; idx += 1
    vec[idx] = min(ch_counts.get("email", 0), 10) / 10.0; idx += 1
    vec[idx] = min(ch_counts.get("portal", 0), 10) / 10.0; idx += 1
    vec[idx] = min(ch_counts.get("app", 0), 10) / 10.0; idx += 1
    vec[idx] = min(ch_counts.get("ivr", 0), 10) / 10.0; idx += 1
    vec[idx] = min(ch_recency.get("sms", 90), 90) / 90.0; idx += 1
    vec[idx] = min(ch_recency.get("email", 90), 90) / 90.0; idx += 1
    vec[idx] = min(ch_recency.get("portal", 90), 90) / 90.0; idx += 1
    vec[idx] = min(ch_recency.get("app", 90), 90) / 90.0; idx += 1
    vec[idx] = min(ch_recency.get("ivr", 90), 90) / 90.0; idx += 1

    # Gap-specific detail (10) — top 5 open gaps
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

    # Per-measure attempt summary (24) — top 3 priority open gaps × 8 features each
    # No index encoding: weight is meaningful ordinal, channel is one-hot
    if measure_attempt_summary is None:
        measure_attempt_summary = []
    for i in range(3):
        if i < len(measure_attempt_summary):
            ms = measure_attempt_summary[i]
            best_ch = ms.get("best_channel", "")
            vec[idx] = MEASURE_WEIGHTS.get(ms.get("measure", ""), 1) / 3.0; idx += 1
            vec[idx] = min(ms.get("attempts", 0), 10) / 10.0; idx += 1
            vec[idx] = min(ms.get("days_since", 90), 90) / 90.0; idx += 1
            vec[idx] = 1.0 if best_ch == "sms" else 0.0; idx += 1
            vec[idx] = 1.0 if best_ch == "email" else 0.0; idx += 1
            vec[idx] = 1.0 if best_ch == "portal" else 0.0; idx += 1
            vec[idx] = 1.0 if best_ch == "app" else 0.0; idx += 1
            vec[idx] = 1.0 if best_ch == "ivr" else 0.0; idx += 1
        else:
            for _ in range(8):
                vec[idx] = 0.0; idx += 1

    assert idx == TIER3_END, f"Tier 3 wrote {idx} features, expected {TIER3_END}"

    return vec
