"""
Central configuration for the RL Agent Foundations project.
Defines HEDIS measures, channels, action space, reward weights, and hyperparameters.
"""
from collections import namedtuple
from typing import Dict, List, Set

# ---------------------------------------------------------------------------
# HEDIS Measures
# ---------------------------------------------------------------------------
HEDIS_MEASURES = [
    "COL", "BCS", "EED",           # Screenings
    "FVA", "FVO", "AIS", "FLU",    # Vaccines
    "CBP", "BPD", "HBD", "KED",    # Chronic conditions
    "MAC", "MRA", "MDS",           # Medication adherence (triple-weighted)
    "DSF", "DRR", "DMC02",         # Mental health
    "TRC_M",                        # Care coordination
]

MEASURE_CATEGORIES = {
    "screenings": ["COL", "BCS", "EED"],
    "vaccines": ["FVA", "FVO", "AIS", "FLU"],
    "chronic": ["CBP", "BPD", "HBD", "KED"],
    "medication_adherence": ["MAC", "MRA", "MDS"],
    "mental_health": ["DSF", "DRR", "DMC02"],
    "care_coordination": ["TRC_M"],
}

TRIPLE_WEIGHTED: Set[str] = {"MAC", "MRA", "MDS", "DMC02", "TRC_M"}
MEASURE_WEIGHTS: Dict[str, float] = {
    m: (3.0 if m in TRIPLE_WEIGHTED else 1.0) for m in HEDIS_MEASURES
}

MEASURE_DESCRIPTIONS = {
    "COL": "Colorectal Cancer Screening",
    "BCS": "Breast Cancer Screening",
    "EED": "Eye Exam for Patients with Diabetes",
    "FVA": "Adult Immunization Status - Tdap/Td",
    "FVO": "Pneumococcal Vaccination",
    "AIS": "Adult Immunization Status - Zoster",
    "FLU": "Influenza Vaccination",
    "CBP": "Controlling High Blood Pressure",
    "BPD": "Blood Pressure Control for Patients with Diabetes",
    "HBD": "Hemoglobin A1C Control for Patients with Diabetes",
    "KED": "Kidney Health Evaluation for Patients with Diabetes",
    "MAC": "Medication Adherence for Cholesterol (Statins)",
    "MRA": "Medication Adherence for Hypertension (RAS Antagonists)",
    "MDS": "Medication Adherence for Diabetes (Oral Agents)",
    "DSF": "Depression Screening and Follow-Up",
    "DRR": "Depression Remission or Response",
    "DMC02": "Antidepressant Medication Management",
    "TRC_M": "Transitions of Care - Medication Reconciliation",
}

# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------
CHANNELS = ["sms", "email", "portal", "app", "ivr"]

# ---------------------------------------------------------------------------
# Action Space — curated (measure, channel, variant) tuples
# ---------------------------------------------------------------------------
Action = namedtuple("Action", ["action_id", "measure", "channel", "variant", "description"])

def _build_action_catalog() -> List[Action]:
    """Build the curated action catalog. Index 0 = no_action."""
    actions = [Action(0, "NO_ACTION", "none", "none", "Take no action")]
    idx = 1

    # --- Screenings: COL, BCS, EED — 7 actions each ---
    for measure in ["COL", "BCS", "EED"]:
        desc = MEASURE_DESCRIPTIONS[measure]
        templates = [
            ("sms", "scheduling_link", f"SMS with scheduling link for {desc}"),
            ("sms", "incentive_offer", f"SMS with $50 incentive offer for {desc}"),
            ("email", "educational", f"Email with educational content about {desc}"),
            ("email", "scheduling_link", f"Email with embedded scheduler for {desc}"),
            ("portal", "scheduling_link", f"Portal in-app scheduler for {desc}"),
            ("app", "push_reminder", f"App push reminder for {desc}"),
            ("ivr", "appointment_reminder", f"IVR automated call for {desc}"),
        ]
        for channel, variant, description in templates:
            actions.append(Action(idx, measure, channel, variant, description))
            idx += 1

    # --- Vaccines: FVA, FVO, AIS, FLU — 6 actions each ---
    for measure in ["FVA", "FVO", "AIS", "FLU"]:
        desc = MEASURE_DESCRIPTIONS[measure]
        templates = [
            ("sms", "pharmacy_locator", f"SMS with pharmacy locator for {desc}"),
            ("sms", "zero_cost_reminder", f"SMS zero-cost reminder for {desc}"),
            ("email", "educational", f"Email with educational content about {desc}"),
            ("email", "pharmacy_locator", f"Email with pharmacy map for {desc}"),
            ("app", "push_reminder", f"App push reminder for {desc}"),
            ("ivr", "appointment_reminder", f"IVR automated call for {desc}"),
        ]
        for channel, variant, description in templates:
            actions.append(Action(idx, measure, channel, variant, description))
            idx += 1

    # --- Medication Adherence: MAC, MRA, MDS — 8 actions each ---
    for measure in ["MAC", "MRA", "MDS"]:
        desc = MEASURE_DESCRIPTIONS[measure]
        templates = [
            ("sms", "refill_reminder", f"SMS refill reminder for {desc}"),
            ("sms", "mail_order_offer", f"SMS 90-day mail order offer for {desc}"),
            ("email", "refill_reminder", f"Email refill reminder for {desc}"),
            ("email", "side_effect_support", f"Email side effect support for {desc}"),
            ("portal", "adherence_dashboard", f"Portal adherence dashboard for {desc}"),
            ("app", "refill_reminder", f"App push refill reminder for {desc}"),
            ("app", "adherence_gamification", f"App gamified adherence tracker for {desc}"),
            ("ivr", "refill_reminder", f"IVR automated refill call for {desc}"),
        ]
        for channel, variant, description in templates:
            actions.append(Action(idx, measure, channel, variant, description))
            idx += 1

    # --- Chronic Management: CBP, BPD, HBD, KED — 7 actions each ---
    for measure in ["CBP", "BPD", "HBD", "KED"]:
        desc = MEASURE_DESCRIPTIONS[measure]
        templates = [
            ("sms", "appointment_reminder", f"SMS appointment reminder for {desc}"),
            ("sms", "home_device_offer", f"SMS free home monitoring device for {desc}"),
            ("email", "educational", f"Email educational content for {desc}"),
            ("email", "care_mgmt_enrollment", f"Email care management enrollment for {desc}"),
            ("portal", "lab_scheduling", f"Portal lab scheduling for {desc}"),
            ("app", "home_monitoring", f"App home monitoring integration for {desc}"),
            ("ivr", "appointment_reminder", f"IVR automated appointment call for {desc}"),
        ]
        for channel, variant, description in templates:
            actions.append(Action(idx, measure, channel, variant, description))
            idx += 1

    # --- Mental Health: DSF, DRR, DMC02 — 7 actions each ---
    for measure in ["DSF", "DRR", "DMC02"]:
        desc = MEASURE_DESCRIPTIONS[measure]
        templates = [
            ("sms", "screening_reminder", f"SMS screening reminder for {desc}"),
            ("sms", "telehealth_link", f"SMS telehealth link for {desc}"),
            ("email", "resource_guide", f"Email behavioral health resource guide for {desc}"),
            ("email", "medication_support", f"Email medication support info for {desc}"),
            ("portal", "screening_tool", f"Portal screening tool for {desc}"),
            ("app", "telehealth_link", f"App telehealth link for {desc}"),
            ("ivr", "care_navigator", f"IVR care navigator transfer for {desc}"),
        ]
        for channel, variant, description in templates:
            actions.append(Action(idx, measure, channel, variant, description))
            idx += 1

    # --- Care Transitions: TRC_M — 6 actions ---
    desc = MEASURE_DESCRIPTIONS["TRC_M"]
    templates = [
        ("sms", "followup_reminder", f"SMS post-discharge follow-up for {desc}"),
        ("sms", "med_reconciliation", f"SMS medication reconciliation for {desc}"),
        ("email", "discharge_checklist", f"Email discharge checklist for {desc}"),
        ("portal", "med_review_scheduler", f"Portal med review scheduler for {desc}"),
        ("app", "followup_reminder", f"App post-discharge reminder for {desc}"),
        ("ivr", "care_navigator", f"IVR transition nurse transfer for {desc}"),
    ]
    for channel, variant, description in templates:
        actions.append(Action(idx, "TRC_M", channel, variant, description))
        idx += 1

    return actions


ACTION_CATALOG: List[Action] = _build_action_catalog()
NUM_ACTIONS: int = len(ACTION_CATALOG)
ACTION_BY_ID: Dict[int, Action] = {a.action_id: a for a in ACTION_CATALOG}

# Lookup helpers
ACTION_IDS_BY_MEASURE: Dict[str, List[int]] = {}
for a in ACTION_CATALOG:
    ACTION_IDS_BY_MEASURE.setdefault(a.measure, []).append(a.action_id)

ACTION_IDS_BY_CHANNEL: Dict[str, List[int]] = {}
for a in ACTION_CATALOG:
    ACTION_IDS_BY_CHANNEL.setdefault(a.channel, []).append(a.action_id)

# ---------------------------------------------------------------------------
# State Space
# ---------------------------------------------------------------------------
STATE_DIM = 96  # padded

# ---------------------------------------------------------------------------
# Cohort & Simulation
# ---------------------------------------------------------------------------
COHORT_SIZE = 5000
SIMULATION_DAYS = 30
MAX_CONTACTS_PER_WEEK = 3
MIN_DAYS_BETWEEN_SAME_MEASURE = 7

# ---------------------------------------------------------------------------
# Message Budget — global per-patient outreach budget
# ---------------------------------------------------------------------------
# Each patient has a finite message budget per quarter. Once exhausted, the
# patient is suppressed until the next quarter. This forces the agent to learn
# WHEN to stay silent and conserve budget for high-impact moments.
MESSAGE_BUDGET_PER_QUARTER = 12       # Max messages per 90-day quarter
MESSAGE_BUDGET_PER_YEAR = 45          # Hard annual cap
BUDGET_WARNING_THRESHOLD = 0.25       # Warn when <25% budget remaining
BUDGET_CRITICAL_THRESHOLD = 0.10      # Heavy penalty when <10% remaining
BUDGET_REPLENISH_INTERVAL_DAYS = 90   # Quarterly replenishment

# ---------------------------------------------------------------------------
# Reward Weights
# ---------------------------------------------------------------------------
REWARD_WEIGHTS = {
    "gap_closure": 1.0,
    "engagement_deliver": 0.05,
    "engagement_click": 0.1,
    "action_cost": -0.01,
    "fatigue": -0.05,
    # Budget conservation rewards
    "budget_conservation": 0.02,      # Small reward for choosing no_action when budget is low
    "budget_waste": -0.08,            # Penalty for sending low-value messages when budget < 25%
    "budget_critical_penalty": -0.15, # Harsh penalty for any message when budget < 10%
}

# ---------------------------------------------------------------------------
# CQL Hyperparameters
# ---------------------------------------------------------------------------
CQL_CONFIG = {
    "min_q_weight": 5.0,
    "lagrangian": True,
    "bc_iters": 50,
    "cql_iters": 100,
    "lr": 3e-4,
}

# ---------------------------------------------------------------------------
# Lag Distributions (days) per measure category
# ---------------------------------------------------------------------------
LAG_DISTRIBUTIONS = {
    "screenings": {"min": 14, "max": 60, "mean": 30},
    "vaccines": {"min": 1, "max": 14, "mean": 5},
    "chronic": {"min": 7, "max": 45, "mean": 21},
    "medication_adherence": {"min": 30, "max": 90, "mean": 60},
    "mental_health": {"min": 14, "max": 60, "mean": 30},
    "care_coordination": {"min": 3, "max": 30, "mean": 14},
}

# ---------------------------------------------------------------------------
# World Model Hyperparameters
# ---------------------------------------------------------------------------
DYNAMICS_MODEL_CONFIG = {
    "action_embed_dim": 32,
    "hidden_dims": [256, 256],
    "lr": 1e-3,
    "epochs": 50,
    "batch_size": 256,
}

REWARD_MODEL_CONFIG = {
    "action_embed_dim": 32,
    "hidden_dims": [128, 64],
    "lr": 1e-3,
    "epochs": 50,
    "batch_size": 256,
}

# ---------------------------------------------------------------------------
# STARS Target
# ---------------------------------------------------------------------------
STARS_BONUS_THRESHOLD = 4.0

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
DASHBOARD_PORT = 8050
DASHBOARD_UPDATE_INTERVAL_MS = 5000

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = "data"
GENERATED_DATA_DIR = f"{DATA_DIR}/generated"
SIMULATION_DATA_DIR = f"{DATA_DIR}/simulation"
CHECKPOINTS_DIR = "training/checkpoints"
