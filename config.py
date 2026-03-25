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

# CMS 2026 Star Ratings measure weights
# Based on CMS measure categorization:
#   Outcome/Intermediate Outcome = 3
#   Patient experience/access = 2 (reduced from 4)
#   Process/New measures = 1
# Note: MAC, MRA, MDS are Part D measures, temporarily weight 1 for 2026,
#       returning to weight 3 in 2027+. We model them at weight 3 for
#       forward-looking FY2028 (MY2025-2026) optimization.
MEASURE_WEIGHTS: Dict[str, float] = {
    "COL": 1,     # Process (new specifications for 2026)
    "BCS": 3,     # Outcome
    "EED": 3,     # Outcome (retiring 2029)
    "FVA": 1,     # Process
    "FVO": 1,     # Process
    "AIS": 1,     # Process
    "FLU": 1,     # Process
    "CBP": 3,     # Intermediate Outcome
    "BPD": 3,     # Intermediate Outcome
    "HBD": 3,     # Intermediate Outcome
    "KED": 1,     # New measure for 2026
    "MAC": 3,     # Part D Intermediate Outcome (weight 1 in 2026, 3 in 2027+)
    "MRA": 3,     # Part D Intermediate Outcome (weight 1 in 2026, 3 in 2027+)
    "MDS": 3,     # Part D Intermediate Outcome (weight 1 in 2026, 3 in 2027+)
    "DSF": 1,     # Process
    "DRR": 1,     # Outcome (new)
    "DMC02": 3,   # Intermediate Outcome
    "TRC_M": 1,   # Process
}

# CMS Star Rating cut points — thresholds for each star level per measure
# Based on 2026 Star Ratings cut points (approximate, from CMS published data)
# Format: {measure: {2: threshold, 3: threshold, 4: threshold, 5: threshold}}
MEASURE_CUT_POINTS: Dict[str, Dict[int, float]] = {
    "COL": {2: 0.50, 3: 0.60, 4: 0.70, 5: 0.78},
    "BCS": {2: 0.60, 3: 0.70, 4: 0.78, 5: 0.84},
    "EED": {2: 0.55, 3: 0.65, 4: 0.75, 5: 0.86},
    "FVA": {2: 0.30, 3: 0.45, 4: 0.55, 5: 0.65},
    "FVO": {2: 0.40, 3: 0.55, 4: 0.65, 5: 0.75},
    "AIS": {2: 0.25, 3: 0.35, 4: 0.45, 5: 0.55},
    "FLU": {2: 0.55, 3: 0.65, 4: 0.73, 5: 0.80},
    "CBP": {2: 0.55, 3: 0.65, 4: 0.78, 5: 0.86},
    "BPD": {2: 0.50, 3: 0.60, 4: 0.72, 5: 0.82},
    "HBD": {2: 0.65, 3: 0.75, 4: 0.84, 5: 0.91},
    "KED": {2: 0.35, 3: 0.48, 4: 0.62, 5: 0.74},
    "MAC": {2: 0.78, 3: 0.84, 4: 0.89, 5: 0.93},
    "MRA": {2: 0.78, 3: 0.84, 4: 0.89, 5: 0.93},
    "MDS": {2: 0.76, 3: 0.82, 4: 0.87, 5: 0.92},
    "DSF": {2: 0.40, 3: 0.55, 4: 0.65, 5: 0.75},
    "DRR": {2: 0.20, 3: 0.30, 4: 0.40, 5: 0.50},
    "DMC02": {2: 0.45, 3: 0.55, 4: 0.65, 5: 0.75},
    "TRC_M": {2: 0.35, 3: 0.50, 4: 0.60, 5: 0.72},
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
# Channel index mapping
# ---------------------------------------------------------------------------
CHANNEL_INDEX: Dict[str, int] = {ch: i for i, ch in enumerate(CHANNELS)}
CHANNEL_INDEX["none"] = -1

# ---------------------------------------------------------------------------
# Measure category lookup
# ---------------------------------------------------------------------------
def get_measure_category(measure: str) -> str:
    """Get the category for a HEDIS measure. Public utility used across modules."""
    for cat, measures in MEASURE_CATEGORIES.items():
        if measure in measures:
            return cat
    return "chronic"

# ---------------------------------------------------------------------------
# Best channel per measure category (learnable pattern in the simulation)
# ---------------------------------------------------------------------------
BEST_CHANNEL_BY_CATEGORY: Dict[str, str] = {
    "screenings": "sms",
    "vaccines": "sms",
    "chronic": "app",
    "medication_adherence": "app",
    "mental_health": "portal",
    "care_coordination": "ivr",
}

SECOND_BEST_CHANNEL_BY_CATEGORY: Dict[str, str] = {
    "screenings": "email",
    "vaccines": "email",
    "chronic": "sms",
    "medication_adherence": "sms",
    "mental_health": "sms",
    "care_coordination": "sms",
}

# ---------------------------------------------------------------------------
# State Space
# ---------------------------------------------------------------------------
STATE_DIM = 96  # padded

# ---------------------------------------------------------------------------
# Cohort & Simulation
# ---------------------------------------------------------------------------
COHORT_SIZE = 5000
SIMULATION_DAYS = 90
MAX_CONTACTS_PER_WEEK = 3  # Business rule: max 3 messages per 7-day rolling window
MIN_DAYS_BETWEEN_SAME_MEASURE = 7
MIN_DAYS_BETWEEN_EMAIL = 30  # Email suppression: max 1 email per 30 days

# ---------------------------------------------------------------------------
# Feature vector index constants (for models that manipulate raw state vectors)
# Layout: demographics(6) + clinical(6) + conditions(8) + meds(4) + gaps(18)
#        + engagement(11) + risk(4) + budget(4) + temporal(3) + action_hist(10)
#        + gap_specific(10) + padding
# ---------------------------------------------------------------------------
FEAT_IDX_GAP_FLAGS_START = 24   # 6+6+8+4
FEAT_IDX_ENGAGEMENT_START = 42  # 24+18
FEAT_IDX_RISK_START = 53        # 42+11
FEAT_IDX_BUDGET_START = 57      # 53+4
FEAT_IDX_TEMPORAL_START = 67    # 57+10 (global budget 4 + patient context 6)

# Normalization constants
CONTACT_NORMALIZATION_MAX = 20
DAYS_SINCE_NORMALIZATION_MAX = 90
YEAR_DAYS = 365

# ---------------------------------------------------------------------------
# Gap closure base rates (annual, without intervention) — domain knowledge
# These represent the baseline probability a gap closes in a year without outreach.
# In production, these could be estimated from historical claims data.
# ---------------------------------------------------------------------------
GAP_CLOSURE_BASE_RATES: Dict[str, float] = {
    "COL": 0.55, "BCS": 0.65, "EED": 0.50,
    "FVA": 0.40, "FVO": 0.50, "AIS": 0.30, "FLU": 0.60,
    "CBP": 0.62, "BPD": 0.58, "HBD": 0.52, "KED": 0.45,
    "MAC": 0.70, "MRA": 0.72, "MDS": 0.68,
    "DSF": 0.35, "DRR": 0.30, "DMC02": 0.55,
    "TRC_M": 0.48,
}

# Channel outreach lift factors (multiplicative boost on base rate)
OUTREACH_LIFT: Dict[str, float] = {
    "sms": 1.15, "email": 1.08, "portal": 1.12, "app": 1.18, "ivr": 1.05,
}

# ---------------------------------------------------------------------------
# Gap closure probability factors (per-interaction dynamics)
# ---------------------------------------------------------------------------
CLOSURE_BASE_MULTIPLIER = 0.30        # base_rate * this * archetype factors = per-interaction closure prob
CLOSURE_BEST_CHANNEL_FACTOR = 2.5     # Multiplied when using best channel for measure
CLOSURE_CLICKED_FACTOR = 3.0          # Multiplied if patient clicked
CLOSURE_OPENED_FACTOR = 1.5           # Multiplied if patient opened/viewed
CLOSURE_DELIVERED_FACTOR = 1.1        # Multiplied if delivered
CLOSURE_PROB_CAP = 0.5                # Maximum per-interaction closure probability

# ---------------------------------------------------------------------------
# Message Budget — GLOBAL shared pool across all patients
# ---------------------------------------------------------------------------
# The organization has a total message budget = AVG_MESSAGES_PER_PATIENT × cohort_size.
# The agent decides how to ALLOCATE this pool — some patients may get 20+ messages
# (high-value, responsive), others get 2 (unresponsive, already closed).
# This creates a resource allocation problem: don't waste messages on patients
# who won't respond; concentrate on high-value targets.
AVG_MESSAGES_PER_PATIENT = 30         # Average messages per patient per quarter (150k total for 5k cohort)
BUDGET_WARNING_THRESHOLD = 0.25       # Warn when <25% of global budget remaining
BUDGET_CRITICAL_THRESHOLD = 0.10      # Heavy penalty when <10% remaining

def compute_global_budget(cohort_size: int) -> int:
    """Total message budget for the quarter = avg × cohort."""
    return AVG_MESSAGES_PER_PATIENT * cohort_size

# ---------------------------------------------------------------------------
# Reward Weights
# ---------------------------------------------------------------------------
# Simplified reward: gap closure is the dominant signal.
# Small shaping rewards guide the agent but never overpower gap closure.
REWARD_WEIGHTS = {
    "gap_closure": 1.0,              # The real objective (×1 or ×3 for weighted measures)
    "engagement_click": 0.05,        # Small bonus for patient engagement
    "channel_diversity": 0.01,       # Small bonus for using an underutilized channel
}

# ---------------------------------------------------------------------------
# CQL Hyperparameters
# ---------------------------------------------------------------------------
CQL_CONFIG = {
    "min_q_weight": 1.0,     # Reduced from 5.0 — less aggressive conservatism
    "lagrangian": True,
    "bc_iters": 50,
    "cql_iters": 100,
    "lr": 3e-4,
}

# ---------------------------------------------------------------------------
# Lag Distributions (days) per measure category
# ---------------------------------------------------------------------------
# Lag distributions (days between action and gap closure observation).
# Shorter lags so results are visible within the first 2-3 weeks of simulation.
LAG_DISTRIBUTIONS = {
    "screenings": {"min": 2, "max": 10, "mean": 5},
    "vaccines": {"min": 1, "max": 5, "mean": 2},
    "chronic": {"min": 2, "max": 8, "mean": 4},
    "medication_adherence": {"min": 3, "max": 12, "mean": 6},
    "mental_health": {"min": 2, "max": 10, "mean": 5},
    "care_coordination": {"min": 1, "max": 5, "mean": 3},
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
