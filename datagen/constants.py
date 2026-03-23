"""
Clinical ranges, demographic distributions, and data generation parameters.
Used by all datagen modules to produce realistic mock healthcare data.
"""
import numpy as np

# ---------------------------------------------------------------------------
# Demographics
# ---------------------------------------------------------------------------
AGE_RANGE = (18, 95)
MA_AGE_MEAN = 72  # Medicare Advantage skews older
MA_AGE_STD = 10
SEX_DISTRIBUTION = {"M": 0.44, "F": 0.56}
DUAL_ELIGIBLE_RATE = 0.20
LIS_RATE = 0.25
SNP_RATE = 0.10

# Top ZIP3 prefixes (representative mix)
ZIP3_OPTIONS = [
    "331", "333", "336", "337", "338",  # Florida
    "770", "773", "775", "776", "778",  # Texas
    "400", "401", "402", "403", "404",  # Kentucky
    "606", "607", "608", "609", "610",  # Illinois
    "100", "101", "102", "103", "104",  # New York
    "900", "901", "902", "903", "904",  # California
    "303", "304", "305", "306", "307",  # Georgia
    "481", "482", "483", "484", "485",  # Michigan
]

# ---------------------------------------------------------------------------
# Clinical Ranges
# ---------------------------------------------------------------------------
BP_SYSTOLIC = {"mean": 135, "std": 18, "min": 90, "max": 200}
BP_DIASTOLIC = {"mean": 82, "std": 12, "min": 55, "max": 120}
A1C = {"mean": 7.5, "std": 1.8, "min": 4.0, "max": 14.0}
BMI = {"mean": 29.5, "std": 6.0, "min": 16.0, "max": 55.0}
PHQ9 = {"mean": 5, "std": 5, "min": 0, "max": 27}

CKD_STAGE_DISTRIBUTION = {0: 0.55, 1: 0.10, 2: 0.15, 3: 0.12, 4: 0.06, 5: 0.02}

# ---------------------------------------------------------------------------
# Condition Prevalence (in MA population)
# ---------------------------------------------------------------------------
CONDITION_PREVALENCE = {
    "diabetes": 0.33,
    "hypertension": 0.58,
    "hyperlipidemia": 0.45,
    "depression": 0.18,
    "ckd": 0.25,
    "chd": 0.20,
    "copd": 0.12,
    "chf": 0.10,
}

# ---------------------------------------------------------------------------
# Medication Fill Rates (Proportion of Days Covered)
# ---------------------------------------------------------------------------
MED_FILL_RATE = {
    "statin": {"mean": 0.75, "std": 0.18},
    "ace_arb": {"mean": 0.78, "std": 0.16},
    "diabetes_oral": {"mean": 0.72, "std": 0.20},
    "antidepressant": {"mean": 0.65, "std": 0.22},
}

# ---------------------------------------------------------------------------
# Engagement / Channel Availability
# ---------------------------------------------------------------------------
SMS_CONSENT_RATE = 0.72
EMAIL_AVAILABLE_RATE = 0.85
PORTAL_REGISTERED_RATE = 0.45
APP_INSTALLED_RATE = 0.30

CHANNEL_RESPONSE_RATES = {
    "sms": {"mean": 0.35, "std": 0.20},
    "email": {"mean": 0.20, "std": 0.15},
    "portal": {"mean": 0.40, "std": 0.20},
    "app": {"mean": 0.45, "std": 0.20},
    "ivr": {"mean": 0.15, "std": 0.10},
}

# ---------------------------------------------------------------------------
# Risk Scores
# ---------------------------------------------------------------------------
RISK_SCORE_RANGES = {
    "readmission_risk": {"mean": 0.12, "std": 0.08, "min": 0.0, "max": 1.0},
    "disenrollment_risk": {"mean": 0.08, "std": 0.06, "min": 0.0, "max": 1.0},
    "non_compliance_risk": {"mean": 0.30, "std": 0.18, "min": 0.0, "max": 1.0},
    "composite_acuity": {"mean": 2.5, "std": 1.2, "min": 0.0, "max": 5.0},
}

# ---------------------------------------------------------------------------
# Gap Closure Base Rates (annual, without intervention)
# ---------------------------------------------------------------------------
GAP_CLOSURE_BASE_RATES = {
    "COL": 0.55,
    "BCS": 0.65,
    "EED": 0.50,
    "FVA": 0.40,
    "FVO": 0.50,
    "AIS": 0.30,
    "FLU": 0.60,
    "CBP": 0.62,
    "BPD": 0.58,
    "HBD": 0.52,
    "KED": 0.45,
    "MAC": 0.70,
    "MRA": 0.72,
    "MDS": 0.68,
    "DSF": 0.35,
    "DRR": 0.30,
    "DMC02": 0.55,
    "TRC_M": 0.48,
}

# Lift from outreach (multiplicative factor on base rate)
OUTREACH_LIFT = {
    "sms": 1.15,
    "email": 1.08,
    "portal": 1.12,
    "app": 1.18,
    "ivr": 1.05,
}

# ---------------------------------------------------------------------------
# Action Eligibility Constraint Rates
# ---------------------------------------------------------------------------
OPT_OUT_RATE = 0.05          # Global communication opt-out
GRIEVANCE_HOLD_RATE = 0.02   # Active grievance suppression
SUPPRESSION_RATE = 0.08      # General suppression (recent contact, etc.)

# ---------------------------------------------------------------------------
# Historical Activity Generation
# ---------------------------------------------------------------------------
HISTORICAL_RECORDS = 200000
HISTORICAL_DATE_RANGE = ("2024-01-01", "2025-06-30")

# Behavioral policy channel preferences (what the business has been doing)
BEHAVIORAL_CHANNEL_PROBS = {
    "sms": 0.30,
    "email": 0.35,
    "portal": 0.10,
    "app": 0.08,
    "ivr": 0.17,
}

# Delivery/engagement rates by channel
DELIVERY_RATES = {
    "sms": 0.95,
    "email": 0.92,
    "portal": 1.0,
    "app": 0.98,
    "ivr": 0.70,
}

OPEN_RATES = {
    "sms": 0.82,
    "email": 0.25,
    "portal": 0.60,
    "app": 0.55,
    "ivr": 0.70,  # "answered"
}

CLICK_RATES = {
    "sms": 0.15,
    "email": 0.08,
    "portal": 0.35,
    "app": 0.30,
    "ivr": 0.0,  # no clicks for IVR
}

# ---------------------------------------------------------------------------
# Simulation World Parameters
# ---------------------------------------------------------------------------
DAILY_ACTIONS_PER_PATIENT = 1  # Max actions per patient per day
MEASURE_YEAR_START = "2026-01-01"
MEASURE_YEAR_END = "2026-12-31"

# Gap opening rates (probability a closed gap re-opens next month — mostly 0)
GAP_REOPEN_RATES = {m: 0.0 for m in GAP_CLOSURE_BASE_RATES}
# Medication adherence gaps can reopen if fill rate drops
GAP_REOPEN_RATES.update({"MAC": 0.08, "MRA": 0.07, "MDS": 0.09})


def sample_truncated_normal(mean, std, min_val, max_val, size=1, rng=None):
    """Sample from a truncated normal distribution."""
    if rng is None:
        rng = np.random.default_rng()
    samples = rng.normal(mean, std, size)
    return np.clip(samples, min_val, max_val)
