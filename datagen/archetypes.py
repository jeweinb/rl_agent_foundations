"""
Patient Archetypes — 12 behavioral segments that define how patients respond
to outreach across channels, measures, and timing.

Each archetype specifies:
- Clinical profile (which conditions/gaps are common)
- Channel affinity (which channels they respond to)
- Engagement intensity (high/med/low responders)
- Timing sensitivity (how spacing affects response)
- Measure responsiveness (which gap closure actions work)

The model must discover these latent segments from the data. The archetypes
create learnable structure: "SMS refill reminders work great for Archetype 3
but terribly for Archetype 7."
"""
from typing import Dict, Any, List


# Each archetype is a dict with behavioral distributions
ARCHETYPES: List[Dict[str, Any]] = [

    # =========================================================================
    # HIGH ENGAGERS — respond well, close gaps, love digital
    # =========================================================================
    {
        "name": "Digital Native",
        "weight": 0.08,  # 8% of cohort
        "description": "Younger MA member, tech-savvy, uses app and portal regularly",
        "demographics": {"age_mean": 55, "age_std": 8},
        "conditions": {"diabetes": 0.20, "hypertension": 0.30, "hyperlipidemia": 0.25,
                       "depression": 0.15, "ckd": 0.08, "chd": 0.05, "copd": 0.04, "chf": 0.03},
        "channel_affinity": {"app": 0.85, "portal": 0.75, "sms": 0.60, "email": 0.40, "ivr": 0.10},
        "channel_engagement": {"app": 0.70, "portal": 0.55, "sms": 0.40, "email": 0.20, "ivr": 0.05},
        "overall_responsiveness": 0.75,
        "timing_optimal_days": 7,   # Responds best with weekly contact
        "timing_decay": 0.3,        # Moderate penalty for too-frequent contact
        "gap_closure_boost": {"screenings": 1.5, "vaccines": 1.3, "chronic": 1.2,
                              "medication_adherence": 1.8, "mental_health": 1.0, "care_coordination": 1.0},
        "portal_registered": 0.95, "app_installed": 0.90, "sms_consent": 0.85, "email_available": 0.95,
    },

    {
        "name": "Proactive Health Manager",
        "weight": 0.10,
        "description": "Engaged member who tracks health metrics, responds to educational content",
        "demographics": {"age_mean": 68, "age_std": 7},
        "conditions": {"diabetes": 0.40, "hypertension": 0.55, "hyperlipidemia": 0.50,
                       "depression": 0.10, "ckd": 0.20, "chd": 0.18, "copd": 0.08, "chf": 0.06},
        "channel_affinity": {"email": 0.80, "portal": 0.65, "sms": 0.55, "app": 0.35, "ivr": 0.30},
        "channel_engagement": {"email": 0.55, "portal": 0.45, "sms": 0.35, "app": 0.25, "ivr": 0.20},
        "overall_responsiveness": 0.70,
        "timing_optimal_days": 10,
        "timing_decay": 0.2,
        "gap_closure_boost": {"screenings": 1.8, "vaccines": 1.6, "chronic": 1.5,
                              "medication_adherence": 1.4, "mental_health": 1.2, "care_coordination": 1.3},
        "portal_registered": 0.70, "app_installed": 0.35, "sms_consent": 0.80, "email_available": 0.95,
    },

    # =========================================================================
    # SMS RESPONDERS — prefer text, moderate engagement
    # =========================================================================
    {
        "name": "SMS-First Responder",
        "weight": 0.12,
        "description": "Responds well to short text messages, ignores email, doesn't use apps",
        "demographics": {"age_mean": 72, "age_std": 9},
        "conditions": {"diabetes": 0.35, "hypertension": 0.60, "hyperlipidemia": 0.45,
                       "depression": 0.12, "ckd": 0.22, "chd": 0.20, "copd": 0.10, "chf": 0.08},
        "channel_affinity": {"sms": 0.90, "ivr": 0.35, "email": 0.15, "portal": 0.10, "app": 0.05},
        "channel_engagement": {"sms": 0.55, "ivr": 0.20, "email": 0.08, "portal": 0.05, "app": 0.02},
        "overall_responsiveness": 0.55,
        "timing_optimal_days": 14,
        "timing_decay": 0.5,         # Very sensitive to over-contact
        "gap_closure_boost": {"screenings": 1.3, "vaccines": 1.6, "chronic": 1.0,
                              "medication_adherence": 1.5, "mental_health": 0.8, "care_coordination": 0.9},
        "portal_registered": 0.20, "app_installed": 0.10, "sms_consent": 0.95, "email_available": 0.80,
    },

    {
        "name": "Refill Reminder Responder",
        "weight": 0.10,
        "description": "Primarily responds to medication refill reminders via SMS, ignores everything else",
        "demographics": {"age_mean": 70, "age_std": 10},
        "conditions": {"diabetes": 0.45, "hypertension": 0.65, "hyperlipidemia": 0.60,
                       "depression": 0.08, "ckd": 0.25, "chd": 0.25, "copd": 0.12, "chf": 0.10},
        "channel_affinity": {"sms": 0.80, "email": 0.20, "ivr": 0.25, "portal": 0.10, "app": 0.15},
        "channel_engagement": {"sms": 0.45, "email": 0.10, "ivr": 0.12, "portal": 0.05, "app": 0.08},
        "overall_responsiveness": 0.45,
        "timing_optimal_days": 14,
        "timing_decay": 0.4,
        "gap_closure_boost": {"screenings": 0.7, "vaccines": 0.8, "chronic": 0.9,
                              "medication_adherence": 2.2, "mental_health": 0.5, "care_coordination": 0.6},
        "portal_registered": 0.25, "app_installed": 0.20, "sms_consent": 0.90, "email_available": 0.85,
    },

    # =========================================================================
    # PHONE/IVR RESPONDERS — older, prefer voice contact
    # =========================================================================
    {
        "name": "Phone Caller",
        "weight": 0.08,
        "description": "Older member who prefers phone calls, answers IVR, distrusts digital",
        "demographics": {"age_mean": 80, "age_std": 6},
        "conditions": {"diabetes": 0.30, "hypertension": 0.70, "hyperlipidemia": 0.40,
                       "depression": 0.20, "ckd": 0.30, "chd": 0.30, "copd": 0.18, "chf": 0.15},
        "channel_affinity": {"ivr": 0.85, "sms": 0.25, "email": 0.10, "portal": 0.05, "app": 0.02},
        "channel_engagement": {"ivr": 0.50, "sms": 0.15, "email": 0.05, "portal": 0.02, "app": 0.01},
        "overall_responsiveness": 0.45,
        "timing_optimal_days": 21,
        "timing_decay": 0.6,         # Hates being contacted too often
        "gap_closure_boost": {"screenings": 1.2, "vaccines": 1.4, "chronic": 1.3,
                              "medication_adherence": 1.0, "mental_health": 1.1, "care_coordination": 1.8},
        "portal_registered": 0.15, "app_installed": 0.05, "sms_consent": 0.75, "email_available": 0.70,
    },

    # =========================================================================
    # CHRONIC CONDITION FOCUSED — respond to condition-specific outreach
    # =========================================================================
    {
        "name": "Diabetic Engager",
        "weight": 0.10,
        "description": "Diabetic patient who responds well to diabetes-specific outreach and monitoring",
        "demographics": {"age_mean": 65, "age_std": 10},
        "conditions": {"diabetes": 0.95, "hypertension": 0.70, "hyperlipidemia": 0.60,
                       "depression": 0.25, "ckd": 0.35, "chd": 0.20, "copd": 0.08, "chf": 0.08},
        "channel_affinity": {"sms": 0.65, "app": 0.55, "email": 0.45, "portal": 0.40, "ivr": 0.30},
        "channel_engagement": {"sms": 0.40, "app": 0.40, "email": 0.25, "portal": 0.30, "ivr": 0.18},
        "overall_responsiveness": 0.60,
        "timing_optimal_days": 7,
        "timing_decay": 0.25,
        "gap_closure_boost": {"screenings": 1.0, "vaccines": 1.0, "chronic": 2.0,
                              "medication_adherence": 1.8, "mental_health": 0.9, "care_coordination": 1.2},
        "portal_registered": 0.50, "app_installed": 0.45, "sms_consent": 0.85, "email_available": 0.80,
    },

    {
        "name": "Cardiac Risk Patient",
        "weight": 0.08,
        "description": "High cardiac risk, responds to BP/cholesterol management, needs care coordination",
        "demographics": {"age_mean": 73, "age_std": 8},
        "conditions": {"diabetes": 0.30, "hypertension": 0.90, "hyperlipidemia": 0.80,
                       "depression": 0.15, "ckd": 0.25, "chd": 0.60, "copd": 0.15, "chf": 0.25},
        "channel_affinity": {"sms": 0.50, "email": 0.55, "ivr": 0.45, "portal": 0.30, "app": 0.20},
        "channel_engagement": {"sms": 0.30, "email": 0.35, "ivr": 0.30, "portal": 0.15, "app": 0.10},
        "overall_responsiveness": 0.50,
        "timing_optimal_days": 14,
        "timing_decay": 0.35,
        "gap_closure_boost": {"screenings": 0.8, "vaccines": 1.0, "chronic": 1.8,
                              "medication_adherence": 1.6, "mental_health": 0.7, "care_coordination": 1.5},
        "portal_registered": 0.30, "app_installed": 0.20, "sms_consent": 0.75, "email_available": 0.85,
    },

    # =========================================================================
    # MENTAL HEALTH SENSITIVE — respond to behavioral health outreach
    # =========================================================================
    {
        "name": "Behavioral Health Seeker",
        "weight": 0.06,
        "description": "Depression/anxiety, responds to telehealth and screening tools, prefers portal privacy",
        "demographics": {"age_mean": 58, "age_std": 12},
        "conditions": {"diabetes": 0.20, "hypertension": 0.35, "hyperlipidemia": 0.25,
                       "depression": 0.85, "ckd": 0.10, "chd": 0.08, "copd": 0.06, "chf": 0.04},
        "channel_affinity": {"portal": 0.80, "app": 0.60, "sms": 0.45, "email": 0.50, "ivr": 0.15},
        "channel_engagement": {"portal": 0.50, "app": 0.40, "sms": 0.25, "email": 0.30, "ivr": 0.08},
        "overall_responsiveness": 0.55,
        "timing_optimal_days": 14,
        "timing_decay": 0.45,
        "gap_closure_boost": {"screenings": 0.9, "vaccines": 0.8, "chronic": 0.9,
                              "medication_adherence": 1.3, "mental_health": 2.5, "care_coordination": 1.0},
        "portal_registered": 0.80, "app_installed": 0.55, "sms_consent": 0.70, "email_available": 0.85,
    },

    # =========================================================================
    # LOW ENGAGERS — hard to reach, require specific approaches
    # =========================================================================
    {
        "name": "Passive Ignorer",
        "weight": 0.10,
        "description": "Rarely responds to any outreach. Best left alone unless high-value opportunity",
        "demographics": {"age_mean": 74, "age_std": 10},
        "conditions": {"diabetes": 0.25, "hypertension": 0.50, "hyperlipidemia": 0.35,
                       "depression": 0.20, "ckd": 0.15, "chd": 0.15, "copd": 0.12, "chf": 0.10},
        "channel_affinity": {"sms": 0.15, "email": 0.10, "ivr": 0.12, "portal": 0.05, "app": 0.03},
        "channel_engagement": {"sms": 0.08, "email": 0.04, "ivr": 0.06, "portal": 0.02, "app": 0.01},
        "overall_responsiveness": 0.12,
        "timing_optimal_days": 30,   # Only respond if given lots of space
        "timing_decay": 0.8,         # Very sensitive to over-contact
        "gap_closure_boost": {"screenings": 0.5, "vaccines": 0.6, "chronic": 0.4,
                              "medication_adherence": 0.5, "mental_health": 0.3, "care_coordination": 0.4},
        "portal_registered": 0.15, "app_installed": 0.08, "sms_consent": 0.70, "email_available": 0.65,
    },

    {
        "name": "Incentive-Motivated",
        "weight": 0.06,
        "description": "Only responds to financial incentives and offers. Ignores educational content.",
        "demographics": {"age_mean": 62, "age_std": 10},
        "conditions": {"diabetes": 0.30, "hypertension": 0.45, "hyperlipidemia": 0.35,
                       "depression": 0.18, "ckd": 0.12, "chd": 0.10, "copd": 0.08, "chf": 0.06},
        "channel_affinity": {"sms": 0.60, "email": 0.45, "app": 0.30, "portal": 0.20, "ivr": 0.15},
        "channel_engagement": {"sms": 0.35, "email": 0.20, "app": 0.15, "portal": 0.10, "ivr": 0.08},
        "overall_responsiveness": 0.35,
        "timing_optimal_days": 14,
        "timing_decay": 0.4,
        # Only responds well to incentive/offer variants
        "gap_closure_boost": {"screenings": 1.8, "vaccines": 1.5, "chronic": 0.6,
                              "medication_adherence": 0.8, "mental_health": 0.4, "care_coordination": 0.5},
        "variant_boost": {"incentive_offer": 3.0, "zero_cost_reminder": 2.0, "mail_order_offer": 2.5,
                          "home_device_offer": 2.0},  # Special: responds to offers/incentives
        "portal_registered": 0.25, "app_installed": 0.20, "sms_consent": 0.80, "email_available": 0.75,
    },

    # =========================================================================
    # SPECIAL POPULATIONS
    # =========================================================================
    {
        "name": "New Enrollee",
        "weight": 0.06,
        "description": "Recently enrolled, many open gaps, high potential but unknown preferences",
        "demographics": {"age_mean": 66, "age_std": 8},
        "conditions": {"diabetes": 0.30, "hypertension": 0.45, "hyperlipidemia": 0.35,
                       "depression": 0.15, "ckd": 0.15, "chd": 0.12, "copd": 0.08, "chf": 0.05},
        "channel_affinity": {"email": 0.50, "sms": 0.50, "portal": 0.30, "app": 0.25, "ivr": 0.30},
        "channel_engagement": {"email": 0.30, "sms": 0.30, "portal": 0.15, "app": 0.12, "ivr": 0.15},
        "overall_responsiveness": 0.50,
        "timing_optimal_days": 7,    # Welcome window — respond well early
        "timing_decay": 0.2,
        "gap_closure_boost": {"screenings": 1.5, "vaccines": 1.5, "chronic": 1.2,
                              "medication_adherence": 1.2, "mental_health": 1.0, "care_coordination": 1.0},
        "extra_open_gaps": 3,        # Start with more open gaps
        "portal_registered": 0.40, "app_installed": 0.30, "sms_consent": 0.75, "email_available": 0.85,
    },

    {
        "name": "Post-Discharge Complex",
        "weight": 0.06,
        "description": "Recently hospitalized, needs care coordination, high readmission risk",
        "demographics": {"age_mean": 78, "age_std": 8},
        "conditions": {"diabetes": 0.40, "hypertension": 0.70, "hyperlipidemia": 0.45,
                       "depression": 0.30, "ckd": 0.35, "chd": 0.35, "copd": 0.25, "chf": 0.30},
        "channel_affinity": {"ivr": 0.70, "sms": 0.45, "email": 0.30, "portal": 0.15, "app": 0.10},
        "channel_engagement": {"ivr": 0.45, "sms": 0.25, "email": 0.15, "portal": 0.08, "app": 0.05},
        "overall_responsiveness": 0.55,
        "timing_optimal_days": 3,    # Need immediate follow-up post-discharge
        "timing_decay": 0.15,        # OK with frequent contact during transition
        "gap_closure_boost": {"screenings": 0.6, "vaccines": 0.7, "chronic": 1.2,
                              "medication_adherence": 1.3, "mental_health": 1.0, "care_coordination": 2.5},
        "trc_m_eligible": 0.80,      # High TRC_M eligibility
        "readmission_risk_boost": 0.25,
        "portal_registered": 0.20, "app_installed": 0.10, "sms_consent": 0.75, "email_available": 0.70,
    },
]


def assign_archetype(rng) -> Dict[str, Any]:
    """Randomly assign a patient to an archetype based on weights."""
    weights = [a["weight"] for a in ARCHETYPES]
    total = sum(weights)
    weights = [w / total for w in weights]
    idx = rng.choice(len(ARCHETYPES), p=weights)
    return ARCHETYPES[idx]


def get_archetype_names() -> List[str]:
    """Return list of archetype names."""
    return [a["name"] for a in ARCHETYPES]
