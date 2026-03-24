"""
Generate state feature snapshots for each patient.
These represent the patient's observable state at a point in time.
"""
import numpy as np
from typing import List, Dict, Any

from datagen.constants import (
    BP_SYSTOLIC, BP_DIASTOLIC, A1C, BMI, PHQ9,
    CKD_STAGE_DISTRIBUTION,
    MED_FILL_RATE,
    SMS_CONSENT_RATE, EMAIL_AVAILABLE_RATE,
    PORTAL_REGISTERED_RATE, APP_INSTALLED_RATE,
    CHANNEL_RESPONSE_RATES,
    RISK_SCORE_RANGES,
    GAP_CLOSURE_BASE_RATES,
    sample_truncated_normal,
)
from config import HEDIS_MEASURES


def _generate_open_gaps(patient: Dict, rng: np.random.Generator) -> tuple:
    """Determine which gaps are open/closed for a patient based on conditions and base rates."""
    eligible_measures = []
    for m in HEDIS_MEASURES:
        # Determine eligibility based on conditions
        if m in ("EED", "HBD", "KED", "MDS", "BPD") and not patient["conditions"].get("diabetes", False):
            continue
        if m in ("CBP", "MRA") and not patient["conditions"].get("hypertension", False):
            continue
        if m in ("MAC", ) and not patient["conditions"].get("hyperlipidemia", False):
            continue
        if m in ("DSF", "DRR", "DMC02") and not patient["conditions"].get("depression", False):
            # Anyone can be screened for depression, but remission/med mgmt needs diagnosis
            if m != "DSF":
                continue
        if m == "BCS" and patient["sex"] != "F":
            continue
        if m == "TRC_M":
            # Only eligible if recent hospitalization — ~15% of cohort
            if rng.random() > 0.15:
                continue
        eligible_measures.append(m)

    open_gaps = []
    closed_gaps = []
    for m in eligible_measures:
        base_rate = GAP_CLOSURE_BASE_RATES.get(m, 0.5)
        # At start of year, most gaps are open; adjust by how far into the year
        gap_is_open = rng.random() > base_rate * 0.3  # ~70-85% open at snapshot
        if gap_is_open:
            open_gaps.append(m)
        else:
            closed_gaps.append(m)

    return open_gaps, closed_gaps


def generate_state_features(
    patients: List[Dict[str, Any]],
    snapshot_date: str = "2026-01-15",
    rng: np.random.Generator = None,
) -> List[Dict[str, Any]]:
    """Generate state feature snapshots for all patients."""
    if rng is None:
        rng = np.random.default_rng(42)

    snapshots = []
    ckd_stages = list(CKD_STAGE_DISTRIBUTION.keys())
    ckd_probs = list(CKD_STAGE_DISTRIBUTION.values())

    for patient in patients:
        pid = patient["patient_id"]
        conditions = patient["conditions"]

        # Clinical vitals — condition-adjusted
        bp_sys = float(sample_truncated_normal(
            BP_SYSTOLIC["mean"] + (10 if conditions.get("hypertension") else 0),
            BP_SYSTOLIC["std"], BP_SYSTOLIC["min"], BP_SYSTOLIC["max"], rng=rng
        )[0])
        bp_dia = float(sample_truncated_normal(
            BP_DIASTOLIC["mean"] + (5 if conditions.get("hypertension") else 0),
            BP_DIASTOLIC["std"], BP_DIASTOLIC["min"], BP_DIASTOLIC["max"], rng=rng
        )[0])
        a1c = float(sample_truncated_normal(
            A1C["mean"] + (1.5 if conditions.get("diabetes") else -1.0),
            A1C["std"], A1C["min"], A1C["max"], rng=rng
        )[0])
        bmi = float(sample_truncated_normal(
            BMI["mean"], BMI["std"], BMI["min"], BMI["max"], rng=rng
        )[0])
        ckd_stage = int(rng.choice(ckd_stages, p=ckd_probs)) if conditions.get("ckd") else 0
        phq9 = int(sample_truncated_normal(
            PHQ9["mean"] + (8 if conditions.get("depression") else 0),
            PHQ9["std"], PHQ9["min"], PHQ9["max"], rng=rng
        )[0])

        # Medication fill rates
        med_fills = {}
        for med, params in MED_FILL_RATE.items():
            # Only relevant if patient has the condition
            relevant = (
                (med == "statin" and conditions.get("hyperlipidemia"))
                or (med == "ace_arb" and conditions.get("hypertension"))
                or (med == "diabetes_oral" and conditions.get("diabetes"))
                or (med == "antidepressant" and conditions.get("depression"))
            )
            if relevant:
                med_fills[med] = float(np.clip(rng.normal(params["mean"], params["std"]), 0.0, 1.0))
            else:
                med_fills[med] = 0.0

        # Open/closed gaps
        open_gaps, closed_gaps = _generate_open_gaps(patient, rng)

        # Engagement — driven by patient's archetype channel affinity
        sms_consent = patient.get("sms_consent", bool(rng.random() < SMS_CONSENT_RATE))
        email_available = patient.get("email_available", bool(rng.random() < EMAIL_AVAILABLE_RATE))
        portal_registered = patient.get("portal_registered", bool(rng.random() < PORTAL_REGISTERED_RATE))
        app_installed = patient.get("app_installed", bool(rng.random() < APP_INSTALLED_RATE))

        # Use archetype channel engagement rates (with noise) instead of population averages
        ch_eng = patient.get("channel_engagement", {})

        # Preferred channel = highest affinity channel that's available
        ch_aff = patient.get("channel_affinity", {})
        avail_channels = []
        if sms_consent: avail_channels.append(("sms", ch_aff.get("sms", 0.3)))
        if email_available: avail_channels.append(("email", ch_aff.get("email", 0.2)))
        if portal_registered: avail_channels.append(("portal", ch_aff.get("portal", 0.1)))
        if app_installed: avail_channels.append(("app", ch_aff.get("app", 0.1)))
        avail_channels.append(("ivr", ch_aff.get("ivr", 0.15)))
        preferred = max(avail_channels, key=lambda x: x[1])[0] if avail_channels else "sms"

        engagement = {
            "sms_consent": sms_consent,
            "email_available": email_available,
            "portal_registered": portal_registered,
            "app_installed": app_installed,
            "preferred_channel": preferred,
            "total_contacts_90d": int(rng.poisson(4)),
            "sms_response_rate": float(np.clip(
                rng.normal(ch_eng.get("sms", 0.3), 0.1), 0, 1
            )) if sms_consent else 0.0,
            "email_open_rate": float(np.clip(
                rng.normal(ch_eng.get("email", 0.2), 0.1), 0, 1
            )) if email_available else 0.0,
            "portal_engagement_rate": float(np.clip(
                rng.normal(ch_eng.get("portal", 0.15), 0.1), 0, 1
            )) if portal_registered else 0.0,
            "app_engagement_rate": float(np.clip(
                rng.normal(ch_eng.get("app", 0.15), 0.1), 0, 1
            )) if app_installed else 0.0,
            "ivr_completion_rate": float(np.clip(
                rng.normal(ch_eng.get("ivr", 0.15), 0.1), 0, 1
            )),
            "last_contact_date": snapshot_date,
            "days_since_last_contact": int(rng.integers(1, 60)),
        }

        # Risk scores
        risk_scores = {}
        for score_name, params in RISK_SCORE_RANGES.items():
            risk_scores[score_name] = float(np.clip(
                rng.normal(params["mean"], params["std"]),
                params["min"], params["max"]
            ))

        snapshot = {
            "patient_id": pid,
            "snapshot_date": snapshot_date,
            "demographics": {
                "age": patient["age"],
                "sex": patient["sex"],
                "zip3": patient["zip3"],
                "dual_eligible": patient["dual_eligible"],
                "lis_status": patient["lis_status"],
                "snp_flag": patient["snp_flag"],
            },
            "clinical": {
                "bp_systolic_last": round(bp_sys, 1),
                "bp_diastolic_last": round(bp_dia, 1),
                "a1c_last": round(a1c, 1),
                "bmi": round(bmi, 1),
                "ckd_stage": ckd_stage,
                "phq9_score": phq9,
                "conditions": conditions,
            },
            "medication_fill_rates": {
                "statin": round(med_fills["statin"], 3),
                "ace_arb": round(med_fills["ace_arb"], 3),
                "diabetes_oral": round(med_fills["diabetes_oral"], 3),
                "antidepressant": round(med_fills["antidepressant"], 3),
            },
            "open_gaps": open_gaps,
            "closed_gaps": closed_gaps,
            "engagement": engagement,
            "risk_scores": risk_scores,
        }
        snapshots.append(snapshot)

    return snapshots
