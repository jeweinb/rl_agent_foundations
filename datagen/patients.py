"""
Generate synthetic patient profiles using archetype-based behavioral segments.
Each patient is assigned to one of 12 archetypes that define their clinical
profile, channel preferences, and responsiveness patterns.
"""
import numpy as np
from typing import List, Dict, Any

from datagen.constants import (
    AGE_RANGE, SEX_DISTRIBUTION, DUAL_ELIGIBLE_RATE, LIS_RATE, SNP_RATE,
    ZIP3_OPTIONS, sample_truncated_normal,
)
from datagen.archetypes import assign_archetype


def generate_patients(n: int, rng: np.random.Generator = None) -> List[Dict[str, Any]]:
    """Generate n synthetic patient profiles with archetype assignments."""
    if rng is None:
        rng = np.random.default_rng(42)

    patients = []
    sexes = list(SEX_DISTRIBUTION.keys())
    sex_probs = list(SEX_DISTRIBUTION.values())

    for i in range(n):
        patient_id = f"P{10000 + i:05d}"
        sex = rng.choice(sexes, p=sex_probs)

        # Assign archetype — this drives all behavioral patterns
        archetype = assign_archetype(rng)

        # Age from archetype distribution
        age = int(sample_truncated_normal(
            archetype["demographics"]["age_mean"],
            archetype["demographics"]["age_std"],
            AGE_RANGE[0], AGE_RANGE[1], rng=rng
        )[0])

        # Conditions from archetype prevalence (not population-level)
        conditions = {}
        for cond, prev in archetype["conditions"].items():
            conditions[cond] = bool(rng.random() < prev)

        # Co-condition correlations
        if conditions.get("diabetes"):
            if rng.random() < 0.6:
                conditions["hypertension"] = True
            if rng.random() < 0.5:
                conditions["hyperlipidemia"] = True

        patient = {
            "patient_id": patient_id,
            "age": age,
            "sex": sex,
            "zip3": rng.choice(ZIP3_OPTIONS),
            "dual_eligible": bool(rng.random() < DUAL_ELIGIBLE_RATE),
            "lis_status": bool(rng.random() < LIS_RATE),
            "snp_flag": bool(rng.random() < SNP_RATE),
            "conditions": conditions,
            "archetype": archetype["name"],
            # Channel availability from archetype
            "sms_consent": bool(rng.random() < archetype.get("sms_consent", 0.72)),
            "email_available": bool(rng.random() < archetype.get("email_available", 0.85)),
            "portal_registered": bool(rng.random() < archetype.get("portal_registered", 0.45)),
            "app_installed": bool(rng.random() < archetype.get("app_installed", 0.30)),
            # Behavioral parameters (stored for use by state_features and historical_activity)
            "channel_affinity": archetype["channel_affinity"],
            "channel_engagement": archetype["channel_engagement"],
            "overall_responsiveness": archetype["overall_responsiveness"],
            "timing_optimal_days": archetype["timing_optimal_days"],
            "timing_decay": archetype["timing_decay"],
            "gap_closure_boost": archetype["gap_closure_boost"],
            "variant_boost": archetype.get("variant_boost", {}),
        }
        patients.append(patient)

    return patients
