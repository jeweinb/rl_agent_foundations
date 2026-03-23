"""
Generate synthetic patient profiles for the MA population.
"""
import numpy as np
from typing import List, Dict, Any

from datagen.constants import (
    MA_AGE_MEAN, MA_AGE_STD, AGE_RANGE,
    SEX_DISTRIBUTION, DUAL_ELIGIBLE_RATE, LIS_RATE, SNP_RATE,
    ZIP3_OPTIONS, CONDITION_PREVALENCE,
    sample_truncated_normal,
)


def generate_patients(n: int, rng: np.random.Generator = None) -> List[Dict[str, Any]]:
    """Generate n synthetic patient profiles."""
    if rng is None:
        rng = np.random.default_rng(42)

    patients = []
    sexes = list(SEX_DISTRIBUTION.keys())
    sex_probs = list(SEX_DISTRIBUTION.values())

    ages = sample_truncated_normal(
        MA_AGE_MEAN, MA_AGE_STD, AGE_RANGE[0], AGE_RANGE[1], size=n, rng=rng
    ).astype(int)

    for i in range(n):
        patient_id = f"P{10000 + i:05d}"
        sex = rng.choice(sexes, p=sex_probs)

        conditions = {}
        for cond, prev in CONDITION_PREVALENCE.items():
            conditions[cond] = bool(rng.random() < prev)

        # Age-adjusted condition rates (older = more likely)
        age_factor = max(0.5, min(2.0, ages[i] / 70.0))
        for cond in ["hypertension", "chd", "chf", "copd", "ckd"]:
            if not conditions[cond] and rng.random() < CONDITION_PREVALENCE[cond] * (age_factor - 1) * 0.5:
                conditions[cond] = True

        # Diabetes co-conditions
        if conditions["diabetes"]:
            if rng.random() < 0.6:
                conditions["hypertension"] = True
            if rng.random() < 0.5:
                conditions["hyperlipidemia"] = True

        patient = {
            "patient_id": patient_id,
            "age": int(ages[i]),
            "sex": sex,
            "zip3": rng.choice(ZIP3_OPTIONS),
            "dual_eligible": bool(rng.random() < DUAL_ELIGIBLE_RATE),
            "lis_status": bool(rng.random() < LIS_RATE),
            "snp_flag": bool(rng.random() < SNP_RATE),
            "conditions": conditions,
        }
        patients.append(patient)

    return patients
