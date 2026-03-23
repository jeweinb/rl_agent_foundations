"""
Orchestrates generation of all four mock datasets.
"""
import json
import os
import numpy as np
from typing import Dict, Any

from config import COHORT_SIZE, GENERATED_DATA_DIR
from datagen.patients import generate_patients
from datagen.state_features import generate_state_features
from datagen.historical_activity import generate_historical_activity
from datagen.gap_closure import generate_gap_closure
from datagen.action_eligibility import generate_action_eligibility


def generate_all(seed: int = 42, cohort_size: int = COHORT_SIZE) -> Dict[str, Any]:
    """Generate all four datasets and save to disk."""
    rng = np.random.default_rng(seed)
    os.makedirs(GENERATED_DATA_DIR, exist_ok=True)

    print(f"Generating {cohort_size} patient profiles...")
    patients = generate_patients(cohort_size, rng=rng)

    print("Generating state feature snapshots...")
    state_snapshots = generate_state_features(patients, rng=rng)

    print("Generating historical activity records...")
    historical = generate_historical_activity(patients, state_snapshots, rng=rng)

    print("Generating gap closure timelines...")
    gap_closure = generate_gap_closure(state_snapshots, rng=rng)

    print("Generating action eligibility snapshots...")
    eligibility = generate_action_eligibility(state_snapshots, rng=rng)

    # Save all datasets
    datasets = {
        "state_features": state_snapshots,
        "historical_activity": historical,
        "gap_closure": gap_closure,
        "action_eligibility": eligibility,
    }

    for name, data in datasets.items():
        filepath = os.path.join(GENERATED_DATA_DIR, f"{name}.json")
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  Saved {name}: {len(data)} records → {filepath}")

    # Also save raw patient profiles (useful for reference)
    patients_path = os.path.join(GENERATED_DATA_DIR, "patients.json")
    with open(patients_path, "w") as f:
        json.dump(patients, f, indent=2, default=str)
    print(f"  Saved patients: {len(patients)} records → {patients_path}")

    print(f"\nAll datasets generated in {GENERATED_DATA_DIR}/")
    return datasets
