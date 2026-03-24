"""
Generate longitudinal gap closure dataset.
Shows each patient's gap status changing over time across the measurement year.
Used to train the reward model.
"""
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any

from datagen.constants import GAP_CLOSURE_BASE_RATES, GAP_REOPEN_RATES
from config import HEDIS_MEASURES, MEASURE_CATEGORIES


from config import get_measure_category


CLOSURE_EVENT_TYPES = {
    "COL": ["claim_colonoscopy", "claim_cologuard", "claim_fit_test"],
    "BCS": ["claim_mammogram"],
    "EED": ["claim_retinal_exam", "claim_dilated_eye_exam"],
    "FVA": ["claim_tdap_vaccine"],
    "FVO": ["claim_pneumococcal_vaccine"],
    "AIS": ["claim_zoster_vaccine"],
    "FLU": ["claim_flu_vaccine", "pharmacy_flu_vaccine"],
    "CBP": ["claim_bp_reading_controlled"],
    "BPD": ["claim_bp_reading_controlled"],
    "HBD": ["claim_a1c_lab_controlled"],
    "KED": ["claim_kidney_lab"],
    "MAC": ["pharmacy_statin_fill_80pdc"],
    "MRA": ["pharmacy_ras_fill_80pdc"],
    "MDS": ["pharmacy_diabetes_fill_80pdc"],
    "DSF": ["claim_phq9_screening"],
    "DRR": ["claim_phq9_remission"],
    "DMC02": ["pharmacy_antidepressant_6mo"],
    "TRC_M": ["claim_med_reconciliation"],
}


def generate_gap_closure(
    state_snapshots: List[Dict[str, Any]],
    rng: np.random.Generator = None,
) -> List[Dict[str, Any]]:
    """Generate longitudinal gap closure timelines for all patients."""
    if rng is None:
        rng = np.random.default_rng(42)

    year_start = datetime(2026, 1, 1)
    timelines = []

    for snap in state_snapshots:
        pid = snap["patient_id"]
        all_measures = snap["open_gaps"] + snap["closed_gaps"]

        for measure in all_measures:
            starts_open = measure in snap["open_gaps"]
            base_rate = GAP_CLOSURE_BASE_RATES.get(measure, 0.5)
            reopen_rate = GAP_REOPEN_RATES.get(measure, 0.0)

            # Generate monthly timeline
            timeline = []
            gap_open = starts_open
            closure_date = None
            closure_event = None

            # Gaps that start closed already have a closure
            if not starts_open:
                closure_date = year_start.strftime("%Y-%m-%d")
                events = CLOSURE_EVENT_TYPES.get(measure, ["claim_generic"])
                closure_event = rng.choice(events)

            for month in range(12):
                check_date = year_start + timedelta(days=month * 30)
                date_str = check_date.strftime("%Y-%m-%d")

                if gap_open:
                    # Monthly probability of closure (from annual rate)
                    monthly_close_prob = 1 - (1 - base_rate) ** (1 / 12)
                    # Higher probability later in year (urgency)
                    urgency_factor = 1.0 + (month / 12) * 0.5
                    if rng.random() < monthly_close_prob * urgency_factor:
                        gap_open = False
                        closure_date = date_str
                        events = CLOSURE_EVENT_TYPES.get(measure, ["claim_generic"])
                        closure_event = rng.choice(events)
                        timeline.append({
                            "date": date_str,
                            "gap_open": False,
                            "closure_event": closure_event,
                        })
                    else:
                        timeline.append({"date": date_str, "gap_open": True})
                else:
                    # Check for reopening (mainly medication adherence)
                    if rng.random() < reopen_rate:
                        gap_open = True
                        timeline.append({"date": date_str, "gap_open": True, "reopen_event": "adherence_drop"})
                    else:
                        timeline.append({"date": date_str, "gap_open": False})

            final_status = "open" if gap_open else "closed"
            timelines.append({
                "patient_id": pid,
                "measure_year": 2026,
                "measure": measure,
                "timeline": timeline,
                "final_status": final_status,
                "closure_date": closure_date,
                "closure_event": closure_event,
            })

    return timelines
