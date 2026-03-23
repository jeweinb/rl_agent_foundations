"""
Lagged reward mechanism.
Handles the delay between actions and gap closure observations.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional

from config import MEASURE_CATEGORIES, LAG_DISTRIBUTIONS


def _get_category(measure: str) -> str:
    for cat, measures in MEASURE_CATEGORIES.items():
        if measure in measures:
            return cat
    return "chronic"


class LaggedRewardQueue:
    """Manages delayed reward observations.

    When an action is taken, the reward model predicts closure probability.
    The actual gap closure event is scheduled to resolve at a future day
    based on measure-specific lag distributions.
    """

    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng or np.random.default_rng()
        self.pending: List[Dict] = []
        self.resolved: List[Dict] = []

    def schedule(
        self,
        current_day: int,
        patient_id: str,
        measure: str,
        action_id: int,
        closure_prob: float,
    ):
        """Schedule a potential gap closure event.

        Args:
            current_day: Current simulation day.
            patient_id: Patient ID.
            measure: HEDIS measure code.
            action_id: Action that was taken.
            closure_prob: Predicted probability of gap closure.
        """
        category = _get_category(measure)
        lag_params = LAG_DISTRIBUTIONS.get(category, LAG_DISTRIBUTIONS["chronic"])

        # Sample lag duration
        lag_days = int(self.rng.normal(lag_params["mean"], (lag_params["max"] - lag_params["min"]) / 4))
        lag_days = max(lag_params["min"], min(lag_params["max"], lag_days))

        # Determine if gap will actually close
        will_close = self.rng.random() < closure_prob

        resolve_day = current_day + lag_days

        self.pending.append({
            "resolve_day": resolve_day,
            "patient_id": patient_id,
            "measure": measure,
            "action_id": action_id,
            "closure_prob": closure_prob,
            "will_close": will_close,
            "reward": 1.0 if will_close else 0.0,
            "scheduled_day": current_day,
        })

    def collect(self, current_day: int) -> List[Dict]:
        """Collect all rewards that have resolved by current_day.

        Returns:
            List of resolved reward records.
        """
        ready = [p for p in self.pending if p["resolve_day"] <= current_day]
        self.pending = [p for p in self.pending if p["resolve_day"] > current_day]
        self.resolved.extend(ready)
        return ready

    def get_pending_count(self) -> int:
        return len(self.pending)

    def get_resolved_count(self) -> int:
        return len(self.resolved)

    def get_all_resolved(self) -> List[Dict]:
        return list(self.resolved)
