"""
Simulation metrics computation.
STARS scores, cumulative reward, regret curves, and per-measure tracking.
"""
import numpy as np
from typing import Dict, List, Any

from config import HEDIS_MEASURES, MEASURE_WEIGHTS, STARS_BONUS_THRESHOLD
from environment.reward import compute_stars_score, get_measure_stars_detail


class MetricsTracker:
    """Tracks simulation metrics over time."""

    def __init__(self):
        self.daily_metrics: List[Dict[str, Any]] = []
        self.cumulative_reward = 0.0
        self.cumulative_actions = 0
        self.gap_closures_by_measure: Dict[str, int] = {m: 0 for m in HEDIS_MEASURES}
        self.total_patients_by_measure: Dict[str, int] = {m: 0 for m in HEDIS_MEASURES}
        self.model_versions: List[Dict] = []

    def record_day(
        self,
        day: int,
        daily_reward: float,
        daily_actions: int,
        daily_gap_closures: Dict[str, int],
        daily_total_patients: Dict[str, int],
        champion_score: float = None,
        challenger_score: float = None,
        model_promoted: bool = False,
        model_version: int = 1,
        action_distribution: Dict[str, int] = None,
        state_machine_funnel: Dict[str, int] = None,
        avg_budget_remaining: float = None,
        budget_exhausted_count: int = 0,
    ):
        """Record metrics for one simulated day."""
        self.cumulative_reward += daily_reward
        self.cumulative_actions += daily_actions

        for m, count in daily_gap_closures.items():
            self.gap_closures_by_measure[m] = self.gap_closures_by_measure.get(m, 0) + count
        for m, count in daily_total_patients.items():
            # Set denominator once on first day, don't keep growing it
            if m not in self.total_patients_by_measure or self.total_patients_by_measure[m] == 0:
                self.total_patients_by_measure[m] = count

        # Compute measure closure rates (cumulative closures / initial eligible population)
        closure_rates = {}
        for m in HEDIS_MEASURES:
            total = self.total_patients_by_measure.get(m, 0)
            closed = self.gap_closures_by_measure.get(m, 0)
            closure_rates[m] = min(closed / max(total, 1), 1.0)

        stars_score = compute_stars_score(closure_rates)
        measure_detail = get_measure_stars_detail(closure_rates)

        metrics = {
            "day": day,
            "daily_reward": daily_reward,
            "cumulative_reward": self.cumulative_reward,
            "daily_actions": daily_actions,
            "cumulative_actions": self.cumulative_actions,
            "stars_score": stars_score,
            "above_bonus_threshold": stars_score >= STARS_BONUS_THRESHOLD,
            "measure_closure_rates": closure_rates,
            "measure_detail": measure_detail,
            "champion_score": champion_score,
            "challenger_score": challenger_score,
            "model_promoted": model_promoted,
            "model_version": model_version,
            "action_distribution": action_distribution or {},
            "state_machine_funnel": state_machine_funnel or {},
            "avg_budget_remaining": avg_budget_remaining,
            "budget_exhausted_count": budget_exhausted_count,
        }
        self.daily_metrics.append(metrics)
        return metrics

    def get_stars_trajectory(self) -> List[float]:
        return [m["stars_score"] for m in self.daily_metrics]

    def get_cumulative_reward_curve(self) -> List[float]:
        return [m["cumulative_reward"] for m in self.daily_metrics]

    def get_regret_curve(self, oracle_reward_per_day: float = 5.0) -> List[float]:
        """Compute cumulative regret vs an oracle policy."""
        regret = []
        cumulative_regret = 0.0
        for m in self.daily_metrics:
            cumulative_regret += oracle_reward_per_day - m["daily_reward"]
            regret.append(cumulative_regret)
        return regret

    def get_latest(self) -> Dict[str, Any]:
        if self.daily_metrics:
            return self.daily_metrics[-1]
        return {}

    def to_records(self) -> List[Dict[str, Any]]:
        return list(self.daily_metrics)
