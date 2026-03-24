"""
HEDIS Gap Closure Gymnasium Environment.

Wraps learned dynamics and reward models into a standard Gymnasium interface.
Used for champion vs challenger evaluation (NOT for CQL training — CQL trains offline).
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from config import (
    NUM_ACTIONS, STATE_DIM, HEDIS_MEASURES,
    MAX_CONTACTS_PER_WEEK, MIN_DAYS_BETWEEN_SAME_MEASURE,
    MEASURE_CATEGORIES, LAG_DISTRIBUTIONS,
    FEAT_IDX_GAP_FLAGS_START,
)
from environment.action_space import decode_action, is_no_action, get_action_info
from environment.state_space import snapshot_to_vector
from environment.action_masking import compute_action_mask
from environment.reward import compute_reward


class HEDISEnv(gym.Env):
    """
    HEDIS Gap Closure Environment.

    Observation: Dict with 'observations' (Box) and 'action_mask' (MultiBinary)
    Action: Discrete(NUM_ACTIONS) — 125 curated actions + no_action
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        patient_snapshots: List[Dict[str, Any]],
        eligibility_snapshots: List[Dict[str, Any]],
        dynamics_model=None,
        reward_model=None,
        max_steps_per_episode: int = 30,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.patient_snapshots = patient_snapshots
        self.eligibility_by_patient = {
            e["patient_id"]: e for e in eligibility_snapshots
        }
        self.dynamics_model = dynamics_model
        self.reward_model = reward_model
        self.max_steps = max_steps_per_episode
        self.render_mode = render_mode

        # Spaces
        self.observation_space = spaces.Dict({
            "observations": spaces.Box(
                low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32
            ),
            "action_mask": spaces.MultiBinary(NUM_ACTIONS),
        })
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # State
        self._patient_idx = 0
        self._current_snapshot = None
        self._state_vector = None
        self._step_count = 0
        self._contacts_this_week = 0
        self._recent_measures: Dict[str, int] = {}
        self._action_history: List[int] = []
        self._open_gaps: set = set()
        self._episode_reward = 0.0
        self._day_of_year = 15
        # Message budget
        from config import MESSAGE_BUDGET_PER_QUARTER
        self._budget_max = MESSAGE_BUDGET_PER_QUARTER
        self._budget_remaining = MESSAGE_BUDGET_PER_QUARTER

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)

        # Select next patient (cycle through cohort)
        if options and "patient_idx" in options:
            self._patient_idx = options["patient_idx"]
        else:
            self._patient_idx = (self._patient_idx + 1) % len(self.patient_snapshots)

        self._current_snapshot = self.patient_snapshots[self._patient_idx].copy()
        self._step_count = 0
        self._contacts_this_week = 0
        self._recent_measures = {}
        self._action_history = []
        self._open_gaps = set(self._current_snapshot.get("open_gaps", []))
        self._episode_reward = 0.0
        self._day_of_year = options.get("day_of_year", 15) if options else 15
        self._budget_remaining = self._budget_max

        self._state_vector = snapshot_to_vector(
            self._current_snapshot,
            action_history=self._action_history,
            day_of_year=self._day_of_year,
            budget_remaining=self._budget_remaining,
            budget_max=self._budget_max,
        )

        obs = {
            "observations": self._state_vector.copy(),
            "action_mask": self._compute_mask(),
        }
        info = {
            "patient_id": self._current_snapshot["patient_id"],
            "open_gaps": list(self._open_gaps),
            "day_of_year": self._day_of_year,
        }
        return obs, info

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        self._step_count += 1
        self._day_of_year += 1

        measure, channel, variant = get_action_info(action)
        no_act = is_no_action(action)

        # Simulate engagement outcome
        delivered, opened, clicked = False, False, False
        if not no_act:
            delivered = self.np_random.random() < self._get_delivery_prob(channel)
            if delivered:
                opened = self.np_random.random() < self._get_open_prob(channel)
                if opened:
                    clicked = self.np_random.random() < self._get_click_prob(channel)

        # Predict gap closure using reward model or heuristic
        gap_closed = False
        if not no_act and measure in self._open_gaps:
            closure_prob = self._predict_closure(measure, channel, clicked)
            gap_closed = self.np_random.random() < closure_prob
            if gap_closed:
                self._open_gaps.discard(measure)

        # Compute reward (budget-aware)
        reward = compute_reward(
            measure=measure,
            delivered=delivered,
            opened=opened,
            clicked=clicked,
            gap_closed=gap_closed,
            contacts_this_week=self._contacts_this_week,
            days_since_same_measure=self._recent_measures.get(measure, 999) if measure else 999,
            is_no_action=no_act,
            budget_remaining=self._budget_remaining,
            budget_max=self._budget_max,
        )
        self._episode_reward += reward

        # Update tracking
        if not no_act:
            self._budget_remaining = max(0, self._budget_remaining - 1)
            self._contacts_this_week += 1
            if self._step_count % 7 == 0:
                self._contacts_this_week = 0
            if measure:
                self._recent_measures[measure] = 0
            self._action_history.insert(0, action)
            self._action_history = self._action_history[:5]

        # Age recent measures
        for m in list(self._recent_measures.keys()):
            self._recent_measures[m] += 1

        # Update state via dynamics model or heuristic
        self._update_state(action, gap_closed)

        # Termination
        terminated = len(self._open_gaps) == 0  # All gaps closed
        truncated = self._step_count >= self.max_steps

        obs = {
            "observations": self._state_vector.copy(),
            "action_mask": self._compute_mask(),
        }
        info = {
            "patient_id": self._current_snapshot["patient_id"],
            "action_id": action,
            "measure": measure,
            "channel": channel,
            "variant": variant,
            "delivered": delivered,
            "opened": opened,
            "clicked": clicked,
            "gap_closed": gap_closed,
            "open_gaps": list(self._open_gaps),
            "episode_reward": self._episode_reward,
            "day_of_year": self._day_of_year,
            "budget_remaining": self._budget_remaining,
            "budget_max": self._budget_max,
        }
        return obs, reward, terminated, truncated, info

    def _compute_mask(self) -> np.ndarray:
        """Compute current action mask."""
        engagement = self._current_snapshot.get("engagement", {})
        channel_availability = {
            "sms": engagement.get("sms_consent", False),
            "email": engagement.get("email_available", False),
            "portal": engagement.get("portal_registered", False),
            "app": engagement.get("app_installed", False),
            "ivr": True,  # IVR always available
        }
        return compute_action_mask(
            open_gaps=self._open_gaps,
            channel_availability=channel_availability,
            contacts_this_week=self._contacts_this_week,
            recent_measures=self._recent_measures,
            budget_remaining=self._budget_remaining,
        )

    def _update_state(self, action: int, gap_closed: bool):
        """Update internal state vector using dynamics model or heuristic."""
        if self.dynamics_model is not None:
            # Use learned dynamics model
            action_tensor = np.array([action], dtype=np.int64)
            state_tensor = self._state_vector.reshape(1, -1)
            next_state = self.dynamics_model.predict(state_tensor, action_tensor)
            self._state_vector = next_state.flatten()
        else:
            # Heuristic state transition
            self._state_vector = snapshot_to_vector(
                self._current_snapshot,
                action_history=self._action_history,
                day_of_year=self._day_of_year,
                budget_remaining=self._budget_remaining,
                budget_max=self._budget_max,
            )
            # Update gap flags in state vector
            gap_start_idx = FEAT_IDX_GAP_FLAGS_START
            for i, m in enumerate(HEDIS_MEASURES):
                self._state_vector[gap_start_idx + i] = 1.0 if m in self._open_gaps else 0.0

    def _predict_closure(self, measure: str, channel: str, clicked: bool) -> float:
        """Predict gap closure probability."""
        if self.reward_model is not None:
            # Use learned reward model
            state_tensor = self._state_vector.reshape(1, -1)
            # Would call reward_model.predict(state, action, days) here
            return 0.05  # Placeholder
        else:
            # Heuristic closure probability
            from datagen.constants import GAP_CLOSURE_BASE_RATES, OUTREACH_LIFT
            base = GAP_CLOSURE_BASE_RATES.get(measure, 0.5)
            lift = OUTREACH_LIFT.get(channel, 1.0)
            daily_prob = 1 - (1 - base * lift) ** (1 / 365)
            if clicked:
                daily_prob *= 3.0
            return min(daily_prob * 5, 0.3)  # Cap at 30% per interaction

    def _get_delivery_prob(self, channel: str) -> float:
        from datagen.constants import DELIVERY_RATES
        return DELIVERY_RATES.get(channel, 0.9)

    def _get_open_prob(self, channel: str) -> float:
        from datagen.constants import OPEN_RATES
        return OPEN_RATES.get(channel, 0.3)

    def _get_click_prob(self, channel: str) -> float:
        from datagen.constants import CLICK_RATES
        return CLICK_RATES.get(channel, 0.1)
