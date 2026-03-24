"""
HEDIS Gap Closure Gymnasium Environment — Model-Based.

The env uses LEARNED models to approximate the world:
  - Dynamics model: predicts next patient state given (state, action)
  - Reward model: predicts P(gap_closure | state, action, days_elapsed)

This is the true model-based offline RL approach. The agent is evaluated on
a learned approximation of the world, not ground truth. CQL's conservatism
compensates for model error.

For actions/masking, the env uses the real business rules (these are known
constraints, not something to learn).
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple

from config import (
    NUM_ACTIONS, STATE_DIM, HEDIS_MEASURES, MEASURE_WEIGHTS,
    MAX_CONTACTS_PER_WEEK, MIN_DAYS_BETWEEN_SAME_MEASURE,
    FEAT_IDX_GAP_FLAGS_START, compute_global_budget,
)
from environment.action_space import decode_action, is_no_action, get_action_info
from environment.action_masking import compute_action_mask
from environment.reward import compute_reward


class HEDISEnv(gym.Env):
    """
    Model-based HEDIS environment.

    Uses learned dynamics model for state transitions and learned reward model
    for gap closure prediction. Business rules (masking, suppression) use
    ground truth since those are known constraints.
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
        self._budget_max = 0
        self._budget_remaining = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)

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

        cohort = len(self.patient_snapshots)
        self._budget_max = compute_global_budget(cohort)
        self._budget_remaining = self._budget_max

        # Build initial state vector from snapshot
        from environment.state_space import snapshot_to_vector
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

        # --- State transition via LEARNED dynamics model ---
        if not no_act and self.dynamics_model is not None:
            action_arr = np.array([action], dtype=np.int64)
            state_arr = self._state_vector.reshape(1, -1)
            next_state = self.dynamics_model.predict(state_arr, action_arr, add_noise=True)
            self._state_vector = next_state.flatten().astype(np.float32)
        elif not no_act:
            # Fallback: small random perturbation
            self._state_vector += self.np_random.normal(0, 0.01, STATE_DIM).astype(np.float32)
            self._state_vector = np.clip(self._state_vector, -5.0, 5.0)

        # --- Gap closure prediction via LEARNED reward model ---
        # Each step represents ~1 day. Query the model at a per-step horizon.
        # This gives realistic per-interaction closure probabilities.
        gap_closed = False
        if not no_act and measure in self._open_gaps:
            if self.reward_model is not None:
                state_arr = self._state_vector.reshape(1, -1)
                action_arr = np.array([action])
                # Query at 7-day horizon — "will this action lead to closure within a week?"
                days_arr = np.array([7.0], dtype=np.float32)
                closure_prob = float(self.reward_model.predict(state_arr, action_arr, days_arr)[0])
                # Cap to prevent unrealistic 90%+ closure per interaction
                closure_prob = min(closure_prob, 0.15)
            else:
                # Conservative fallback
                from datagen.constants import GAP_CLOSURE_BASE_RATES
                closure_prob = GAP_CLOSURE_BASE_RATES.get(measure, 0.5) * 0.02

            gap_closed = self.np_random.random() < closure_prob
            if gap_closed:
                self._open_gaps.discard(measure)
                # Update gap flags in state vector
                if measure in HEDIS_MEASURES:
                    gap_idx = FEAT_IDX_GAP_FLAGS_START + HEDIS_MEASURES.index(measure)
                    self._state_vector[gap_idx] = 0.0

        # Compute reward
        reward = compute_reward(
            measure=measure,
            clicked=self.np_random.random() < 0.15 if not no_act else False,
            gap_closed=gap_closed,
            is_no_action=no_act,
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

        # Termination
        terminated = len(self._open_gaps) == 0
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
            "delivered": not no_act,
            "opened": not no_act and self.np_random.random() < 0.4,
            "clicked": not no_act and self.np_random.random() < 0.15,
            "gap_closed": gap_closed,
            "open_gaps": list(self._open_gaps),
            "episode_reward": self._episode_reward,
            "day_of_year": self._day_of_year,
            "budget_remaining": self._budget_remaining,
            "budget_max": self._budget_max,
        }
        return obs, reward, terminated, truncated, info

    def _compute_mask(self) -> np.ndarray:
        """Compute action mask using real business rules."""
        engagement = self._current_snapshot.get("engagement", {})
        channel_availability = {
            "sms": engagement.get("sms_consent", False),
            "email": engagement.get("email_available", False),
            "portal": engagement.get("portal_registered", False),
            "app": engagement.get("app_installed", False),
            "ivr": True,
        }
        return compute_action_mask(
            open_gaps=self._open_gaps,
            channel_availability=channel_availability,
            contacts_this_week=self._contacts_this_week,
            recent_measures=self._recent_measures,
            budget_remaining=self._budget_remaining,
        )
