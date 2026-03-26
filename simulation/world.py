"""
World Simulator — encapsulates all business rules, patient state, and outcome generation.

The daily cycle asks the world:
1. get_patient_context(pid) → state vector + action mask + metadata
2. execute_action(pid, action_id) → outcome (engagement, closure prob, reward)
3. advance_day() → process lagged rewards, advance state machine, age rolling windows

All suppression rules, budget logic, archetype behavior, and closure probabilities
live here — not scattered across the simulation loop.
"""
import numpy as np
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict

from config import (
    HEDIS_MEASURES, NUM_ACTIONS, ACTION_BY_ID,
    MAX_CONTACTS_PER_WEEK, MIN_DAYS_BETWEEN_SAME_MEASURE,
    CLOSURE_BASE_MULTIPLIER, CLOSURE_CLICKED_FACTOR, CLOSURE_OPENED_FACTOR,
    CLOSURE_DELIVERED_FACTOR, CLOSURE_PROB_CAP, MEASURE_WEIGHTS,
    GAP_CLOSURE_BASE_RATES,
    compute_global_budget, get_measure_category,
    AVG_MESSAGES_PER_PATIENT, BUDGET_WARNING_THRESHOLD, BUDGET_CRITICAL_THRESHOLD,
)
from environment.state_space import snapshot_to_vector
from environment.action_masking import compute_action_mask
from environment.reward import compute_reward
from simulation.action_state_machine import ActionLifecycleTracker, ActionState
from simulation.lagged_rewards import LaggedRewardQueue


class PatientState:
    """Tracks per-patient live state across the simulation."""

    __slots__ = [
        "pid", "snapshot", "messages_sent", "contact_days",
        "channels_used", "responses", "last_closure_day",
        "recent_measures", "last_email_day", "closed_measures",
        "channel_action_counts", "channel_last_day",
    ]

    def __init__(self, pid: str, snapshot: Dict[str, Any]):
        self.pid = pid
        self.snapshot = snapshot
        self.messages_sent = 0
        self.contact_days: List[int] = []   # Rolling 30-day window
        self.channels_used: Set[str] = set()
        self.responses = 0                   # Clicks/accepts
        self.last_email_day = -999            # Day last email was sent
        self.last_closure_day = 0
        self.recent_measures: Dict[str, int] = {}  # measure → day last contacted
        self.closed_measures: Set[str] = set()  # Measures that have been closed
        # Lifetime channel affinity tracking
        self.channel_action_counts: Dict[str, int] = {"sms": 0, "email": 0, "portal": 0, "app": 0, "ivr": 0}
        self.channel_last_day: Dict[str, int] = {"sms": -999, "email": -999, "portal": -999, "app": -999, "ivr": -999}

    @property
    def response_rate(self) -> float:
        return self.responses / max(self.messages_sent, 1)

    def contacts_in_window(self, current_day: int, window: int = 7) -> int:
        """Count contacts within the rolling window."""
        return sum(1 for d in self.contact_days if current_day - d < window)

    def prune_contact_window(self, current_day: int, window: int = 30):
        """Remove contact days outside the rolling window. Keep 30 days for 7d/14d/30d features."""
        self.contact_days = [d for d in self.contact_days if current_day - d < window]

    def age_recent_measures(self, current_day: int):
        """Update days since last contact per measure."""
        # recent_measures stores the day of last contact, compute age on the fly
        pass

    def days_since_measure(self, measure: str, current_day: int) -> int:
        """Days since this measure was last contacted."""
        last = self.recent_measures.get(measure)
        if last is None:
            return MIN_DAYS_BETWEEN_SAME_MEASURE + 1  # Never contacted
        return current_day - last


class WorldSimulator:
    """Encapsulates all business rules, patient state, and outcome generation.

    When dynamics_model and/or reward_model are provided, the simulator uses
    learned models for state transitions and closure probabilities — making it
    suitable for production deployment where ground-truth archetypes don't exist.
    Business rules (budget, contact limits, masking) always use ground truth.
    """

    def __init__(
        self,
        patient_snapshots: List[Dict[str, Any]],
        eligibility_snapshots: List[Dict[str, Any]],
        rng: np.random.Generator = None,
        dynamics_model=None,
        reward_model=None,
    ):
        self.rng = rng or np.random.default_rng()
        self.day = 0

        # Learned models (None = use ground-truth archetype logic)
        self.dynamics_model = dynamics_model
        self.reward_model = reward_model

        # Patient state
        self.patients: Dict[str, PatientState] = {}
        for snap in patient_snapshots:
            self.patients[snap["patient_id"]] = PatientState(snap["patient_id"], snap)

        # Eligibility lookup
        self.eligibility = {e["patient_id"]: e for e in eligibility_snapshots}

        # Global budget
        self.budget_total = compute_global_budget(len(patient_snapshots))
        self.budget_remaining = self.budget_total
        self.budget_used = 0

        # Shared systems
        self.state_machine = ActionLifecycleTracker(rng=self.rng)
        self.lagged_queue = LaggedRewardQueue(rng=self.rng)

        # Daily accumulators (reset each day)
        self.daily_gap_closures: Dict[str, int] = {}
        self.daily_actions_taken = 0

        # Cache state vectors per patient (updated by dynamics model when available)
        self._state_vectors: Dict[str, np.ndarray] = {}

        # Baseline HEDIS compliance: patients NOT in initial open_gaps already meet the measure.
        # Used to convert gap-closure counts to full HEDIS compliance rates (for STARS scoring).
        n_total = len(self.patients)
        self._n_initially_meeting: Dict[str, int] = {}
        for m in HEDIS_MEASURES:
            n_open = sum(1 for ps in self.patients.values() if m in ps.snapshot.get("open_gaps", []))
            self._n_initially_meeting[m] = n_total - n_open
        self._n_total_patients: int = n_total

    @property
    def cohort_avg_messages(self) -> float:
        if not self.patients:
            return 0.0
        return np.mean([p.messages_sent for p in self.patients.values()])

    # -------------------------------------------------------------------------
    # 1. Get patient context (state vector + action mask)
    # -------------------------------------------------------------------------
    def get_patient_context(self, pid: str) -> Dict[str, Any]:
        """Get everything the agent needs to make a decision for this patient."""
        ps = self.patients[pid]
        snap = ps.snapshot
        open_gaps = set(snap.get("open_gaps", []))

        # Compute system-level features
        budget_daily_spend = self.budget_used / max(self.day, 1)
        pending = self.state_machine.get_pending_actions(pid)
        pending_measures = {p["measure"] for p in pending}

        # Build state vector with all 3 tiers of features
        state_vec = snapshot_to_vector(
            snap,
            # Tier 2: System state
            day_of_year=self.day * 4,
            budget_remaining=self.budget_remaining,
            budget_max=self.budget_total,
            budget_daily_spend=budget_daily_spend,
            cohort_size=len(self.patients),
            cohort_avg_messages=self.cohort_avg_messages,
            stars_score=getattr(self, '_current_stars', 1.0),
            stars_7d_trend=getattr(self, '_stars_7d_trend', 0.0),
            pct_measures_above_4=getattr(self, '_pct_above_4', 0.0),
            lowest_measure_stars=getattr(self, '_lowest_stars', 1.0),
            cohort_channel_rates=getattr(self, '_cohort_channel_rates', None),
            # Tier 3: Action context
            patient_messages_received=ps.messages_sent,
            patient_response_rate=ps.response_rate,
            patient_contacts_7d=ps.contacts_in_window(self.day, 7),
            patient_contacts_14d=ps.contacts_in_window(self.day, 14),
            patient_contacts_30d=ps.contacts_in_window(self.day, 30),
            patient_days_since_contact=(
                self.day - ps.contact_days[-1] if ps.contact_days else 90
            ),
            patient_channels_used=len(ps.channels_used),
            patient_channel_success=getattr(ps, '_channel_success', None),
            patient_days_since_closure=(
                self.day - ps.last_closure_day if ps.last_closure_day > 0 else 90.0
            ),
            patient_avg_gap_age=self.day * 4.0,
            num_pending_actions=len(pending),
            num_in_flight_measures=len(pending_measures),
            # Channel affinity: lifetime counts + recency
            channel_affinity_counts=ps.channel_action_counts,
            channel_affinity_recency={
                ch: (self.day - d if d >= 0 else 90)
                for ch, d in ps.channel_last_day.items()
            },
        )

        # If dynamics model available, blend with learned state prediction
        if self.dynamics_model is not None and pid in self._state_vectors:
            prev_vec = self._state_vectors[pid]
            state_vec = self._blend_state(state_vec, prev_vec)

        # Cache for next step's dynamics model input
        self._state_vectors[pid] = state_vec

        # Build action mask with all business rules
        engagement = snap.get("engagement", {})
        channel_avail = {
            "sms": engagement.get("sms_consent", False),
            "email": engagement.get("email_available", False),
            "portal": engagement.get("portal_registered", False),
            "app": engagement.get("app_installed", False),
            "ivr": True,
        }

        # Pending actions block same-measure re-sends
        pending = self.state_machine.get_pending_actions(pid)
        pending_measures = {p["measure"] for p in pending}
        recent = {m: 0 for m in pending_measures}
        # Also check actual recent measures
        for m, last_day in ps.recent_measures.items():
            if self.day - last_day < MIN_DAYS_BETWEEN_SAME_MEASURE:
                recent[m] = self.day - last_day

        mask = compute_action_mask(
            open_gaps=open_gaps,
            channel_availability=channel_avail,
            contacts_this_week=ps.contacts_in_window(self.day),
            recent_measures=recent,
            budget_remaining=self.budget_remaining,
            days_since_last_mail=self.day - getattr(ps, 'last_mail_day', -999),
        )

        return {
            "state_vec": state_vec,
            "mask": mask,
            "open_gaps": open_gaps,
            "patient_state": ps,
        }

    # -------------------------------------------------------------------------
    # 2. Execute action and compute outcome
    # -------------------------------------------------------------------------
    def execute_action(self, pid: str, action_id: int) -> Dict[str, Any]:
        """Execute an action for a patient and return the outcome."""
        ps = self.patients[pid]
        snap = ps.snapshot
        is_no_act = action_id == 0
        action_info = ACTION_BY_ID.get(action_id)

        if is_no_act or action_info is None:
            reward = compute_reward(measure=None, is_no_action=True)
            return {
                "action_id": 0, "measure": None, "channel": None, "variant": None,
                "reward": reward, "is_no_action": True,
                "engagement": {},
            }

        # Track in state machine with patient archetype for personalized transitions
        tracking_id = f"day{self.day:02d}_{pid}_{action_id}"
        patient_arch = {
            "channel_affinity": snap.get("channel_affinity", {}),
            "channel_engagement": snap.get("channel_engagement", {}),
            "overall_responsiveness": snap.get("overall_responsiveness", 0.5),
            "variant_boost": snap.get("variant_boost", {}),
        }
        self.state_machine.create_action(
            tracking_id, pid, action_id, action_info.measure,
            action_info.channel, action_info.variant, self.day,
            patient_archetype=patient_arch,
        )
        self.state_machine.advance(tracking_id, self.day)  # → QUEUED

        signals = self.state_machine.get_engagement_signals(tracking_id)

        # Compute closure probability: learned model or ground-truth archetype
        if self.reward_model is not None and pid in self._state_vectors:
            closure_prob = self._model_closure_prob(pid, action_id)
        else:
            closure_prob = self._compute_closure_prob(ps, action_info, signals)

        # Schedule lagged reward
        self.lagged_queue.schedule(
            current_day=self.day,
            patient_id=pid,
            measure=action_info.measure,
            action_id=action_id,
            closure_prob=min(closure_prob, CLOSURE_PROB_CAP),
        )

        # Compute reward
        reward = compute_reward(
            measure=action_info.measure,
            clicked=signals.get("clicked", False),
            gap_closed=False,  # Gap closure is lagged
            is_no_action=False,
        )

        # Channel diversity bonus: reward using channels the patient hasn't seen much
        from config import REWARD_WEIGHTS
        channels_used = ps.channels_used if isinstance(ps.channels_used, set) else set()
        if action_info.channel not in channels_used:
            reward += REWARD_WEIGHTS.get("channel_diversity", 0.01)

        # Update patient state
        self.budget_remaining = max(0, self.budget_remaining - 1)
        self.budget_used += 1
        ps.messages_sent += 1
        ps.contact_days.append(self.day)
        ps.channels_used.add(action_info.channel)
        ps.channel_action_counts[action_info.channel] = ps.channel_action_counts.get(action_info.channel, 0) + 1
        ps.channel_last_day[action_info.channel] = self.day
        ps.recent_measures[action_info.measure] = self.day
        if action_info.channel == "email":
            ps.last_email_day = self.day
        self.daily_actions_taken += 1

        # Apply dynamics model to evolve state vector for this patient
        if self.dynamics_model is not None and pid in self._state_vectors:
            self._apply_dynamics(pid, action_id)

        return {
            "action_id": action_id,
            "measure": action_info.measure,
            "channel": action_info.channel,
            "variant": action_info.variant,
            "reward": reward,
            "is_no_action": False,
            "engagement": signals,
            "budget_remaining": self.budget_remaining,
            "budget_max": self.budget_total,
            "patient_messages": ps.messages_sent,
        }

    # -------------------------------------------------------------------------
    # 3. Advance day — process lagged rewards, advance state machine
    # -------------------------------------------------------------------------
    def advance_day(self) -> Dict[str, Any]:
        """End-of-day processing. Returns daily summary."""
        # Organic gap closures — gaps that close WITHOUT outreach
        # (patient visits doctor independently, fills Rx, gets vaccine, etc.)
        from config import ORGANIC_CLOSURE_DAILY_RATE
        organic_closures = {}
        for pid, ps in self.patients.items():
            for m in list(ps.snapshot.get("open_gaps", [])):
                if m in getattr(ps, 'closed_measures', set()):
                    continue  # Already closed
                daily_prob = ORGANIC_CLOSURE_DAILY_RATE.get(m, 0.001)
                # Modulate by patient responsiveness (more engaged patients close more organically)
                responsiveness = ps.snapshot.get("overall_responsiveness", 0.5)
                adjusted_prob = daily_prob * (0.5 + responsiveness)
                if self.rng.random() < adjusted_prob:
                    ps.closed_measures.add(m)
                    ps.last_closure_day = self.day
                    organic_closures[m] = organic_closures.get(m, 0) + 1

        # Advance all pending state machine actions one step
        transitions = self.state_machine.advance_all(self.day)

        # Count state machine acceptances as patient responses
        for t in transitions:
            if t.get("to_state") in (ActionState.ACCEPTED, ActionState.COMPLETED):
                ps = self.patients.get(t.get("patient_id"))
                if ps:
                    ps.responses += 1

        # Prune rolling contact windows
        for ps in self.patients.values():
            ps.prune_contact_window(self.day)

        # Process lagged rewards — THIS is where gap closure reward materializes
        resolved = self.lagged_queue.collect(self.day)
        self.daily_gap_closures = {m: 0 for m in HEDIS_MEASURES}
        closure_reward = 0.0
        reward_updates = []  # Track which past experiences need reward updates
        for r in resolved:
            if r["will_close"]:
                measure = r["measure"]
                self.daily_gap_closures[measure] = \
                    self.daily_gap_closures.get(measure, 0) + 1
                measure_weight = MEASURE_WEIGHTS.get(measure, 1)
                closure_reward += 1.0 * measure_weight
                ps = self.patients.get(r["patient_id"])
                if ps:
                    ps.last_closure_day = self.day
                    ps.closed_measures.add(measure)
                    ps.responses += 1
                # Track for caller to handle retroactive updates
                reward_updates.append({
                    "patient_id": r["patient_id"],
                    "scheduled_day": r.get("scheduled_day", 0),
                    "measure": measure,
                    "reward_delta": 1.0 * measure_weight,
                })

        # Count patients per measure (for closure rate denominator)
        total_patients = {m: 0 for m in HEDIS_MEASURES}
        for ps in self.patients.values():
            for m in ps.snapshot.get("open_gaps", []):
                total_patients[m] += 1

        # Merge organic closures into daily gap closures
        for m, count in organic_closures.items():
            self.daily_gap_closures[m] = self.daily_gap_closures.get(m, 0) + count

        summary = {
            "day": self.day,
            "daily_actions": self.daily_actions_taken,
            "gap_closures": dict(self.daily_gap_closures),
            "organic_closures": organic_closures,
            "closure_reward": closure_reward,
            "reward_updates": reward_updates,
            "total_patients": total_patients,
            "n_initially_meeting": self._n_initially_meeting,
            "n_total_patients": self._n_total_patients,
            "budget_remaining": self.budget_remaining,
            "budget_max": self.budget_total,
            "budget_used_today": self.daily_actions_taken,
            "pending_rewards": self.lagged_queue.get_pending_count(),
            "state_machine_funnel": self.state_machine.get_funnel_stats(),
        }

        # Update system-level metrics for Tier 2 features
        self._update_system_metrics(total_patients)

        # Reset daily accumulators
        self.daily_actions_taken = 0

        # Advance day counter
        self.day += 1

        return summary

    # -------------------------------------------------------------------------
    # Warm start
    # -------------------------------------------------------------------------
    def warm_start(self, rng: np.random.Generator = None):
        """Initialize patients mid-flight with varied histories."""
        if rng is None:
            rng = self.rng

        stats = {"fresh": 0, "mid_flight": 0, "heavy_contact": 0, "near_exhausted": 0}

        for pid, ps in self.patients.items():
            stage = rng.random()

            if stage < 0.20:
                budget_used = 0
                stats["fresh"] += 1
            elif stage < 0.55:
                budget_used = int(rng.integers(2, 7))
                stats["mid_flight"] += 1
                # Schedule pending lagged rewards
                open_gaps = ps.snapshot.get("open_gaps", [])
                for _ in range(min(int(rng.integers(1, 4)), len(open_gaps))):
                    m = rng.choice(open_gaps)
                    prob = GAP_CLOSURE_BASE_RATES.get(m, 0.5) * 0.15
                    self.lagged_queue.schedule(0, pid, m, 1, min(prob, 0.5))
            elif stage < 0.85:
                budget_used = int(rng.integers(6, 10))
                stats["heavy_contact"] += 1
                open_gaps = ps.snapshot.get("open_gaps", [])
                for _ in range(min(int(rng.integers(2, 6)), len(open_gaps))):
                    m = rng.choice(open_gaps)
                    prob = GAP_CLOSURE_BASE_RATES.get(m, 0.5) * 0.20
                    self.lagged_queue.schedule(0, pid, m, 1, min(prob, 0.5))
            else:
                budget_used = int(rng.integers(9, 15))
                stats["near_exhausted"] += 1
                open_gaps = ps.snapshot.get("open_gaps", [])
                for _ in range(min(int(rng.integers(0, 3)), len(open_gaps))):
                    m = rng.choice(open_gaps)
                    self.lagged_queue.schedule(0, pid, m, 1, 0.9)

            ps.messages_sent = budget_used
            # Stagger contact history across the rolling window
            if budget_used > 0:
                recent = sorted(rng.choice(range(-6, 1),
                               size=min(budget_used, 3), replace=False).tolist())
                ps.contact_days = recent

            self.budget_remaining -= budget_used
            self.budget_used += budget_used

        # Resolve any immediate lagged rewards and track in closed_measures
        resolved = self.lagged_queue.collect(0)
        day0_closures = 0
        for r in resolved:
            if r["will_close"]:
                day0_closures += 1
                ps = self.patients.get(r["patient_id"])
                if ps:
                    ps.closed_measures.add(r["measure"])
                    ps.last_closure_day = 0

        return {
            "stats": stats,
            "budget_remaining": self.budget_remaining,
            "budget_total": self.budget_total,
            "day0_closures": day0_closures,
            "pending_rewards": self.lagged_queue.get_pending_count(),
            "in_flight_actions": len(self.state_machine.actions),
        }

    # -------------------------------------------------------------------------
    # Internal: system-level metric tracking for Tier 2 features
    # -------------------------------------------------------------------------
    def _update_system_metrics(self, total_patients: Dict[str, int]):
        """Update cached system-level metrics used by Tier 2 features."""
        from environment.reward import compute_stars_score, measure_rate_to_stars

        # Compute HEDIS compliance rates (not gap-closure fractions).
        # hedis_rate[m] = (patients already meeting + patients who closed gap) / all patients
        closure_rates = {}
        for m in HEDIS_MEASURES:
            n_already = self._n_initially_meeting.get(m, 0)
            n_closed = sum(1 for ps in self.patients.values()
                          if m in getattr(ps, 'closed_measures', set()))
            closure_rates[m] = (n_already + n_closed) / max(self._n_total_patients, 1)

        self._current_stars = compute_stars_score(closure_rates)

        # STARS 7-day trend
        prev_stars = getattr(self, '_prev_stars', self._current_stars)
        self._stars_7d_trend = self._current_stars - prev_stars
        self._prev_stars = self._current_stars

        # Per-measure star ratings
        measure_stars = {m: measure_rate_to_stars(m, r) for m, r in closure_rates.items()}
        above_4 = sum(1 for s in measure_stars.values() if s >= 4.0)
        self._pct_above_4 = above_4 / max(len(measure_stars), 1)
        self._lowest_stars = min(measure_stars.values()) if measure_stars else 1.0

        # Cohort channel acceptance rates
        funnel = self.state_machine.get_funnel_stats()
        total_actions = max(sum(funnel.values()), 1)
        # Approximate channel rates from state machine data
        self._cohort_channel_rates = getattr(self, '_cohort_channel_rates', {
            "sms": 0.3, "email": 0.2, "portal": 0.15, "app": 0.1, "ivr": 0.15,
        })

    # -------------------------------------------------------------------------
    # Internal: closure probability
    # -------------------------------------------------------------------------
    def _compute_closure_prob(self, ps: PatientState, action_info, signals) -> float:
        """Compute gap closure probability using patient archetype data."""
        snap = ps.snapshot
        base_rate = GAP_CLOSURE_BASE_RATES.get(action_info.measure, 0.5)
        category = get_measure_category(action_info.measure)

        # Archetype-driven factors
        gap_boost_data = snap.get("gap_closure_boost", {})
        gap_boost = gap_boost_data.get(category, 1.0) if isinstance(gap_boost_data, dict) else 1.0
        ch_affinity = snap.get("channel_affinity", {}).get(action_info.channel, 0.3)
        responsiveness = snap.get("overall_responsiveness", 0.5)

        closure_prob = base_rate * CLOSURE_BASE_MULTIPLIER * gap_boost * ch_affinity * responsiveness

        # Variant boost
        variant_boost = snap.get("variant_boost", {}).get(action_info.variant, 1.0)
        closure_prob *= variant_boost

        # Engagement multipliers
        if signals.get("clicked"):
            closure_prob *= CLOSURE_CLICKED_FACTOR
        elif signals.get("opened"):
            closure_prob *= CLOSURE_OPENED_FACTOR
        elif signals.get("delivered"):
            closure_prob *= CLOSURE_DELIVERED_FACTOR

        return closure_prob

    # -------------------------------------------------------------------------
    # Internal: learned model methods
    # -------------------------------------------------------------------------
    def _model_closure_prob(self, pid: str, action_id: int) -> float:
        """Use learned reward model for closure probability."""
        import torch
        state_vec = self._state_vectors[pid]
        state_arr = state_vec.reshape(1, -1)
        action_arr = np.array([action_id])
        # Use current simulation day as horizon
        days_arr = np.array([float(max(self.day, 1))], dtype=np.float32)

        prob = float(self.reward_model.predict(state_arr, action_arr, days_arr)[0])
        return max(0.0, min(prob, 1.0))

    def _apply_dynamics(self, pid: str, action_id: int):
        """Use learned dynamics model to evolve patient state vector."""
        import torch
        from config import (
            FEAT_IDX_DEMOGRAPHICS_START, FEAT_IDX_CONDITIONS_START,
            FEAT_IDX_GAP_FLAGS_START, FEAT_IDX_CHANNEL_AVAIL_START,
            TIER2_START,
        )
        state_vec = self._state_vectors[pid]
        next_state = self.dynamics_model.predict(
            state_vec, np.array([action_id]), add_noise=True
        ).flatten().astype(np.float32)

        # Freeze features controlled by ground-truth business rules:
        # Demographics (0-5) are static
        for idx in range(FEAT_IDX_DEMOGRAPHICS_START, FEAT_IDX_DEMOGRAPHICS_START + 6):
            next_state[idx] = state_vec[idx]
        # Conditions (12-19) are static
        for idx in range(FEAT_IDX_CONDITIONS_START, FEAT_IDX_CONDITIONS_START + 8):
            next_state[idx] = state_vec[idx]

        # Gap flags stay binary (closures handled by lagged reward system)
        for idx in range(FEAT_IDX_GAP_FLAGS_START, FEAT_IDX_GAP_FLAGS_START + len(HEDIS_MEASURES)):
            next_state[idx] = 1.0 if next_state[idx] > 0.5 else 0.0
        # Channel availability stays binary
        for idx in range(FEAT_IDX_CHANNEL_AVAIL_START, FEAT_IDX_CHANNEL_AVAIL_START + 4):
            next_state[idx] = 1.0 if next_state[idx] > 0.5 else 0.0

        # Tier 2 and Tier 3 are always recomputed from ground truth, not predicted
        next_state[TIER2_START:] = state_vec[TIER2_START:]

        next_state = np.clip(next_state, -3.0, 3.0)
        self._state_vectors[pid] = next_state

    def _blend_state(self, ground_truth_vec: np.ndarray, model_vec: np.ndarray) -> np.ndarray:
        """Blend ground-truth and model-predicted state vectors.

        3-Tier layout:
          Tier 1 (0-49):  Patient state — model learns clinical/med evolution
          Tier 2 (50-69): System state — always ground truth
          Tier 3 (70-127): Action context — always ground truth
        """
        from config import (
            FEAT_IDX_CLINICAL_START, FEAT_IDX_MEDICATIONS_START,
            FEAT_IDX_RISK_START, TIER2_START,
        )
        blended = ground_truth_vec.copy()

        # Within Tier 1, let the model predict clinical and medication evolution:
        # Clinical vitals (6-11): model learns evolution
        blended[FEAT_IDX_CLINICAL_START:FEAT_IDX_CLINICAL_START + 6] = \
            model_vec[FEAT_IDX_CLINICAL_START:FEAT_IDX_CLINICAL_START + 6]

        # Medication fill rates (20-23): model learns adherence
        blended[FEAT_IDX_MEDICATIONS_START:FEAT_IDX_MEDICATIONS_START + 4] = \
            model_vec[FEAT_IDX_MEDICATIONS_START:FEAT_IDX_MEDICATIONS_START + 4]

        # Risk scores (42-45): model learns risk evolution
        blended[FEAT_IDX_RISK_START:FEAT_IDX_RISK_START + 4] = \
            model_vec[FEAT_IDX_RISK_START:FEAT_IDX_RISK_START + 4]

        # Tier 2 and Tier 3 always use ground truth
        return np.clip(blended, -3.0, 3.0)
