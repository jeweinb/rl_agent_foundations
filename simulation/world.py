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
    CLOSURE_CLICKED_FACTOR, CLOSURE_OPENED_FACTOR, CLOSURE_DELIVERED_FACTOR,
    CLOSURE_PROB_CAP, MEASURE_WEIGHTS,
    compute_global_budget, get_measure_category,
    AVG_MESSAGES_PER_PATIENT, BUDGET_WARNING_THRESHOLD, BUDGET_CRITICAL_THRESHOLD,
)
from datagen.constants import GAP_CLOSURE_BASE_RATES
from environment.state_space import snapshot_to_vector
from environment.action_masking import compute_action_mask
from environment.reward import compute_reward
from simulation.action_state_machine import ActionLifecycleTracker
from simulation.lagged_rewards import LaggedRewardQueue


class PatientState:
    """Tracks per-patient live state across the simulation."""

    __slots__ = [
        "pid", "snapshot", "messages_sent", "contact_days",
        "channels_used", "responses", "last_closure_day",
        "recent_measures",
    ]

    def __init__(self, pid: str, snapshot: Dict[str, Any]):
        self.pid = pid
        self.snapshot = snapshot
        self.messages_sent = 0
        self.contact_days: List[int] = []   # Rolling 7-day window
        self.channels_used: Set[str] = set()
        self.responses = 0                   # Clicks/accepts
        self.last_closure_day = 0
        self.recent_measures: Dict[str, int] = {}  # measure → day last contacted

    @property
    def response_rate(self) -> float:
        return self.responses / max(self.messages_sent, 1)

    def contacts_in_window(self, current_day: int, window: int = 7) -> int:
        """Count contacts within the rolling window."""
        return sum(1 for d in self.contact_days if current_day - d < window)

    def prune_contact_window(self, current_day: int, window: int = 7):
        """Remove contact days outside the rolling window."""
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
    """Encapsulates all business rules, patient state, and outcome generation."""

    def __init__(
        self,
        patient_snapshots: List[Dict[str, Any]],
        eligibility_snapshots: List[Dict[str, Any]],
        rng: np.random.Generator = None,
    ):
        self.rng = rng or np.random.default_rng()
        self.day = 0

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

        # Build state vector with live features
        state_vec = snapshot_to_vector(
            snap,
            day_of_year=self.day * 4,
            budget_remaining=self.budget_remaining,
            budget_max=self.budget_total,
            patient_messages_received=ps.messages_sent,
            cohort_avg_messages=self.cohort_avg_messages,
            patient_response_rate=ps.response_rate,
            patient_avg_gap_age=self.day * 4.0,
            patient_days_since_closure=(
                self.day - ps.last_closure_day if ps.last_closure_day > 0 else 90.0
            ),
            patient_channels_used=len(ps.channels_used),
        )

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

        # Compute archetype-driven closure probability
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

        # Update patient state
        self.budget_remaining = max(0, self.budget_remaining - 1)
        self.budget_used += 1
        ps.messages_sent += 1
        ps.contact_days.append(self.day)
        ps.channels_used.add(action_info.channel)
        ps.recent_measures[action_info.measure] = self.day
        self.daily_actions_taken += 1

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
        # Advance all pending state machine actions one step
        self.state_machine.advance_all(self.day)

        # Prune rolling contact windows
        for ps in self.patients.values():
            ps.prune_contact_window(self.day)

        # Process lagged rewards — THIS is where gap closure reward materializes
        resolved = self.lagged_queue.collect(self.day)
        self.daily_gap_closures = {m: 0 for m in HEDIS_MEASURES}
        closure_reward = 0.0
        for r in resolved:
            if r["will_close"]:
                measure = r["measure"]
                self.daily_gap_closures[measure] = \
                    self.daily_gap_closures.get(measure, 0) + 1
                # Gap closure reward (the real objective)
                measure_weight = MEASURE_WEIGHTS.get(measure, 1)
                closure_reward += 1.0 * measure_weight
                # Update patient's last closure day
                ps = self.patients.get(r["patient_id"])
                if ps:
                    ps.last_closure_day = self.day
                    ps.responses += 1

        # Count patients per measure (for closure rate denominator)
        total_patients = {m: 0 for m in HEDIS_MEASURES}
        for ps in self.patients.values():
            for m in ps.snapshot.get("open_gaps", []):
                total_patients[m] += 1

        summary = {
            "day": self.day,
            "daily_actions": self.daily_actions_taken,
            "gap_closures": dict(self.daily_gap_closures),
            "closure_reward": closure_reward,
            "total_patients": total_patients,
            "budget_remaining": self.budget_remaining,
            "budget_max": self.budget_total,
            "budget_used_today": self.daily_actions_taken,
            "pending_rewards": self.lagged_queue.get_pending_count(),
            "state_machine_funnel": self.state_machine.get_funnel_stats(),
        }

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

        # Resolve any immediate lagged rewards
        resolved = self.lagged_queue.collect(0)
        day0_closures = sum(1 for r in resolved if r["will_close"])

        return {
            "stats": stats,
            "budget_remaining": self.budget_remaining,
            "budget_total": self.budget_total,
            "day0_closures": day0_closures,
            "pending_rewards": self.lagged_queue.get_pending_count(),
            "in_flight_actions": len(self.state_machine.actions),
        }

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
