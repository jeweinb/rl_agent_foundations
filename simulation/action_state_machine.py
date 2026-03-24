"""
Action Lifecycle State Machine.

Simulates an external third-party system that tracks the disposition of each
action through its lifecycle. Feeds engagement signals back to the RL system
and influences eligibility constraints.

States:
    CREATED → QUEUED → PRESENTED → VIEWED → ACCEPTED → COMPLETED
                                         ↘ DECLINED
                                  ↘ EXPIRED
                         ↘ FAILED

Transitions are probabilistic based on channel, measure, and patient engagement.
"""
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum


class ActionState(str, Enum):
    CREATED = "CREATED"
    QUEUED = "QUEUED"
    PRESENTED = "PRESENTED"
    VIEWED = "VIEWED"
    ACCEPTED = "ACCEPTED"
    DECLINED = "DECLINED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"


# Ordered lifecycle stages (happy path). Used by dashboards for funnels/Sankeys.
LIFECYCLE_STAGES = [
    ActionState.CREATED,
    ActionState.QUEUED,
    ActionState.PRESENTED,
    ActionState.VIEWED,
    ActionState.ACCEPTED,
    ActionState.COMPLETED,
]

# Terminal states (action lifecycle is finished)
TERMINAL_STATES = {
    ActionState.COMPLETED,
    ActionState.DECLINED,
    ActionState.FAILED,
    ActionState.EXPIRED,
}

# All states in display order (lifecycle + failure modes)
ALL_STATES_ORDERED = [
    ActionState.CREATED,
    ActionState.QUEUED,
    ActionState.PRESENTED,
    ActionState.VIEWED,
    ActionState.ACCEPTED,
    ActionState.COMPLETED,
    ActionState.DECLINED,
    ActionState.FAILED,
    ActionState.EXPIRED,
]

# Valid state transitions
VALID_TRANSITIONS = {
    ActionState.CREATED: [ActionState.QUEUED],
    ActionState.QUEUED: [ActionState.PRESENTED, ActionState.FAILED],
    ActionState.PRESENTED: [ActionState.VIEWED, ActionState.EXPIRED],
    ActionState.VIEWED: [ActionState.ACCEPTED, ActionState.DECLINED],
    ActionState.ACCEPTED: [ActionState.COMPLETED, ActionState.FAILED],
    ActionState.DECLINED: [],  # terminal
    ActionState.COMPLETED: [],  # terminal
    ActionState.FAILED: [],  # terminal
    ActionState.EXPIRED: [],  # terminal
}

# Transition probabilities by channel
CHANNEL_TRANSITION_PROBS = {
    "sms": {
        "QUEUED→PRESENTED": 0.95,
        "PRESENTED→VIEWED": 0.82,
        "VIEWED→ACCEPTED": 0.25,
        "ACCEPTED→COMPLETED": 0.60,
    },
    "email": {
        "QUEUED→PRESENTED": 0.92,
        "PRESENTED→VIEWED": 0.25,
        "VIEWED→ACCEPTED": 0.35,
        "ACCEPTED→COMPLETED": 0.55,
    },
    "portal": {
        "QUEUED→PRESENTED": 1.0,
        "PRESENTED→VIEWED": 0.60,
        "VIEWED→ACCEPTED": 0.45,
        "ACCEPTED→COMPLETED": 0.65,
    },
    "app": {
        "QUEUED→PRESENTED": 0.98,
        "PRESENTED→VIEWED": 0.55,
        "VIEWED→ACCEPTED": 0.40,
        "ACCEPTED→COMPLETED": 0.70,
    },
    "ivr": {
        "QUEUED→PRESENTED": 0.70,
        "PRESENTED→VIEWED": 0.70,
        "VIEWED→ACCEPTED": 0.20,
        "ACCEPTED→COMPLETED": 0.50,
    },
}


class ActionLifecycleTracker:
    """Tracks the lifecycle of all actions via the state machine."""

    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng or np.random.default_rng()
        self.actions: Dict[str, Dict[str, Any]] = {}  # tracking_id -> action record
        self.history: List[Dict[str, Any]] = []  # All state transitions

    def create_action(
        self,
        tracking_id: str,
        patient_id: str,
        action_id: int,
        measure: str,
        channel: str,
        variant: str,
        day: int,
        patient_archetype: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Register a new action in the state machine.

        Args:
            patient_archetype: Dict with channel_affinity, channel_engagement,
                overall_responsiveness, variant_boost from the patient's archetype.
                Used to modulate transition probabilities.
        """
        record = {
            "tracking_id": tracking_id,
            "patient_id": patient_id,
            "action_id": action_id,
            "measure": measure,
            "channel": channel,
            "variant": variant,
            "day_created": day,
            "current_state": ActionState.CREATED,
            "state_history": [
                {"state": ActionState.CREATED, "day": day, "timestamp": datetime.now().isoformat()}
            ],
            "terminal": False,
            "patient_archetype": patient_archetype or {},
        }
        self.actions[tracking_id] = record
        self.history.append({
            "tracking_id": tracking_id,
            "patient_id": patient_id,
            "from_state": None,
            "to_state": ActionState.CREATED,
            "day": day,
        })
        return record

    def advance(self, tracking_id: str, day: int) -> Optional[Dict[str, Any]]:
        """Attempt to advance an action to its next state.

        Transition probabilities are modulated by the patient's archetype:
        - Channel affinity boosts PRESENTED→VIEWED (patient opens messages on preferred channels)
        - Channel engagement boosts VIEWED→ACCEPTED (patient acts on channels they engage with)
        - Overall responsiveness scales all patient-dependent transitions
        - Variant boost increases acceptance for content types the archetype responds to

        Returns the transition record if a transition occurred, None otherwise.
        """
        record = self.actions.get(tracking_id)
        if record is None or record["terminal"]:
            return None

        current = record["current_state"]
        valid_next = VALID_TRANSITIONS.get(current, [])
        if not valid_next:
            record["terminal"] = True
            return None

        channel = record["channel"]
        base_probs = CHANNEL_TRANSITION_PROBS.get(channel, {})

        # Get archetype modifiers
        arch = record.get("patient_archetype", {})
        ch_affinity = arch.get("channel_affinity", {}).get(channel, 0.5)
        ch_engagement = arch.get("channel_engagement", {}).get(channel, 0.2)
        responsiveness = arch.get("overall_responsiveness", 0.5)
        variant_boost = arch.get("variant_boost", {}).get(record.get("variant", ""), 1.0)

        # Determine transition with archetype-modulated probabilities
        if current == ActionState.CREATED:
            next_state = ActionState.QUEUED
        elif current == ActionState.QUEUED:
            # Delivery: mostly channel infrastructure, slight archetype effect
            prob = base_probs.get("QUEUED→PRESENTED", 0.9)
            next_state = ActionState.PRESENTED if self.rng.random() < prob else ActionState.FAILED
        elif current == ActionState.PRESENTED:
            # Viewing: heavily driven by channel affinity (does patient open this channel?)
            base = base_probs.get("PRESENTED→VIEWED", 0.5)
            prob = min(base * (0.5 + ch_affinity), 0.98)
            next_state = ActionState.VIEWED if self.rng.random() < prob else ActionState.EXPIRED
        elif current == ActionState.VIEWED:
            # Acceptance: driven by engagement level + variant match + responsiveness
            base = base_probs.get("VIEWED→ACCEPTED", 0.3)
            prob = min(base * (0.3 + ch_engagement) * responsiveness * variant_boost, 0.90)
            next_state = ActionState.ACCEPTED if self.rng.random() < prob else ActionState.DECLINED
        elif current == ActionState.ACCEPTED:
            # Completion: driven by overall responsiveness
            base = base_probs.get("ACCEPTED→COMPLETED", 0.6)
            prob = min(base * (0.5 + responsiveness * 0.5), 0.95)
            next_state = ActionState.COMPLETED if self.rng.random() < prob else ActionState.FAILED
        else:
            return None

        # Execute transition
        record["current_state"] = next_state
        record["state_history"].append({
            "state": next_state,
            "day": day,
            "timestamp": datetime.now().isoformat(),
        })
        if next_state in (ActionState.COMPLETED, ActionState.FAILED,
                          ActionState.DECLINED, ActionState.EXPIRED):
            record["terminal"] = True

        transition = {
            "tracking_id": tracking_id,
            "patient_id": record["patient_id"],
            "from_state": current,
            "to_state": next_state,
            "day": day,
            "measure": record["measure"],
            "channel": record["channel"],
        }
        self.history.append(transition)
        return transition

    def advance_all(self, day: int) -> List[Dict[str, Any]]:
        """Advance all non-terminal actions by one step."""
        transitions = []
        for tid in list(self.actions.keys()):
            t = self.advance(tid, day)
            if t:
                transitions.append(t)
        return transitions

    def get_engagement_signals(self, tracking_id: str) -> Dict[str, bool]:
        """Extract engagement signals from action state for reward computation."""
        record = self.actions.get(tracking_id, {})
        state = record.get("current_state", ActionState.CREATED)
        return {
            "delivered": state in (ActionState.PRESENTED, ActionState.VIEWED,
                                   ActionState.ACCEPTED, ActionState.COMPLETED,
                                   ActionState.DECLINED),
            "opened": state in (ActionState.VIEWED, ActionState.ACCEPTED,
                                ActionState.COMPLETED, ActionState.DECLINED),
            "clicked": state in (ActionState.ACCEPTED, ActionState.COMPLETED),
            "completed": state == ActionState.COMPLETED,
            "failed": state == ActionState.FAILED,
            "expired": state == ActionState.EXPIRED,
        }

    def get_pending_actions(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get all non-terminal actions for a patient (influences eligibility)."""
        return [
            r for r in self.actions.values()
            if r["patient_id"] == patient_id and not r["terminal"]
        ]

    def get_patient_history(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get full action history for a patient (for patient journey dashboard)."""
        return [
            r for r in self.actions.values()
            if r["patient_id"] == patient_id
        ]

    def get_funnel_stats(self) -> Dict[str, int]:
        """Get counts by terminal state for funnel visualization."""
        counts = {s.value: 0 for s in ActionState}
        for record in self.actions.values():
            counts[record["current_state"].value] += 1
        return counts

    def to_records(self) -> List[Dict[str, Any]]:
        """Export all action records for persistence."""
        records = []
        for r in self.actions.values():
            records.append({
                "tracking_id": r["tracking_id"],
                "patient_id": r["patient_id"],
                "action_id": r["action_id"],
                "measure": r["measure"],
                "channel": r["channel"],
                "variant": r["variant"],
                "day_created": r["day_created"],
                "current_state": r["current_state"].value,
                "state_history": [
                    {"state": sh["state"].value, "day": sh["day"], "timestamp": sh["timestamp"]}
                    for sh in r["state_history"]
                ],
                "terminal": r["terminal"],
            })
        return records
