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
    ) -> Dict[str, Any]:
        """Register a new action in the state machine."""
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
        probs = CHANNEL_TRANSITION_PROBS.get(channel, {})

        # Determine transition
        if current == ActionState.CREATED:
            next_state = ActionState.QUEUED
        elif current == ActionState.QUEUED:
            prob = probs.get("QUEUED→PRESENTED", 0.9)
            next_state = ActionState.PRESENTED if self.rng.random() < prob else ActionState.FAILED
        elif current == ActionState.PRESENTED:
            prob = probs.get("PRESENTED→VIEWED", 0.5)
            next_state = ActionState.VIEWED if self.rng.random() < prob else ActionState.EXPIRED
        elif current == ActionState.VIEWED:
            prob = probs.get("VIEWED→ACCEPTED", 0.3)
            next_state = ActionState.ACCEPTED if self.rng.random() < prob else ActionState.DECLINED
        elif current == ActionState.ACCEPTED:
            prob = probs.get("ACCEPTED→COMPLETED", 0.6)
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
