"""
Action space encoding/decoding for the HEDIS environment.
Maps between integer action indices and (measure, channel, variant) tuples.
"""
from typing import Optional, Tuple
from config import ACTION_CATALOG, ACTION_BY_ID, NUM_ACTIONS, Action


def decode_action(action_id: int) -> Action:
    """Decode an integer action ID to its Action namedtuple."""
    if action_id < 0 or action_id >= NUM_ACTIONS:
        raise ValueError(f"Invalid action_id: {action_id}. Must be in [0, {NUM_ACTIONS})")
    return ACTION_BY_ID[action_id]


def is_no_action(action_id: int) -> bool:
    """Check if the action is the no-op action."""
    return action_id == 0


def get_action_measure(action_id: int) -> Optional[str]:
    """Get the HEDIS measure for an action, or None for no_action."""
    action = decode_action(action_id)
    return None if action.measure == "NO_ACTION" else action.measure


def get_action_channel(action_id: int) -> Optional[str]:
    """Get the channel for an action, or None for no_action."""
    action = decode_action(action_id)
    return None if action.channel == "none" else action.channel


def get_action_info(action_id: int) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (measure, channel, variant) for an action. All None for no_action."""
    action = decode_action(action_id)
    if action.measure == "NO_ACTION":
        return None, None, None
    return action.measure, action.channel, action.variant
