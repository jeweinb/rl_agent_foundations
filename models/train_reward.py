"""
Training script for the reward model.
Learns P(gap_closure | state, action, days_elapsed) from gap closure data.
"""
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from config import (
    STATE_DIM, NUM_ACTIONS, REWARD_MODEL_CONFIG as CFG,
    GENERATED_DATA_DIR, HEDIS_MEASURES,
)
from models.reward_model import RewardModel
from environment.state_space import snapshot_to_vector


class ClosureDataset(Dataset):
    """Dataset of (state, action, days_elapsed, gap_closed) tuples."""

    def __init__(self, states, actions, days_elapsed, labels):
        self.states = torch.FloatTensor(states)
        self.actions = torch.LongTensor(actions)
        self.days = torch.FloatTensor(days_elapsed)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.days[idx], self.labels[idx]


def prepare_closure_data(
    state_snapshots: list,
    historical_activity: list,
    gap_closure: list,
) -> tuple:
    """Build (state, action, days, closed) tuples from historical + gap closure data."""
    # Build patient state lookup
    patient_states = {}
    for snap in state_snapshots:
        vec = snapshot_to_vector(snap)
        patient_states[snap["patient_id"]] = vec

    states, actions, days_elapsed, labels = [], [], [], []

    for record in historical_activity:
        pid = record["patient_id"]
        if pid not in patient_states:
            continue

        state_vec = patient_states[pid]
        action_id = record["action_id"]
        outcome = record["outcome"]

        # Create samples at different time horizons
        for horizon_days, closed_key in [
            (30, "gap_closed_within_30d"),
            (90, "gap_closed_within_90d"),
        ]:
            closed = outcome.get(closed_key, False)
            states.append(state_vec)
            actions.append(action_id)
            days_elapsed.append(float(horizon_days))
            labels.append(1.0 if closed else 0.0)

        # If we know actual days to closure, add that as a sample too
        if outcome.get("days_to_closure") is not None:
            dtc = outcome["days_to_closure"]
            states.append(state_vec)
            actions.append(action_id)
            days_elapsed.append(float(dtc))
            labels.append(1.0)

    return (
        np.array(states),
        np.array(actions),
        np.array(days_elapsed),
        np.array(labels),
    )


def train_reward_model(
    state_snapshots: list = None,
    historical_activity: list = None,
    gap_closure: list = None,
    model: RewardModel = None,
    epochs: int = None,
    batch_size: int = None,
    lr: float = None,
    verbose: bool = True,
) -> RewardModel:
    """Train the reward model on gap closure data."""
    if epochs is None:
        epochs = CFG["epochs"]
    if batch_size is None:
        batch_size = CFG["batch_size"]
    if lr is None:
        lr = CFG["lr"]

    # Load data if not provided
    if state_snapshots is None:
        with open(f"{GENERATED_DATA_DIR}/state_features.json") as f:
            state_snapshots = json.load(f)
    if historical_activity is None:
        with open(f"{GENERATED_DATA_DIR}/historical_activity.json") as f:
            historical_activity = json.load(f)
    if gap_closure is None:
        with open(f"{GENERATED_DATA_DIR}/gap_closure.json") as f:
            gap_closure = json.load(f)

    if verbose:
        print("Preparing closure data...")
    states, actions, days, labels = prepare_closure_data(
        state_snapshots, historical_activity, gap_closure
    )
    if verbose:
        pos_rate = labels.mean()
        print(f"  {len(states)} samples prepared (positive rate: {pos_rate:.3f})")

    dataset = ClosureDataset(states, actions, days, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if model is None:
        model = RewardModel()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for state_batch, action_batch, days_batch, label_batch in dataloader:
            optimizer.zero_grad()
            loss = model.compute_loss(state_batch, action_batch, days_batch, label_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if verbose and (epoch + 1) % 10 == 0:
            avg_loss = total_loss / max(n_batches, 1)
            print(f"  Epoch {epoch + 1}/{epochs} — loss: {avg_loss:.6f}")

    if verbose:
        print("Reward model training complete.")
    return model
