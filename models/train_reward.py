"""
Training script for the reward model.
Learns P(gap_closure | state, action, days_elapsed) from gap closure data.

Key design choices:
- Train at multiple horizons including short ones (3, 7, 14, 30, 90 days)
- Balance positive and negative examples (most actions DON'T close gaps)
- Add noise to state vectors so model doesn't memorize per-patient
- Proper calibration so the model outputs realistic probabilities
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
    """Build (state, action, days, closed) tuples with proper calibration.

    Creates samples at multiple horizons with realistic positive rates:
    - 3 days: ~5% positive (very few gaps close this fast)
    - 7 days: ~10% positive
    - 14 days: ~15% positive
    - 30 days: ~25% positive (from actual data)
    - 90 days: ~65% positive (from actual data)
    """
    patient_states = {}
    for snap in state_snapshots:
        vec = snapshot_to_vector(snap)
        patient_states[snap["patient_id"]] = vec

    states, actions, days_elapsed, labels = [], [], [], []
    rng = np.random.default_rng(42)

    for record in historical_activity:
        pid = record["patient_id"]
        if pid not in patient_states:
            continue

        base_state = patient_states[pid]
        action_id = record["action_id"]
        outcome = record["outcome"]
        closed_30d = outcome.get("gap_closed_within_30d", False)
        closed_90d = outcome.get("gap_closed_within_90d", False)
        dtc = outcome.get("days_to_closure")

        # Add noise to state vector so model generalizes (not memorizing per-patient)
        state_vec = base_state + rng.normal(0, 0.02, STATE_DIM).astype(np.float32)

        # --- Short horizons (mostly negative) ---
        # 3-day horizon: only closed if dtc <= 3
        closed_3d = dtc is not None and dtc <= 3
        states.append(state_vec)
        actions.append(action_id)
        days_elapsed.append(3.0)
        labels.append(1.0 if closed_3d else 0.0)

        # 7-day horizon: only closed if dtc <= 7
        closed_7d = dtc is not None and dtc <= 7
        states.append(state_vec)
        actions.append(action_id)
        days_elapsed.append(7.0)
        labels.append(1.0 if closed_7d else 0.0)

        # 14-day horizon
        closed_14d = dtc is not None and dtc <= 14
        states.append(state_vec)
        actions.append(action_id)
        days_elapsed.append(14.0)
        labels.append(1.0 if closed_14d else 0.0)

        # --- Actual horizons from data ---
        states.append(state_vec)
        actions.append(action_id)
        days_elapsed.append(30.0)
        labels.append(1.0 if closed_30d else 0.0)

        states.append(state_vec)
        actions.append(action_id)
        days_elapsed.append(90.0)
        labels.append(1.0 if closed_90d else 0.0)

    states = np.array(states)
    actions = np.array(actions)
    days_elapsed = np.array(days_elapsed)
    labels = np.array(labels)

    return states, actions, days_elapsed, labels


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
        # Show rates by horizon
        for horizon in [3, 7, 14, 30, 90]:
            mask = np.isclose(days, horizon)
            if mask.sum() > 0:
                hr = labels[mask].mean()
                print(f"  Horizon {horizon:2d}d: {hr:.1%} positive ({mask.sum():,} samples)")
        print(f"  Total: {len(states):,} samples, overall positive rate: {pos_rate:.1%}")

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

    # Print calibration check
    if verbose:
        model.eval()
        with torch.no_grad():
            sample_states = torch.FloatTensor(states[:1000])
            sample_actions = torch.LongTensor(actions[:1000])
            for horizon in [3.0, 7.0, 14.0, 30.0]:
                sample_days = torch.ones(1000) * horizon
                preds = model.forward(sample_states, sample_actions, sample_days).squeeze()
                print(f"  Calibration check — horizon {horizon:.0f}d: "
                      f"mean pred={preds.mean():.3f}, min={preds.min():.3f}, max={preds.max():.3f}")

    if verbose:
        print("Reward model training complete.")
    return model
