"""
Training script for the dynamics model.
Learns state transitions from historical activity + state features data.
"""
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from config import STATE_DIM, NUM_ACTIONS, DYNAMICS_MODEL_CONFIG as CFG, GENERATED_DATA_DIR
from models.dynamics_model import DynamicsModel
from environment.state_space import snapshot_to_vector


class TransitionDataset(Dataset):
    """Dataset of (state, action, next_state) transitions."""

    def __init__(self, states, actions, next_states):
        self.states = torch.FloatTensor(states)
        self.actions = torch.LongTensor(actions)
        self.next_states = torch.FloatTensor(next_states)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.next_states[idx]


def prepare_transition_data(
    state_snapshots: list,
    historical_activity: list,
) -> tuple:
    """Build transition tuples from historical data.

    Since we don't have true sequential state transitions, we synthesize them
    by applying small perturbations based on actions taken. This gives the
    dynamics model reasonable training signal.
    """
    # Build patient state lookup
    patient_states = {}
    for snap in state_snapshots:
        vec = snapshot_to_vector(snap)
        patient_states[snap["patient_id"]] = (vec, snap)

    states, actions, next_states = [], [], []
    rng = np.random.default_rng(42)

    for record in historical_activity:
        pid = record["patient_id"]
        if pid not in patient_states:
            continue

        state_vec, snap = patient_states[pid]
        action_id = record["action_id"]

        # Synthesize next state: small perturbation based on outcome
        delta = rng.normal(0, 0.01, size=STATE_DIM).astype(np.float32)

        # If gap was closed, update the gap flag
        if record["outcome"]["gap_closed_within_30d"]:
            from config import HEDIS_MEASURES, FEAT_IDX_GAP_FLAGS_START
            measure = record["measure"]
            if measure in HEDIS_MEASURES:
                gap_idx = FEAT_IDX_GAP_FLAGS_START + HEDIS_MEASURES.index(measure)
                delta[gap_idx] = -state_vec[gap_idx]  # Set to 0 (closed)

        # Engagement affects response rates slightly (Tier 3: engagement rates at indices 119-123)
        from config import FEAT_IDX_ENGAGEMENT_START
        if record["outcome"]["clicked"]:
            delta[rng.integers(FEAT_IDX_ENGAGEMENT_START, FEAT_IDX_ENGAGEMENT_START + 5)] += 0.02

        # Contact count increases (Tier 3: patient_contacts_30d at index 110)
        from config import FEAT_IDX_CONTACT_HISTORY_START
        delta[FEAT_IDX_CONTACT_HISTORY_START + 4] = min(0.05, delta[FEAT_IDX_CONTACT_HISTORY_START + 4] + 0.05)

        # Days since contact resets (Tier 3: patient_days_since_last_contact at index 111)
        delta[FEAT_IDX_CONTACT_HISTORY_START + 5] = -state_vec[FEAT_IDX_CONTACT_HISTORY_START + 5]

        next_state = np.clip(state_vec + delta, -5.0, 5.0)
        states.append(state_vec)
        actions.append(action_id)
        next_states.append(next_state)

    return np.array(states), np.array(actions), np.array(next_states)


def train_dynamics_model(
    state_snapshots: list = None,
    historical_activity: list = None,
    model: DynamicsModel = None,
    epochs: int = None,
    batch_size: int = None,
    lr: float = None,
    verbose: bool = True,
) -> DynamicsModel:
    """Train the dynamics model on transition data."""
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

    if verbose:
        print("Preparing transition data...")
    states, actions, next_states = prepare_transition_data(state_snapshots, historical_activity)
    if verbose:
        print(f"  {len(states)} transitions prepared")

    dataset = TransitionDataset(states, actions, next_states)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if model is None:
        model = DynamicsModel()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for state_batch, action_batch, next_state_batch in dataloader:
            optimizer.zero_grad()
            loss = model.compute_loss(state_batch, action_batch, next_state_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if verbose and (epoch + 1) % 10 == 0:
            avg_loss = total_loss / max(n_batches, 1)
            print(f"  Epoch {epoch + 1}/{epochs} — loss: {avg_loss:.6f}")

    if verbose:
        print("Dynamics model training complete.")
    return model
