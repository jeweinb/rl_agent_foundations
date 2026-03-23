"""
Behavior cloning (BC) training using a simple PyTorch policy network.
Phase 1: Learn from historical behavioral policy before CQL fine-tuning.

Uses a standalone PyTorch implementation rather than RLlib's MARWIL
for simplicity and reliability on local deployment.
"""
import json
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional

from config import (
    STATE_DIM, NUM_ACTIONS, CQL_CONFIG, GENERATED_DATA_DIR, CHECKPOINTS_DIR,
)
from training.data_loader import load_datasets, build_offline_episodes


class ActionMaskedPolicy(nn.Module):
    """Policy network with action masking support."""

    def __init__(self, state_dim: int = STATE_DIM, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, num_actions),
        )

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor = None) -> torch.Tensor:
        """Return action logits with optional masking."""
        logits = self.network(obs)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), float("-inf"))
        return logits

    def get_action(self, obs: np.ndarray, action_mask: np.ndarray = None) -> int:
        """Select action for inference."""
        self.eval()
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            mask_t = None
            if action_mask is not None:
                mask_t = torch.FloatTensor(action_mask).unsqueeze(0)
            logits = self.forward(obs_t, mask_t)
            probs = torch.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1).item()

    def get_action_greedy(self, obs: np.ndarray, action_mask: np.ndarray = None) -> int:
        """Select best action greedily for inference."""
        self.eval()
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            mask_t = None
            if action_mask is not None:
                mask_t = torch.FloatTensor(action_mask).unsqueeze(0)
            logits = self.forward(obs_t, mask_t)
            return logits.argmax(dim=-1).item()


class BCDataset(Dataset):
    """Dataset for behavior cloning."""

    def __init__(self, obs, actions, masks):
        self.obs = torch.FloatTensor(obs)
        self.actions = torch.LongTensor(actions)
        self.masks = torch.FloatTensor(masks)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx], self.masks[idx]


def train_behavior_cloning(
    episodes: list = None,
    model: ActionMaskedPolicy = None,
    epochs: int = None,
    batch_size: int = 256,
    lr: float = None,
    verbose: bool = True,
) -> ActionMaskedPolicy:
    """Train a behavior cloning policy from offline episodes."""
    if epochs is None:
        epochs = CQL_CONFIG["bc_iters"]
    if lr is None:
        lr = CQL_CONFIG["lr"]

    if episodes is None:
        datasets = load_datasets()
        episodes = build_offline_episodes(
            datasets["state_features"],
            datasets["historical_activity"],
            datasets["action_eligibility"],
        )

    # Flatten episodes into training data
    all_obs, all_actions, all_masks = [], [], []
    for ep in episodes:
        all_obs.append(ep["obs"])
        all_actions.append(ep["actions"])
        all_masks.append(ep["action_mask"])

    obs = np.concatenate(all_obs)
    actions = np.concatenate(all_actions)
    masks = np.concatenate(all_masks)

    if verbose:
        print(f"BC training data: {len(obs)} transitions from {len(episodes)} episodes")

    dataset = BCDataset(obs, actions, masks)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if model is None:
        model = ActionMaskedPolicy()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for obs_batch, action_batch, mask_batch in dataloader:
            optimizer.zero_grad()
            logits = model.forward(obs_batch, mask_batch)
            loss = criterion(logits, action_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == action_batch).sum().item()
            total += len(action_batch)

        if verbose and (epoch + 1) % 10 == 0:
            avg_loss = total_loss / max(len(dataloader), 1)
            acc = correct / max(total, 1)
            print(f"  Epoch {epoch + 1}/{epochs} — loss: {avg_loss:.4f}, accuracy: {acc:.4f}")

    if verbose:
        print("Behavior cloning training complete.")
    return model
