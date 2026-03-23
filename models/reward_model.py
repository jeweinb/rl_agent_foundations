"""
Learned reward model: predicts P(gap_closure | state, action, days_elapsed).
Bridges the gap between action and lagged reward observation.
"""
import torch
import torch.nn as nn
import numpy as np

from config import STATE_DIM, NUM_ACTIONS, REWARD_MODEL_CONFIG as CFG


class RewardModel(nn.Module):
    """Predicts gap closure probability from (state, action, days_elapsed)."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        num_actions: int = NUM_ACTIONS,
        action_embed_dim: int = CFG["action_embed_dim"],
        hidden_dims: list = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = CFG["hidden_dims"]

        self.action_embedding = nn.Embedding(num_actions, action_embed_dim)
        input_dim = state_dim + action_embed_dim + 1  # +1 for days_elapsed

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.LayerNorm(h_dim),
                nn.Dropout(0.1),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        days_elapsed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            state: (batch, state_dim)
            action: (batch,) integer action indices
            days_elapsed: (batch, 1) or (batch,) days since action

        Returns:
            closure_prob: (batch, 1) sigmoid output in [0, 1]
        """
        action_embed = self.action_embedding(action)
        if days_elapsed.ndim == 1:
            days_elapsed = days_elapsed.unsqueeze(-1)
        # Normalize days
        days_norm = days_elapsed / 90.0

        x = torch.cat([state, action_embed, days_norm], dim=-1)
        logit = self.network(x)
        return torch.sigmoid(logit)

    def predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
        days_elapsed: np.ndarray,
    ) -> np.ndarray:
        """Predict closure probability from numpy inputs."""
        self.eval()
        with torch.no_grad():
            if state.ndim == 1:
                state = state.reshape(1, -1)
            if np.isscalar(action) or action.ndim == 0:
                action = np.array([action])
            if np.isscalar(days_elapsed) or days_elapsed.ndim == 0:
                days_elapsed = np.array([days_elapsed], dtype=np.float32)

            state_t = torch.FloatTensor(state)
            action_t = torch.LongTensor(action)
            days_t = torch.FloatTensor(days_elapsed)

            prob = self.forward(state_t, action_t, days_t)
            return prob.numpy().flatten()

    def compute_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        days_elapsed: torch.Tensor,
        gap_closed: torch.Tensor,
    ) -> torch.Tensor:
        """Binary cross-entropy loss for gap closure prediction."""
        pred = self.forward(state, action, days_elapsed)
        target = gap_closed.float().unsqueeze(-1) if gap_closed.ndim == 1 else gap_closed.float()
        return nn.functional.binary_cross_entropy(pred, target)
