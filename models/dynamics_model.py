"""
Learned dynamics model: predicts next state given (state, action).
s_{t+1} = s_t + f_theta(s_t, embed(a_t))
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from config import STATE_DIM, NUM_ACTIONS, DYNAMICS_MODEL_CONFIG as CFG


class DynamicsModel(nn.Module):
    """Neural network that predicts state deltas: delta_s = f(s, a)."""

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
        input_dim = state_dim + action_embed_dim

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

        # Output: predicted state delta + log_variance for uncertainty
        self.network = nn.Sequential(*layers)
        self.delta_head = nn.Linear(prev_dim, state_dim)
        self.logvar_head = nn.Linear(prev_dim, state_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """
        Args:
            state: (batch, state_dim)
            action: (batch,) integer action indices

        Returns:
            delta_mean: (batch, state_dim) predicted state delta
            delta_logvar: (batch, state_dim) log variance for uncertainty
        """
        action_embed = self.action_embedding(action)
        x = torch.cat([state, action_embed], dim=-1)
        h = self.network(x)
        delta_mean = self.delta_head(h)
        delta_logvar = self.logvar_head(h)
        return delta_mean, delta_logvar

    def predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
        add_noise: bool = True,
    ) -> np.ndarray:
        """Predict next state from numpy inputs.

        Args:
            state: (batch, state_dim) or (state_dim,)
            action: (batch,) or scalar integer action index

        Returns:
            next_state: (batch, state_dim) numpy array
        """
        self.eval()
        with torch.no_grad():
            if state.ndim == 1:
                state = state.reshape(1, -1)
            if np.isscalar(action) or action.ndim == 0:
                action = np.array([action])

            state_t = torch.FloatTensor(state)
            action_t = torch.LongTensor(action)

            delta_mean, delta_logvar = self.forward(state_t, action_t)

            if add_noise:
                std = torch.exp(0.5 * delta_logvar)
                noise = torch.randn_like(std) * std * 0.1  # Small noise
                delta = delta_mean + noise
            else:
                delta = delta_mean

            next_state = state_t + delta
            return next_state.numpy()

    def compute_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Gaussian NLL loss for state prediction."""
        delta_mean, delta_logvar = self.forward(state, action)
        target_delta = next_state - state

        # Gaussian negative log-likelihood
        var = torch.exp(delta_logvar)
        loss = 0.5 * (delta_logvar + (target_delta - delta_mean) ** 2 / var)
        return loss.mean()
