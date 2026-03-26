"""
Actor-Critic Conservative Q-Learning (CQL) with action masking.

Implements CQL-SAC: Soft Actor-Critic backbone with CQL conservative penalty.
- Actor: policy network outputting action probabilities (with masking)
- Critic: twin Q-networks with CQL regularization
- Temperature: auto-tuned entropy coefficient (alpha)

Based on Kumar et al., "Conservative Q-Learning for Offline Reinforcement Learning"
with the SAC (actor-critic) formulation for discrete action spaces.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict
import copy

from config import STATE_DIM, NUM_ACTIONS, CQL_CONFIG, GENERATED_DATA_DIR
from training.data_loader import load_datasets, build_offline_episodes


class Actor(nn.Module):
    """Policy network: outputs action log-probabilities with action masking."""

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

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor = None):
        """Return action log-probabilities.

        Args:
            obs: (batch, state_dim)
            action_mask: (batch, num_actions) boolean mask. True = valid.

        Returns:
            log_probs: (batch, num_actions)
            probs: (batch, num_actions)
        """
        logits = self.network(obs)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), float("-inf"))

        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        # Clamp for numerical stability
        probs = probs.clamp(min=1e-8)
        log_probs = torch.log(probs)
        return log_probs, probs

    def get_action(self, obs: np.ndarray, action_mask: np.ndarray = None) -> int:
        """Sample action for inference."""
        self.eval()
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            mask_t = None
            if action_mask is not None:
                mask_t = torch.FloatTensor(action_mask).unsqueeze(0)
            _, probs = self.forward(obs_t, mask_t)
            return torch.multinomial(probs, 1).item()

    def get_action_greedy(self, obs: np.ndarray, action_mask: np.ndarray = None) -> int:
        """Select best action greedily for inference."""
        self.eval()
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            mask_t = None
            if action_mask is not None:
                mask_t = torch.FloatTensor(action_mask).unsqueeze(0)
            _, probs = self.forward(obs_t, mask_t)
            return probs.argmax(dim=-1).item()


class TwinCritic(nn.Module):
    """Twin Q-networks for SAC-style critic with CQL penalty."""

    def __init__(self, state_dim: int = STATE_DIM, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, num_actions),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, num_actions),
        )

    def forward(self, obs: torch.Tensor):
        """Return Q-values from both critics.

        Returns:
            q1: (batch, num_actions)
            q2: (batch, num_actions)
        """
        return self.q1(obs), self.q2(obs)

    def q_min(self, obs: torch.Tensor):
        """Return minimum Q-values across twin critics (pessimistic)."""
        q1, q2 = self.forward(obs)
        return torch.min(q1, q2)


class ActorCriticCQL:
    """Actor-Critic CQL agent combining Actor, TwinCritic, and auto-tuned alpha."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        num_actions: int = NUM_ACTIONS,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.97,
        tau: float = 0.005,
        min_q_weight: float = 5.0,
        target_entropy: float = None,
    ):
        self.gamma = gamma
        self.tau = tau
        self.min_q_weight = min_q_weight
        self.num_actions = num_actions

        # Networks
        self.actor = Actor(state_dim, num_actions)
        self.critic = TwinCritic(state_dim, num_actions)
        self.critic_target = copy.deepcopy(self.critic)

        # Freeze target
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Auto-tuned entropy coefficient
        if target_entropy is None:
            self.target_entropy = -np.log(1.0 / num_actions) * 0.5  # Half of max entropy
        else:
            self.target_entropy = target_entropy
        self.log_alpha = torch.zeros(1, requires_grad=True)

        # Lagrangian CQL: auto-tune the conservative penalty weight
        self.use_lagrangian = CQL_CONFIG.get("lagrangian", False)
        self.log_cql_alpha = torch.zeros(1, requires_grad=True)
        self.cql_target_penalty = min_q_weight  # Target CQL penalty magnitude

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
        self.cql_alpha_optimizer = torch.optim.Adam([self.log_cql_alpha], lr=lr_critic)

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    def update_critic(self, obs, actions, rewards, next_obs, dones, masks, next_masks):
        """Update twin critics with TD loss + CQL penalty."""
        q1, q2 = self.critic(obs)
        q1_selected = q1.gather(1, actions.unsqueeze(1)).squeeze(1)
        q2_selected = q2.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (SAC-style with entropy)
        with torch.no_grad():
            next_log_probs, next_probs = self.actor(next_obs, next_masks)
            next_q1_target, next_q2_target = self.critic_target(next_obs)
            next_q_min = torch.min(next_q1_target, next_q2_target)

            # V(s') = sum_a pi(a|s') * (Q(s',a) - alpha * log pi(a|s'))
            next_v = (next_probs * (next_q_min - self.alpha * next_log_probs)).sum(dim=-1)
            # Handle -inf log_probs from masked actions
            next_v = torch.where(torch.isnan(next_v), torch.zeros_like(next_v), next_v)

            td_target = rewards + self.gamma * (1 - dones) * next_v

        td_loss1 = F.mse_loss(q1_selected, td_target)
        td_loss2 = F.mse_loss(q2_selected, td_target)

        # CQL penalty: penalize Q-values that are too high for OOD actions
        # Only apply to valid (masked) actions
        masked_q1 = q1.clone()
        masked_q2 = q2.clone()
        masked_q1 = masked_q1.masked_fill(~masks.bool(), float("-inf"))
        masked_q2 = masked_q2.masked_fill(~masks.bool(), float("-inf"))

        # LogSumExp over valid actions
        cql_q1 = torch.logsumexp(masked_q1, dim=1).mean() - q1_selected.mean()
        cql_q2 = torch.logsumexp(masked_q2, dim=1).mean() - q2_selected.mean()

        cql_penalty = cql_q1 + cql_q2

        if self.use_lagrangian:
            # Lagrangian dual: auto-tune CQL weight so penalty stays near target
            cql_alpha = self.log_cql_alpha.exp().clamp(min=0.0)
            critic_loss = td_loss1 + td_loss2 + cql_alpha * cql_penalty

            # Update CQL alpha: increase if penalty > target, decrease otherwise
            cql_alpha_loss = -self.log_cql_alpha.exp() * (cql_penalty.detach() - self.cql_target_penalty)
            self.cql_alpha_optimizer.zero_grad()
            cql_alpha_loss.backward()
            self.cql_alpha_optimizer.step()
        else:
            critic_loss = td_loss1 + td_loss2 + self.min_q_weight * cql_penalty

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        return {
            "td_loss": (td_loss1.item() + td_loss2.item()) / 2,
            "cql_penalty": (cql_q1.item() + cql_q2.item()) / 2,
            "cql_alpha": self.log_cql_alpha.exp().item() if self.use_lagrangian else self.min_q_weight,
            "critic_loss": critic_loss.item(),
            # Q-value diagnostics: detect explosion/collapse
            "q_mean": q1_selected.mean().item(),
            "q_min": q1_selected.min().item(),
            "q_max": q1_selected.max().item(),
            "td_target_mean": td_target.mean().item(),
            "td_target_std": td_target.std().item(),
        }

    def update_actor(self, obs, masks):
        """Update actor to maximize expected Q-value + entropy."""
        log_probs, probs = self.actor(obs, masks)
        q_min = self.critic.q_min(obs)

        # Actor loss: minimize -E[Q(s,a)] + alpha * H(pi)
        # = E_a~pi [alpha * log pi(a|s) - Q(s,a)]
        actor_loss = (probs * (self.alpha * log_probs - q_min)).sum(dim=-1)
        # Handle NaN from masked actions
        actor_loss = torch.where(torch.isnan(actor_loss), torch.zeros_like(actor_loss), actor_loss)
        actor_loss = actor_loss.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        return {"actor_loss": actor_loss.item()}

    def update_alpha(self, obs, masks):
        """Auto-tune entropy coefficient."""
        with torch.no_grad():
            log_probs, probs = self.actor(obs, masks)
            entropy = -(probs * log_probs).sum(dim=-1)
            entropy = torch.where(torch.isnan(entropy), torch.zeros_like(entropy), entropy)

        alpha_loss = (self.log_alpha * (entropy - self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return {"alpha": self.alpha.item(), "entropy": entropy.mean().item()}

    def soft_update_target(self):
        """Soft update target critic."""
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def get_action(self, obs, mask=None):
        return self.actor.get_action(obs, mask)

    def get_action_greedy(self, obs, mask=None):
        return self.actor.get_action_greedy(obs, mask)

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.data,
            "log_cql_alpha": self.log_cql_alpha.data,
        }

    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.critic_target.load_state_dict(state_dict["critic_target"])
        self.log_alpha.data = state_dict["log_alpha"]
        if "log_cql_alpha" in state_dict:
            self.log_cql_alpha.data = state_dict["log_cql_alpha"]


class CQLDataset(Dataset):
    """Dataset for CQL training."""

    def __init__(self, obs, actions, rewards, next_obs, dones, masks, next_masks):
        self.obs = torch.FloatTensor(obs)
        self.actions = torch.LongTensor(actions)
        self.rewards = torch.FloatTensor(rewards)
        self.next_obs = torch.FloatTensor(next_obs)
        self.dones = torch.FloatTensor(dones)
        self.masks = torch.FloatTensor(masks)
        self.next_masks = torch.FloatTensor(next_masks)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return (
            self.obs[idx], self.actions[idx], self.rewards[idx],
            self.next_obs[idx], self.dones[idx],
            self.masks[idx], self.next_masks[idx],
        )


def train_cql(
    episodes: list = None,
    agent: ActorCriticCQL = None,
    bc_policy=None,
    epochs: int = None,
    batch_size: int = 256,
    lr: float = None,
    min_q_weight: float = None,
    gamma: float = 0.97,
    verbose: bool = True,
) -> ActorCriticCQL:
    """Train Actor-Critic CQL on offline episodes.

    Args:
        episodes: List of episode dicts from build_offline_episodes.
        agent: ActorCriticCQL agent. If None, creates new one.
        bc_policy: Optional BC policy to initialize actor from.
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        min_q_weight: CQL conservatism weight (alpha_cql).
        gamma: Discount factor.
        verbose: Print progress.

    Returns:
        Trained ActorCriticCQL agent.
    """
    if epochs is None:
        epochs = CQL_CONFIG["cql_iters"]
    if lr is None:
        lr = CQL_CONFIG["lr"]
    if min_q_weight is None:
        min_q_weight = CQL_CONFIG["min_q_weight"]

    if episodes is None:
        datasets = load_datasets()
        episodes = build_offline_episodes(
            datasets["state_features"],
            datasets["historical_activity"],
            datasets["action_eligibility"],
        )

    # Flatten episodes to transitions
    all_obs, all_actions, all_rewards = [], [], []
    all_next_obs, all_dones = [], []
    all_masks, all_next_masks = [], []

    for ep in episodes:
        T = len(ep["obs"])
        for t in range(T - 1):
            all_obs.append(ep["obs"][t])
            all_actions.append(ep["actions"][t])
            all_rewards.append(ep["rewards"][t])
            all_next_obs.append(ep["obs"][t + 1])
            all_dones.append(float(ep["terminateds"][t] or ep["truncateds"][t]))
            all_masks.append(ep["action_mask"][t])
            all_next_masks.append(ep["action_mask"][min(t + 1, T - 1)])

    dataset = CQLDataset(
        np.array(all_obs), np.array(all_actions), np.array(all_rewards),
        np.array(all_next_obs), np.array(all_dones),
        np.array(all_masks), np.array(all_next_masks),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if verbose:
        print(f"CQL-SAC training: {len(dataset)} transitions, {epochs} epochs")

    # Initialize agent or update existing agent's hyperparameters
    if agent is None:
        agent = ActorCriticCQL(
            lr_actor=lr, lr_critic=lr, lr_alpha=lr,
            gamma=gamma, min_q_weight=min_q_weight,
        )
    else:
        # Ensure existing agent uses current config values (not stale from init)
        agent.gamma = gamma
        agent.min_q_weight = min_q_weight
        agent.cql_target_penalty = min_q_weight
        agent.use_lagrangian = CQL_CONFIG.get("lagrangian", False)
        # Update ALL optimizer learning rates
        for param_group in agent.actor_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in agent.critic_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in agent.alpha_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in agent.cql_alpha_optimizer.param_groups:
            param_group['lr'] = lr

    # Initialize actor from BC policy if provided
    if bc_policy is not None:
        try:
            bc_state = bc_policy.state_dict()
            actor_state = agent.actor.state_dict()
            compatible = {k: v for k, v in bc_state.items() if k in actor_state and v.shape == actor_state[k].shape}
            agent.actor.load_state_dict(compatible, strict=False)
            if verbose:
                print(f"  Initialized actor with {len(compatible)} layers from BC policy")
        except Exception as e:
            if verbose:
                print(f"  Could not initialize from BC: {e}")

    agent.actor.train()
    agent.critic.train()

    # Track per-step training metrics for debugging
    training_history = []  # Per-epoch averages (for nightly trend)
    step_history = []      # Per-batch-step (for per-night drill-down)
    global_step = 0

    for epoch in range(epochs):
        total_critic_loss = 0.0
        total_actor_loss = 0.0
        total_td_loss = 0.0
        total_cql = 0.0
        total_q_mean = 0.0
        total_q_min = 0.0
        total_q_max = 0.0
        total_td_target_mean = 0.0
        n_batches = 0
        last_alpha = 0.0
        last_entropy = 0.0
        last_cql_alpha = 0.0

        for batch in dataloader:
            obs_b, act_b, rew_b, next_obs_b, done_b, mask_b, next_mask_b = batch

            critic_info = agent.update_critic(obs_b, act_b, rew_b, next_obs_b, done_b, mask_b, next_mask_b)
            actor_info = agent.update_actor(obs_b, mask_b)
            alpha_info = agent.update_alpha(obs_b, mask_b)
            agent.soft_update_target()

            total_critic_loss += critic_info["critic_loss"]
            total_td_loss += critic_info["td_loss"]
            total_actor_loss += actor_info["actor_loss"]
            total_cql += critic_info["cql_penalty"]
            total_q_mean += critic_info.get("q_mean", 0.0)
            total_q_min += critic_info.get("q_min", 0.0)
            total_q_max += critic_info.get("q_max", 0.0)
            total_td_target_mean += critic_info.get("td_target_mean", 0.0)
            last_alpha = alpha_info["alpha"]
            last_entropy = alpha_info["entropy"]
            last_cql_alpha = critic_info.get("cql_alpha", agent.min_q_weight)
            n_batches += 1
            global_step += 1

            # Log every 10th step to keep data manageable
            if global_step % 10 == 0:
                step_history.append({
                    "step": global_step,
                    "critic": critic_info["critic_loss"],
                    "td": critic_info["td_loss"],
                    "actor": actor_info["actor_loss"],
                    "cql": critic_info["cql_penalty"],
                    "alpha": last_alpha,
                    "entropy": last_entropy,
                    "q_mean": critic_info.get("q_mean", 0.0),
                    "q_min": critic_info.get("q_min", 0.0),
                    "q_max": critic_info.get("q_max", 0.0),
                })

        epoch_metrics = {
            "epoch": epoch + 1,
            "critic_loss": total_critic_loss / max(n_batches, 1),
            "td_loss": total_td_loss / max(n_batches, 1),
            "actor_loss": total_actor_loss / max(n_batches, 1),
            "cql_penalty": total_cql / max(n_batches, 1),
            "q_mean": total_q_mean / max(n_batches, 1),
            "q_min": total_q_min / max(n_batches, 1),
            "q_max": total_q_max / max(n_batches, 1),
            "td_target_mean": total_td_target_mean / max(n_batches, 1),
            "alpha": last_alpha,
            "entropy": last_entropy,
            "cql_alpha": last_cql_alpha,
        }
        training_history.append(epoch_metrics)

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}/{epochs} — "
                  f"critic: {epoch_metrics['critic_loss']:.4f}, "
                  f"td: {epoch_metrics['td_loss']:.4f}, "
                  f"actor: {epoch_metrics['actor_loss']:.4f}, "
                  f"cql: {epoch_metrics['cql_penalty']:.4f} (α={last_cql_alpha:.3f}), "
                  f"Q[mean={epoch_metrics['q_mean']:.2f} min={epoch_metrics['q_min']:.2f} max={epoch_metrics['q_max']:.2f}], "
                  f"target_mean={epoch_metrics['td_target_mean']:.2f}, "
                  f"entropy={last_entropy:.2f}")

    if verbose:
        print("CQL-SAC training complete.")

    # Attach training history to agent for retrieval
    agent._training_history = training_history
    agent._step_history = step_history
    return agent
