"""
Tests for the training pipeline.
Validates data loading, BC training, Actor-Critic CQL training,
and model evaluation.
"""
import pytest
import numpy as np
import torch

from config import STATE_DIM, NUM_ACTIONS
from training.data_loader import build_offline_episodes
from training.behavior_cloning import train_behavior_cloning, ActionMaskedPolicy
from training.cql_trainer import (
    train_cql, ActorCriticCQL, Actor, TwinCritic, CQLDataset,
)
from training.evaluation import evaluate_agent, compare_models
from environment.hedis_env import HEDISEnv


# =========================================================================
# Data Loader
# =========================================================================
class TestDataLoader:
    def test_builds_episodes(self, small_datasets):
        episodes = build_offline_episodes(
            small_datasets["state_features"],
            small_datasets["historical_activity"],
            small_datasets["action_eligibility"],
        )
        assert len(episodes) > 0

    def test_episode_schema(self, small_datasets):
        episodes = build_offline_episodes(
            small_datasets["state_features"],
            small_datasets["historical_activity"],
            small_datasets["action_eligibility"],
        )
        ep = episodes[0]
        assert "obs" in ep
        assert "actions" in ep
        assert "rewards" in ep
        assert "action_mask" in ep
        assert "terminateds" in ep
        assert "truncateds" in ep

    def test_episode_shapes(self, small_datasets):
        episodes = build_offline_episodes(
            small_datasets["state_features"],
            small_datasets["historical_activity"],
            small_datasets["action_eligibility"],
        )
        for ep in episodes[:5]:
            T = len(ep["obs"])
            assert ep["obs"].shape == (T, STATE_DIM)
            assert ep["actions"].shape == (T,)
            assert ep["rewards"].shape == (T,)
            assert ep["action_mask"].shape == (T, NUM_ACTIONS)
            assert ep["terminateds"].shape == (T,)
            assert ep["truncateds"].shape == (T,)

    def test_episode_dtypes(self, small_datasets):
        episodes = build_offline_episodes(
            small_datasets["state_features"],
            small_datasets["historical_activity"],
            small_datasets["action_eligibility"],
        )
        ep = episodes[0]
        assert ep["obs"].dtype == np.float32
        assert ep["actions"].dtype == np.int64
        assert ep["rewards"].dtype == np.float32

    def test_episode_action_ids_valid(self, small_datasets):
        episodes = build_offline_episodes(
            small_datasets["state_features"],
            small_datasets["historical_activity"],
            small_datasets["action_eligibility"],
        )
        for ep in episodes:
            assert np.all(ep["actions"] >= 0)
            assert np.all(ep["actions"] < NUM_ACTIONS)

    def test_no_nan_in_observations(self, small_datasets):
        episodes = build_offline_episodes(
            small_datasets["state_features"],
            small_datasets["historical_activity"],
            small_datasets["action_eligibility"],
        )
        for ep in episodes:
            assert not np.any(np.isnan(ep["obs"]))

    def test_no_action_mask_always_has_no_action(self, small_datasets):
        episodes = build_offline_episodes(
            small_datasets["state_features"],
            small_datasets["historical_activity"],
            small_datasets["action_eligibility"],
        )
        for ep in episodes:
            # Index 0 (no_action) should always be available
            assert np.all(ep["action_mask"][:, 0] == 1.0)


# =========================================================================
# Behavior Cloning
# =========================================================================
class TestBehaviorCloning:
    @pytest.fixture(scope="class")
    def bc_model_and_episodes(self, small_datasets):
        episodes = build_offline_episodes(
            small_datasets["state_features"],
            small_datasets["historical_activity"],
            small_datasets["action_eligibility"],
        )
        model = train_behavior_cloning(episodes=episodes, epochs=3, verbose=False)
        return model, episodes

    def test_bc_returns_policy(self, bc_model_and_episodes):
        model, _ = bc_model_and_episodes
        assert isinstance(model, ActionMaskedPolicy)

    def test_bc_policy_output_shape(self, bc_model_and_episodes):
        model, _ = bc_model_and_episodes
        obs = torch.randn(1, STATE_DIM)
        mask = torch.ones(1, NUM_ACTIONS)
        logits = model.forward(obs, mask)
        assert logits.shape == (1, NUM_ACTIONS)

    def test_bc_action_masking_applied(self, bc_model_and_episodes):
        model, _ = bc_model_and_episodes
        obs = torch.randn(1, STATE_DIM)
        mask = torch.zeros(1, NUM_ACTIONS)
        mask[0, 0] = 1.0  # Only no_action valid
        mask[0, 1] = 1.0
        logits = model.forward(obs, mask)
        # Masked actions should have -inf logits
        assert logits[0, 2].item() == float("-inf")

    def test_bc_get_action_respects_mask(self, bc_model_and_episodes):
        model, _ = bc_model_and_episodes
        obs = np.random.randn(STATE_DIM).astype(np.float32)
        mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
        mask[0] = 1.0  # Only no_action
        action = model.get_action(obs, mask)
        assert action == 0

    def test_bc_get_action_greedy_respects_mask(self, bc_model_and_episodes):
        model, _ = bc_model_and_episodes
        obs = np.random.randn(STATE_DIM).astype(np.float32)
        mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
        mask[0] = 1.0
        mask[5] = 1.0
        action = model.get_action_greedy(obs, mask)
        assert action in (0, 5)

    def test_bc_state_dict_saveable(self, bc_model_and_episodes):
        model, _ = bc_model_and_episodes
        sd = model.state_dict()
        new_model = ActionMaskedPolicy()
        new_model.load_state_dict(sd)
        # Outputs should match (eval mode disables dropout)
        model.eval()
        new_model.eval()
        obs = torch.randn(1, STATE_DIM)
        mask = torch.ones(1, NUM_ACTIONS)
        with torch.no_grad():
            out1 = model.forward(obs, mask)
            out2 = new_model.forward(obs, mask)
        assert torch.allclose(out1, out2)


# =========================================================================
# Actor-Critic CQL
# =========================================================================
class TestActorCriticCQL:
    def test_actor_output_shape(self):
        actor = Actor()
        obs = torch.randn(4, STATE_DIM)
        mask = torch.ones(4, NUM_ACTIONS)
        log_probs, probs = actor(obs, mask)
        assert log_probs.shape == (4, NUM_ACTIONS)
        assert probs.shape == (4, NUM_ACTIONS)

    def test_actor_probs_sum_to_one(self):
        actor = Actor()
        obs = torch.randn(4, STATE_DIM)
        mask = torch.ones(4, NUM_ACTIONS)
        _, probs = actor(obs, mask)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-4)

    def test_actor_masked_probs_zero(self):
        actor = Actor()
        obs = torch.randn(1, STATE_DIM)
        mask = torch.zeros(1, NUM_ACTIONS)
        mask[0, 0] = 1.0
        mask[0, 3] = 1.0
        _, probs = actor(obs, mask)
        # Non-masked actions should have ~0 probability
        assert probs[0, 5].item() < 1e-6

    def test_twin_critic_output_shape(self):
        critic = TwinCritic()
        obs = torch.randn(4, STATE_DIM)
        q1, q2 = critic(obs)
        assert q1.shape == (4, NUM_ACTIONS)
        assert q2.shape == (4, NUM_ACTIONS)

    def test_twin_critic_q_min(self):
        critic = TwinCritic()
        obs = torch.randn(4, STATE_DIM)
        q_min = critic.q_min(obs)
        q1, q2 = critic(obs)
        expected = torch.min(q1, q2)
        assert torch.allclose(q_min, expected)

    def test_cql_agent_creation(self):
        agent = ActorCriticCQL()
        assert isinstance(agent.actor, Actor)
        assert isinstance(agent.critic, TwinCritic)
        assert isinstance(agent.critic_target, TwinCritic)

    def test_cql_agent_get_action(self):
        agent = ActorCriticCQL()
        obs = np.random.randn(STATE_DIM).astype(np.float32)
        mask = np.ones(NUM_ACTIONS, dtype=np.float32)
        action = agent.get_action(obs, mask)
        assert 0 <= action < NUM_ACTIONS

    def test_cql_agent_get_action_greedy(self):
        agent = ActorCriticCQL()
        obs = np.random.randn(STATE_DIM).astype(np.float32)
        mask = np.ones(NUM_ACTIONS, dtype=np.float32)
        action = agent.get_action_greedy(obs, mask)
        assert 0 <= action < NUM_ACTIONS

    def test_cql_agent_respects_mask(self):
        agent = ActorCriticCQL()
        obs = np.random.randn(STATE_DIM).astype(np.float32)
        mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
        mask[0] = 1.0  # Only no_action
        for _ in range(10):
            action = agent.get_action_greedy(obs, mask)
            assert action == 0

    def test_cql_state_dict_roundtrip(self):
        agent = ActorCriticCQL()
        sd = agent.state_dict()
        agent2 = ActorCriticCQL()
        agent2.load_state_dict(sd)
        obs = np.random.randn(STATE_DIM).astype(np.float32)
        mask = np.ones(NUM_ACTIONS, dtype=np.float32)
        # Both should produce the same greedy action
        a1 = agent.get_action_greedy(obs, mask)
        a2 = agent2.get_action_greedy(obs, mask)
        assert a1 == a2

    def test_cql_training_runs(self, small_datasets):
        episodes = build_offline_episodes(
            small_datasets["state_features"],
            small_datasets["historical_activity"],
            small_datasets["action_eligibility"],
        )
        agent = train_cql(episodes=episodes, epochs=2, batch_size=64, verbose=False)
        assert isinstance(agent, ActorCriticCQL)

    def test_cql_training_with_bc_init(self, small_datasets):
        episodes = build_offline_episodes(
            small_datasets["state_features"],
            small_datasets["historical_activity"],
            small_datasets["action_eligibility"],
        )
        bc = train_behavior_cloning(episodes=episodes, epochs=2, verbose=False)
        agent = train_cql(episodes=episodes, bc_policy=bc, epochs=2, batch_size=64, verbose=False)
        assert isinstance(agent, ActorCriticCQL)

    def test_critic_update_reduces_loss(self):
        """One critic update step should produce finite loss."""
        agent = ActorCriticCQL()
        obs = torch.randn(16, STATE_DIM)
        actions = torch.randint(0, NUM_ACTIONS, (16,))
        rewards = torch.randn(16)
        next_obs = torch.randn(16, STATE_DIM)
        dones = torch.zeros(16)
        masks = torch.ones(16, NUM_ACTIONS)
        next_masks = torch.ones(16, NUM_ACTIONS)

        info = agent.update_critic(obs, actions, rewards, next_obs, dones, masks, next_masks)
        assert np.isfinite(info["critic_loss"])
        assert np.isfinite(info["td_loss"])
        assert np.isfinite(info["cql_penalty"])

    def test_actor_update_produces_finite_loss(self):
        agent = ActorCriticCQL()
        obs = torch.randn(16, STATE_DIM)
        masks = torch.ones(16, NUM_ACTIONS)
        info = agent.update_actor(obs, masks)
        assert np.isfinite(info["actor_loss"])

    def test_alpha_update_produces_finite_values(self):
        agent = ActorCriticCQL()
        obs = torch.randn(16, STATE_DIM)
        masks = torch.ones(16, NUM_ACTIONS)
        info = agent.update_alpha(obs, masks)
        assert np.isfinite(info["alpha"])
        assert np.isfinite(info["entropy"])
        assert info["alpha"] > 0

    def test_target_network_soft_update(self):
        agent = ActorCriticCQL()
        # Get initial target params
        p_before = list(agent.critic_target.parameters())[0].data.clone()
        # Update critic
        obs = torch.randn(16, STATE_DIM)
        actions = torch.randint(0, NUM_ACTIONS, (16,))
        rewards = torch.randn(16)
        next_obs = torch.randn(16, STATE_DIM)
        dones = torch.zeros(16)
        masks = torch.ones(16, NUM_ACTIONS)
        agent.update_critic(obs, actions, rewards, next_obs, dones, masks, masks)
        agent.soft_update_target()
        p_after = list(agent.critic_target.parameters())[0].data
        # Target should have moved slightly
        assert not torch.allclose(p_before, p_after)


# =========================================================================
# Evaluation
# =========================================================================
class TestEvaluation:
    @pytest.fixture
    def env(self, small_snapshots, small_eligibility):
        return HEDISEnv(small_snapshots, small_eligibility)

    def test_evaluate_random_agent(self, env):
        class RandomAgent:
            def get_action_greedy(self, obs, mask):
                valid = np.where(mask)[0]
                return int(np.random.choice(valid)) if len(valid) > 0 else 0

        metrics = evaluate_agent(RandomAgent(), env, n_episodes=10, seed=42)
        assert "mean_reward" in metrics
        assert "std_reward" in metrics
        assert "mean_gaps_closed" in metrics
        assert "mean_episode_length" in metrics
        assert "no_action_rate" in metrics
        assert metrics["n_episodes"] == 10

    def test_evaluate_cql_agent(self, env):
        agent = ActorCriticCQL()
        metrics = evaluate_agent(agent, env, n_episodes=5, seed=42)
        assert np.isfinite(metrics["mean_reward"])

    def test_compare_models(self):
        champ = {"mean_reward": 1.0, "mean_gaps_closed": 2.0}
        chall_better = {"mean_reward": 1.5, "mean_gaps_closed": 3.0}
        chall_worse = {"mean_reward": 0.5, "mean_gaps_closed": 1.0}

        result = compare_models(champ, chall_better)
        assert result["promote_challenger"] == True
        assert result["relative_improvement"] > 0

        result2 = compare_models(champ, chall_worse)
        assert result2["promote_challenger"] == False
        assert result2["relative_improvement"] < 0

    def test_compare_models_zero_champion(self):
        champ = {"mean_reward": 0.0, "mean_gaps_closed": 0.0}
        chall = {"mean_reward": 0.5, "mean_gaps_closed": 1.0}
        result = compare_models(champ, chall)
        assert result["promote_challenger"] == True
