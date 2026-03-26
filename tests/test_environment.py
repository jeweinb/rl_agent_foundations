"""
Tests for the environment modules.
Validates action masking, OOD action handling, state transitions,
reward computation, and gym env behavior.
"""
import pytest
import numpy as np
import json

from config import (
    NUM_ACTIONS, STATE_DIM, HEDIS_MEASURES, CHANNELS,
    ACTION_CATALOG, ACTION_BY_ID, MEASURE_WEIGHTS,
    REWARD_WEIGHTS, STARS_BONUS_THRESHOLD,
    MAX_CONTACTS_PER_WEEK, MIN_DAYS_BETWEEN_SAME_MEASURE,
)
from environment.action_space import (
    decode_action, is_no_action, get_action_measure,
    get_action_channel, get_action_info,
)
from environment.state_space import snapshot_to_vector, FEATURE_NAMES
from environment.action_masking import compute_action_mask, mask_from_eligibility_snapshot
from environment.reward import compute_reward, compute_stars_score
from environment.hedis_env import HEDISEnv


# =========================================================================
# Action Space
# =========================================================================
class TestActionSpace:
    def test_decode_no_action(self):
        a = decode_action(0)
        assert a.measure == "NO_ACTION"
        assert is_no_action(0)

    def test_decode_valid_action(self):
        a = decode_action(1)
        assert a.measure in HEDIS_MEASURES
        assert a.channel in CHANNELS
        assert not is_no_action(1)

    def test_decode_out_of_range_raises(self):
        with pytest.raises(ValueError):
            decode_action(-1)
        with pytest.raises(ValueError):
            decode_action(NUM_ACTIONS)
        with pytest.raises(ValueError):
            decode_action(9999)

    def test_get_action_info_no_action(self):
        m, c, v = get_action_info(0)
        assert m is None
        assert c is None
        assert v is None

    def test_get_action_info_regular(self):
        m, c, v = get_action_info(1)
        assert m is not None
        assert c is not None
        assert v is not None

    def test_all_actions_decodable(self):
        for i in range(NUM_ACTIONS):
            a = decode_action(i)
            assert a.action_id == i


# =========================================================================
# State Space
# =========================================================================
class TestStateSpace:
    def test_vector_shape(self, small_snapshots):
        vec = snapshot_to_vector(small_snapshots[0])
        assert vec.shape == (STATE_DIM,)

    def test_vector_dtype(self, small_snapshots):
        vec = snapshot_to_vector(small_snapshots[0])
        assert vec.dtype == np.float32

    def test_feature_names_match_dim(self):
        assert len(FEATURE_NAMES) == STATE_DIM

    def test_vector_no_nan(self, small_snapshots):
        for s in small_snapshots[:10]:
            vec = snapshot_to_vector(s)
            assert not np.any(np.isnan(vec)), "State vector contains NaN"

    def test_vector_no_inf(self, small_snapshots):
        for s in small_snapshots[:10]:
            vec = snapshot_to_vector(s)
            assert not np.any(np.isinf(vec)), "State vector contains Inf"

    def test_vector_bounded(self, small_snapshots):
        """Most features should be roughly normalized to [0, 1]."""
        for s in small_snapshots[:10]:
            vec = snapshot_to_vector(s)
            # Allow some leeway but nothing extreme
            assert np.all(vec >= -5.0), f"Min value: {vec.min()}"
            assert np.all(vec <= 5.0), f"Max value: {vec.max()}"

    def test_gap_flags_are_binary(self, small_snapshots):
        """Open gap flags should be 0.0 or 1.0."""
        gap_start_idx = 24  # After demographics(6)+clinical(6)+conditions(8)+meds(4)
        for s in small_snapshots[:10]:
            vec = snapshot_to_vector(s)
            for i in range(len(HEDIS_MEASURES)):
                val = vec[gap_start_idx + i]
                assert val in (0.0, 1.0), f"Gap flag at idx {gap_start_idx + i} = {val}"

    def test_action_history_encoding(self, small_snapshots):
        # action_history replaced by channel_affinity_counts/recency in Tier 3
        vec = snapshot_to_vector(
            small_snapshots[0],
            channel_affinity_counts={"sms": 3, "email": 1, "portal": 0, "app": 0, "ivr": 2},
            channel_affinity_recency={"sms": 5, "email": 10, "portal": 90, "app": 90, "ivr": 3},
        )
        assert vec.shape == (STATE_DIM,)

    def test_day_of_year_encoding(self, small_snapshots):
        vec1 = snapshot_to_vector(small_snapshots[0], day_of_year=1)
        vec2 = snapshot_to_vector(small_snapshots[0], day_of_year=180)
        # Temporal features should differ
        # demographics(6)+clinical(6)+conditions(8)+meds(4)+gaps(18)+engagement(11)+risk(4)+budget(10)=67
        temporal_idx = 67
        assert vec1[temporal_idx] != vec2[temporal_idx]


# =========================================================================
# Action Masking
# =========================================================================
class TestActionMasking:
    def test_no_action_always_valid(self):
        mask = compute_action_mask(
            open_gaps=set(), channel_availability={}, suppressed=True
        )
        assert mask[0] == True

    def test_suppressed_blocks_all(self):
        mask = compute_action_mask(
            open_gaps={"COL", "FLU"},
            channel_availability={"sms": True, "email": True, "ivr": True},
            suppressed=True,
        )
        assert mask.sum() == 1  # Only no_action

    def test_opt_out_blocks_all(self):
        mask = compute_action_mask(
            open_gaps={"COL"},
            channel_availability={"sms": True},
            opt_out=True,
        )
        assert mask.sum() == 1

    def test_grievance_hold_blocks_all(self):
        mask = compute_action_mask(
            open_gaps={"COL"},
            channel_availability={"sms": True},
            grievance_hold=True,
        )
        assert mask.sum() == 1

    def test_contact_limit_blocks_all(self):
        mask = compute_action_mask(
            open_gaps={"COL"},
            channel_availability={"sms": True, "email": True, "ivr": True},
            contacts_this_week=MAX_CONTACTS_PER_WEEK,
        )
        assert mask.sum() == 1

    def test_closed_gap_not_in_mask(self):
        mask = compute_action_mask(
            open_gaps={"COL"},  # Only COL is open
            channel_availability={"sms": True, "email": True, "portal": True, "app": True, "ivr": True},
        )
        # FLU actions should all be masked
        for action in ACTION_CATALOG:
            if action.measure == "FLU":
                assert mask[action.action_id] == False

    def test_open_gap_actions_available(self):
        mask = compute_action_mask(
            open_gaps={"COL"},
            channel_availability={"sms": True, "email": True, "portal": True, "app": True, "ivr": True},
        )
        # At least some COL actions should be available
        col_actions = [a for a in ACTION_CATALOG if a.measure == "COL"]
        assert any(mask[a.action_id] for a in col_actions)

    def test_channel_unavailable_blocks_channel(self):
        mask = compute_action_mask(
            open_gaps={"COL"},
            channel_availability={"sms": False, "email": True, "portal": False, "app": False, "ivr": False},
        )
        # SMS COL actions should be blocked
        for a in ACTION_CATALOG:
            if a.measure == "COL" and a.channel == "sms":
                assert mask[a.action_id] == False
        # Email COL actions should be available
        email_col = [a for a in ACTION_CATALOG if a.measure == "COL" and a.channel == "email"]
        assert any(mask[a.action_id] for a in email_col)

    def test_recent_measure_cooldown(self):
        mask = compute_action_mask(
            open_gaps={"COL", "FLU"},
            channel_availability={"sms": True, "email": True, "ivr": True},
            recent_measures={"COL": 2},  # Contacted 2 days ago, cooldown is 7
        )
        # COL actions should be blocked
        for a in ACTION_CATALOG:
            if a.measure == "COL":
                assert mask[a.action_id] == False
        # FLU actions should still be available
        flu_actions = [a for a in ACTION_CATALOG if a.measure == "FLU"]
        assert any(mask[a.action_id] for a in flu_actions)

    def test_mask_from_eligibility_snapshot(self, small_eligibility):
        for e in small_eligibility[:5]:
            mask = mask_from_eligibility_snapshot(e)
            assert mask.shape == (NUM_ACTIONS,)
            assert mask.dtype == bool
            assert mask[0] == True


# =========================================================================
# Reward Function
# =========================================================================
class TestReward:
    def test_no_action_zero_reward(self):
        r = compute_reward(measure=None, is_no_action=True)
        assert r == 0.0

    def test_gap_closure_positive_reward(self):
        r = compute_reward(measure="COL", gap_closed=True, delivered=True)
        assert r > 0.0

    def test_triple_weighted_higher_reward(self):
        r_single = compute_reward(measure="COL", gap_closed=True, delivered=True)
        r_triple = compute_reward(measure="MAC", gap_closed=True, delivered=True)
        assert r_triple > r_single

    def test_no_action_no_click_zero_reward(self):
        r = compute_reward(measure="COL", delivered=False)
        assert r == 0.0  # No gap closure, no click = zero reward

    def test_click_engagement_bonus(self):
        r_no_click = compute_reward(measure="COL", delivered=True)
        r_click = compute_reward(measure="COL", delivered=True, clicked=True)
        assert r_click > r_no_click

    def test_gap_closure_dominates_click(self):
        r_click_only = compute_reward(measure="COL", clicked=True)
        r_closure = compute_reward(measure="COL", gap_closed=True)
        assert r_closure > r_click_only

    def test_no_action_is_zero(self):
        r = compute_reward(measure=None, is_no_action=True)
        assert r == 0.0

    def test_stars_score_range(self):
        rates = {m: 0.5 for m in HEDIS_MEASURES}
        score = compute_stars_score(rates)
        assert 1.0 <= score <= 5.0

    def test_stars_score_monotonic(self):
        score_low = compute_stars_score({m: 0.2 for m in HEDIS_MEASURES})
        score_mid = compute_stars_score({m: 0.5 for m in HEDIS_MEASURES})
        score_high = compute_stars_score({m: 0.9 for m in HEDIS_MEASURES})
        assert score_low < score_mid < score_high

    def test_stars_bonus_threshold(self):
        # 85% across all measures should exceed 4.0 with CMS cut points
        score = compute_stars_score({m: 0.85 for m in HEDIS_MEASURES})
        assert score >= STARS_BONUS_THRESHOLD

    def test_stars_zero_rates(self):
        score = compute_stars_score({m: 0.0 for m in HEDIS_MEASURES})
        assert score >= 1.0

    def test_stars_empty_measures(self):
        score = compute_stars_score({})
        assert score == 1.0


# =========================================================================
# HEDISEnv Gym Environment
# =========================================================================
class TestHEDISEnv:
    @pytest.fixture
    def env(self, small_snapshots, small_eligibility):
        return HEDISEnv(small_snapshots, small_eligibility)

    def test_observation_space(self, env):
        assert "observations" in env.observation_space.spaces
        assert "action_mask" in env.observation_space.spaces
        assert env.observation_space["observations"].shape == (STATE_DIM,)
        assert env.observation_space["action_mask"].n == NUM_ACTIONS

    def test_action_space(self, env):
        assert env.action_space.n == NUM_ACTIONS

    def test_reset_returns_valid_obs(self, env):
        obs, info = env.reset(seed=42)
        assert obs["observations"].shape == (STATE_DIM,)
        assert obs["action_mask"].shape == (NUM_ACTIONS,)
        assert obs["action_mask"][0] == 1  # no_action always valid
        assert "patient_id" in info
        assert "open_gaps" in info

    def test_step_returns_correct_tuple(self, env):
        obs, info = env.reset(seed=42)
        valid_actions = np.where(obs["action_mask"])[0]
        action = valid_actions[0]
        obs2, reward, terminated, truncated, info2 = env.step(int(action))
        assert obs2["observations"].shape == (STATE_DIM,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "action_id" in info2

    def test_no_action_step(self, env):
        obs, info = env.reset(seed=42)
        obs2, reward, terminated, truncated, info2 = env.step(0)
        assert reward == 0.0
        assert info2["measure"] is None

    def test_invalid_action_still_works(self, env):
        """Environment should handle actions gracefully even if masked."""
        obs, info = env.reset(seed=42)
        # Step with no_action (always valid) — should not error
        obs2, reward, terminated, truncated, info2 = env.step(0)
        assert not terminated

    def test_episode_terminates_on_all_gaps_closed(self, env):
        """If all gaps close, episode should terminate."""
        obs, info = env.reset(seed=42)
        # Force close all gaps
        env._open_gaps = set()
        obs2, reward, terminated, truncated, info2 = env.step(0)
        assert terminated

    def test_episode_truncates_on_max_steps(self, env):
        obs, info = env.reset(seed=42)
        for _ in range(env.max_steps):
            obs, reward, terminated, truncated, info = env.step(0)
            if terminated or truncated:
                break
        assert truncated or terminated

    def test_step_info_has_required_fields(self, env):
        obs, info = env.reset(seed=42)
        valid_actions = np.where(obs["action_mask"])[0]
        _, _, _, _, info2 = env.step(int(valid_actions[0]))
        required = {"patient_id", "action_id", "measure", "channel", "variant",
                    "delivered", "opened", "clicked", "gap_closed",
                    "open_gaps", "episode_reward", "day_of_year"}
        assert required.issubset(info2.keys())

    def test_obs_no_nan_after_steps(self, env):
        obs, info = env.reset(seed=42)
        for _ in range(10):
            valid = np.where(obs["action_mask"])[0]
            action = int(np.random.choice(valid))
            obs, reward, terminated, truncated, info = env.step(action)
            assert not np.any(np.isnan(obs["observations"]))
            if terminated or truncated:
                break

    def test_multiple_episodes(self, env):
        """Run multiple reset/episode cycles."""
        for ep in range(5):
            obs, info = env.reset(seed=ep)
            total_reward = 0
            for step in range(10):
                valid = np.where(obs["action_mask"])[0]
                action = int(np.random.choice(valid))
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break

    def test_different_patients_across_resets(self, env):
        """Each reset should cycle to a different patient."""
        patient_ids = set()
        for _ in range(5):
            obs, info = env.reset()
            patient_ids.add(info["patient_id"])
        assert len(patient_ids) > 1
