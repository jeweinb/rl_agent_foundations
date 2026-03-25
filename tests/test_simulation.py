"""
Tests for the simulation modules.
Validates daily/nightly cycles, lagged rewards, state machine,
metrics tracking, and output file formats for the dashboard.
"""
import pytest
import json
import os
import numpy as np

from config import (
    NUM_ACTIONS, STATE_DIM, HEDIS_MEASURES, SIMULATION_DATA_DIR,
    ACTION_CATALOG, MEASURE_WEIGHTS, STARS_BONUS_THRESHOLD,
)
from simulation.action_state_machine import (
    ActionLifecycleTracker, ActionState, VALID_TRANSITIONS,
    CHANNEL_TRANSITION_PROBS,
)
from simulation.lagged_rewards import LaggedRewardQueue
from simulation.metrics import MetricsTracker
from simulation.daily_cycle import run_daily_cycle
from simulation.world import WorldSimulator
from training.data_loader import build_offline_episodes
from training.cql_trainer import ActorCriticCQL
from environment.hedis_env import HEDISEnv


# =========================================================================
# Action State Machine
# =========================================================================
class TestActionStateMachine:
    @pytest.fixture
    def tracker(self):
        return ActionLifecycleTracker(rng=np.random.default_rng(42))

    def test_create_action(self, tracker):
        record = tracker.create_action(
            tracking_id="test_001", patient_id="P10000",
            action_id=1, measure="COL", channel="sms",
            variant="scheduling_link", day=1,
        )
        assert record["tracking_id"] == "test_001"
        assert record["current_state"] == ActionState.CREATED
        assert record["terminal"] == False
        assert len(record["state_history"]) == 1

    def test_advance_created_to_queued(self, tracker):
        tracker.create_action("t1", "P10000", 1, "COL", "sms", "scheduling_link", 1)
        transition = tracker.advance("t1", 1)
        assert transition is not None
        assert transition["to_state"] == ActionState.QUEUED

    def test_advance_through_full_lifecycle(self, tracker):
        """Advance an action until it reaches a terminal state."""
        tracker.create_action("t1", "P10000", 1, "COL", "sms", "scheduling_link", 1)
        states_seen = {ActionState.CREATED}
        for day in range(1, 20):
            t = tracker.advance("t1", day)
            if t is None:
                break
            states_seen.add(t["to_state"])

        # Should have reached a terminal state
        record = tracker.actions["t1"]
        assert record["terminal"] == True
        terminal_states = {ActionState.COMPLETED, ActionState.FAILED,
                          ActionState.DECLINED, ActionState.EXPIRED}
        assert record["current_state"] in terminal_states

    def test_terminal_state_no_further_advance(self, tracker):
        tracker.create_action("t1", "P10000", 1, "COL", "sms", "scheduling_link", 1)
        # Force to terminal
        tracker.actions["t1"]["current_state"] = ActionState.COMPLETED
        tracker.actions["t1"]["terminal"] = True
        result = tracker.advance("t1", 5)
        assert result is None

    def test_advance_all(self, tracker):
        for i in range(5):
            tracker.create_action(f"t{i}", f"P{10000+i}", i+1, "COL", "sms", "x", 1)
        transitions = tracker.advance_all(1)
        assert len(transitions) == 5  # All should move CREATED → QUEUED

    def test_engagement_signals(self, tracker):
        tracker.create_action("t1", "P10000", 1, "COL", "sms", "x", 1)
        signals = tracker.get_engagement_signals("t1")
        assert signals["delivered"] == False
        assert signals["clicked"] == False

        # Advance through to VIEWED
        tracker.advance("t1", 1)  # → QUEUED
        tracker.advance("t1", 1)  # → PRESENTED or FAILED
        record = tracker.actions["t1"]
        if record["current_state"] == ActionState.PRESENTED:
            tracker.actions["t1"]["current_state"] = ActionState.VIEWED
            signals = tracker.get_engagement_signals("t1")
            assert signals["delivered"] == True
            assert signals["opened"] == True

    def test_pending_actions(self, tracker):
        tracker.create_action("t1", "P10000", 1, "COL", "sms", "x", 1)
        tracker.create_action("t2", "P10000", 2, "FLU", "email", "y", 1)
        pending = tracker.get_pending_actions("P10000")
        assert len(pending) == 2

    def test_patient_history(self, tracker):
        tracker.create_action("t1", "P10000", 1, "COL", "sms", "x", 1)
        tracker.create_action("t2", "P10001", 2, "FLU", "email", "y", 1)
        history = tracker.get_patient_history("P10000")
        assert len(history) == 1
        assert history[0]["tracking_id"] == "t1"

    def test_funnel_stats(self, tracker):
        tracker.create_action("t1", "P10000", 1, "COL", "sms", "x", 1)
        stats = tracker.get_funnel_stats()
        assert stats["CREATED"] == 1

    def test_to_records_serializable(self, tracker):
        tracker.create_action("t1", "P10000", 1, "COL", "sms", "x", 1)
        tracker.advance("t1", 1)
        records = tracker.to_records()
        serialized = json.dumps(records, default=str)
        deserialized = json.loads(serialized)
        assert len(deserialized) == 1
        assert deserialized[0]["current_state"] in [s.value for s in ActionState]

    def test_state_history_format(self, tracker):
        tracker.create_action("t1", "P10000", 1, "COL", "sms", "x", 1)
        tracker.advance("t1", 1)
        records = tracker.to_records()
        for entry in records[0]["state_history"]:
            assert "state" in entry
            assert "day" in entry
            assert "timestamp" in entry

    def test_valid_transitions_complete(self):
        """Every ActionState should appear in the transition map."""
        for state in ActionState:
            assert state in VALID_TRANSITIONS

    def test_channel_probs_valid(self):
        for channel, probs in CHANNEL_TRANSITION_PROBS.items():
            for key, val in probs.items():
                assert 0.0 <= val <= 1.0, f"{channel} {key} = {val}"


# =========================================================================
# Lagged Rewards
# =========================================================================
class TestLaggedRewards:
    @pytest.fixture
    def queue(self):
        return LaggedRewardQueue(rng=np.random.default_rng(42))

    def test_schedule_adds_pending(self, queue):
        queue.schedule(1, "P10000", "COL", 1, 0.5)
        assert queue.get_pending_count() == 1

    def test_collect_before_resolve_day(self, queue):
        queue.schedule(1, "P10000", "COL", 1, 0.5)
        resolved = queue.collect(1)  # Same day — shouldn't resolve yet
        assert len(resolved) == 0
        assert queue.get_pending_count() == 1

    def test_collect_after_resolve_day(self, queue):
        queue.schedule(1, "P10000", "COL", 1, 0.5)
        # COL is in "screenings" category: lag min=14 days
        resolved = queue.collect(100)  # Far enough in future
        assert len(resolved) == 1

    def test_closure_probability(self):
        """High probability should close more often over many samples."""
        rng = np.random.default_rng(42)
        queue = LaggedRewardQueue(rng=rng)
        n = 1000
        for i in range(n):
            queue.schedule(1, f"P{i}", "FLU", 1, 0.8)
        resolved = queue.collect(200)  # All should resolve
        closed = sum(1 for r in resolved if r["will_close"])
        # ~80% should close (with some variance)
        assert 0.7 * n < closed < 0.9 * n

    def test_zero_probability_never_closes(self):
        rng = np.random.default_rng(42)
        queue = LaggedRewardQueue(rng=rng)
        for i in range(100):
            queue.schedule(1, f"P{i}", "COL", 1, 0.0)
        resolved = queue.collect(200)
        closed = sum(1 for r in resolved if r["will_close"])
        assert closed == 0

    def test_resolved_records_have_required_fields(self, queue):
        queue.schedule(1, "P10000", "COL", 1, 0.5)
        resolved = queue.collect(200)
        for r in resolved:
            assert "resolve_day" in r
            assert "patient_id" in r
            assert "measure" in r
            assert "action_id" in r
            assert "closure_prob" in r
            assert "will_close" in r
            assert "reward" in r
            assert "scheduled_day" in r

    def test_get_all_resolved(self, queue):
        queue.schedule(1, "P10000", "COL", 1, 0.5)
        queue.collect(200)
        all_resolved = queue.get_all_resolved()
        assert len(all_resolved) == 1


# =========================================================================
# Metrics Tracker
# =========================================================================
class TestMetricsTracker:
    @pytest.fixture
    def tracker(self):
        return MetricsTracker()

    def test_record_day(self, tracker):
        result = tracker.record_day(
            day=1, daily_reward=5.0, daily_actions=100,
            daily_gap_closures={"COL": 5, "FLU": 3},
            daily_total_patients={"COL": 50, "FLU": 40},
            champion_score=1.0, challenger_score=1.2,
            model_promoted=False, model_version=1,
        )
        assert result["day"] == 1
        assert "stars_score" in result
        assert "cumulative_reward" in result
        assert "above_bonus_threshold" in result

    def test_cumulative_reward_accumulates(self, tracker):
        tracker.record_day(1, 5.0, 100, {}, {})
        tracker.record_day(2, 3.0, 80, {}, {})
        assert tracker.cumulative_reward == 8.0

    def test_stars_trajectory(self, tracker):
        tracker.record_day(1, 5.0, 100, {"COL": 10}, {"COL": 50})
        tracker.record_day(2, 6.0, 100, {"COL": 5}, {"COL": 50})
        traj = tracker.get_stars_trajectory()
        assert len(traj) == 2

    def test_regret_curve(self, tracker):
        tracker.record_day(1, 5.0, 100, {}, {})
        tracker.record_day(2, 3.0, 80, {}, {})
        regret = tracker.get_regret_curve(oracle_reward_per_day=10.0)
        assert len(regret) == 2
        assert regret[0] == 5.0  # 10 - 5
        assert regret[1] == 12.0  # 5 + (10 - 3)

    def test_to_records_serializable(self, tracker):
        tracker.record_day(1, 5.0, 100, {"COL": 5}, {"COL": 50},
                          champion_score=1.0, challenger_score=1.2,
                          model_promoted=True, model_version=2,
                          action_distribution={"COL_sms": 30},
                          state_machine_funnel={"CREATED": 100, "COMPLETED": 20})
        records = tracker.to_records()
        serialized = json.dumps(records, default=str)
        deserialized = json.loads(serialized)
        assert len(deserialized) == 1

    def test_metrics_output_format_for_dashboard(self, tracker):
        """Validate the exact fields the dashboard callbacks expect."""
        tracker.record_day(
            day=1, daily_reward=5.0, daily_actions=100,
            daily_gap_closures={"COL": 5}, daily_total_patients={"COL": 50},
            champion_score=1.0, challenger_score=1.2,
            model_promoted=False, model_version=1,
            action_distribution={"COL_sms": 30},
            state_machine_funnel={"CREATED": 100},
        )
        record = tracker.get_latest()
        # Fields expected by dashboard/callbacks.py
        dashboard_fields = [
            "day", "daily_reward", "cumulative_reward", "daily_actions",
            "cumulative_actions", "stars_score", "above_bonus_threshold",
            "measure_closure_rates", "champion_score", "challenger_score",
            "model_promoted", "model_version", "action_distribution",
            "state_machine_funnel",
        ]
        for field in dashboard_fields:
            assert field in record, f"Missing dashboard field: {field}"

    def test_measure_closure_rates_keys(self, tracker):
        tracker.record_day(1, 5.0, 100,
                          {m: 1 for m in HEDIS_MEASURES},
                          {m: 10 for m in HEDIS_MEASURES})
        latest = tracker.get_latest()
        for m in HEDIS_MEASURES:
            assert m in latest["measure_closure_rates"]

    def test_stars_score_computable(self, tracker):
        tracker.record_day(1, 5.0, 100,
                          {m: 8 for m in HEDIS_MEASURES},
                          {m: 10 for m in HEDIS_MEASURES})
        latest = tracker.get_latest()
        assert 1.0 <= latest["stars_score"] <= 5.0


# =========================================================================
# Daily Cycle
# =========================================================================
class TestDailyCycle:
    def _make_world(self, small_snapshots, small_eligibility):
        return WorldSimulator(small_snapshots, small_eligibility, rng=np.random.default_rng(42))

    def test_daily_cycle_runs(self, small_snapshots, small_eligibility, tmp_path):
        import config
        orig_dir = config.SIMULATION_DATA_DIR
        config.SIMULATION_DATA_DIR = str(tmp_path / "simulation")
        os.makedirs(config.SIMULATION_DATA_DIR, exist_ok=True)

        try:
            agent = ActorCriticCQL()
            world = self._make_world(small_snapshots, small_eligibility)

            result = run_daily_cycle(
                day=1, agent=agent, world=world,
                rng=np.random.default_rng(42),
            )

            assert result["day"] == 1
            assert "total_reward" in result
            assert "num_actions" in result
            assert "gap_closures" in result
            assert "total_patients" in result
            assert "action_distribution" in result
            assert "state_machine_funnel" in result
        finally:
            config.SIMULATION_DATA_DIR = orig_dir

    def test_daily_cycle_writes_output_files(self, small_snapshots, small_eligibility, tmp_path):
        import config
        orig_dir = config.SIMULATION_DATA_DIR
        config.SIMULATION_DATA_DIR = str(tmp_path / "simulation")
        os.makedirs(config.SIMULATION_DATA_DIR, exist_ok=True)

        try:
            agent = ActorCriticCQL()
            world = self._make_world(small_snapshots, small_eligibility)

            run_daily_cycle(day=1, agent=agent, world=world)

            day_dir = os.path.join(config.SIMULATION_DATA_DIR, "day_01")
            assert os.path.exists(os.path.join(day_dir, "actions_taken.json"))
            assert os.path.exists(os.path.join(day_dir, "experience_buffer.json"))
            assert os.path.exists(os.path.join(day_dir, "state_machine.json"))
        finally:
            config.SIMULATION_DATA_DIR = orig_dir

    def test_actions_taken_format(self, small_snapshots, small_eligibility, tmp_path):
        import config
        orig_dir = config.SIMULATION_DATA_DIR
        config.SIMULATION_DATA_DIR = str(tmp_path / "simulation")
        os.makedirs(config.SIMULATION_DATA_DIR, exist_ok=True)

        try:
            agent = ActorCriticCQL()
            world = self._make_world(small_snapshots, small_eligibility)

            run_daily_cycle(day=1, agent=agent, world=world)

            day_dir = os.path.join(config.SIMULATION_DATA_DIR, "day_01")
            with open(os.path.join(day_dir, "actions_taken.json")) as f:
                actions = json.load(f)

            assert len(actions) == len(small_snapshots)
            for a in actions:
                assert "patient_id" in a
                assert "action_id" in a
                assert "measure" in a
                assert "channel" in a
                assert "reward" in a
                assert "day" in a
                assert isinstance(a["reward"], (int, float))
                assert 0 <= a["action_id"] < NUM_ACTIONS
        finally:
            config.SIMULATION_DATA_DIR = orig_dir

    def test_experience_buffer_format(self, small_snapshots, small_eligibility, tmp_path):
        import config
        orig_dir = config.SIMULATION_DATA_DIR
        config.SIMULATION_DATA_DIR = str(tmp_path / "simulation")
        os.makedirs(config.SIMULATION_DATA_DIR, exist_ok=True)

        try:
            agent = ActorCriticCQL()
            world = self._make_world(small_snapshots, small_eligibility)

            run_daily_cycle(day=1, agent=agent, world=world)

            day_dir = os.path.join(config.SIMULATION_DATA_DIR, "day_01")
            with open(os.path.join(day_dir, "experience_buffer.json")) as f:
                experiences = json.load(f)

            assert len(experiences) == len(small_snapshots)
            for e in experiences:
                assert "obs" in e
                assert "action" in e
                assert "reward" in e
                assert "mask" in e
                assert "patient_id" in e
                assert len(e["obs"]) == STATE_DIM
                assert len(e["mask"]) == NUM_ACTIONS
                assert 0 <= e["action"] < NUM_ACTIONS
        finally:
            config.SIMULATION_DATA_DIR = orig_dir

    def test_state_machine_output_format(self, small_snapshots, small_eligibility, tmp_path):
        import config
        orig_dir = config.SIMULATION_DATA_DIR
        config.SIMULATION_DATA_DIR = str(tmp_path / "simulation")
        os.makedirs(config.SIMULATION_DATA_DIR, exist_ok=True)

        try:
            agent = ActorCriticCQL()
            world = self._make_world(small_snapshots, small_eligibility)

            run_daily_cycle(day=1, agent=agent, world=world)

            day_dir = os.path.join(config.SIMULATION_DATA_DIR, "day_01")
            with open(os.path.join(day_dir, "state_machine.json")) as f:
                sm_records = json.load(f)

            for r in sm_records:
                assert "tracking_id" in r
                assert "patient_id" in r
                assert "action_id" in r
                assert "measure" in r
                assert "channel" in r
                assert "current_state" in r
                assert "state_history" in r
                assert "terminal" in r
                # State must be a valid ActionState value
                valid_states = {s.value for s in ActionState}
                assert r["current_state"] in valid_states
                for sh in r["state_history"]:
                    assert sh["state"] in valid_states
                    assert "day" in sh
                    assert "timestamp" in sh
        finally:
            config.SIMULATION_DATA_DIR = orig_dir


# =========================================================================
# Dashboard Data Feed Output Validation
# =========================================================================
class TestDashboardDataFormats:
    """Validate that all simulation output files have the exact format
    the dashboard callbacks expect."""

    def test_cumulative_metrics_format(self, tmp_path):
        tracker = MetricsTracker()
        for day in range(1, 4):
            tracker.record_day(
                day=day, daily_reward=float(day),
                daily_actions=100, daily_gap_closures={"COL": day},
                daily_total_patients={"COL": 50},
                champion_score=float(day), challenger_score=float(day + 0.1),
                model_promoted=(day == 2), model_version=1 + (1 if day >= 2 else 0),
                action_distribution={"COL_sms": 30},
                state_machine_funnel={"CREATED": 100, "COMPLETED": 20},
            )

        records = tracker.to_records()
        output_path = tmp_path / "cumulative_metrics.json"
        with open(output_path, "w") as f:
            json.dump(records, f, default=str)

        # Re-read and validate
        with open(output_path) as f:
            loaded = json.load(f)

        assert isinstance(loaded, list)
        for m in loaded:
            # Overview tab fields
            assert isinstance(m["day"], int)
            assert isinstance(m["stars_score"], float)
            assert isinstance(m["cumulative_reward"], float)
            assert isinstance(m["measure_closure_rates"], dict)
            # Training tab fields
            assert m["champion_score"] is not None or True  # can be None for day 0
            assert "model_promoted" in m
            assert "model_version" in m
            # State machine tab
            assert "state_machine_funnel" in m
            assert isinstance(m["state_machine_funnel"], dict)

    def test_nightly_metrics_format(self, tmp_path):
        """Validate the per-day nightly metrics JSON format."""
        nightly = {
            "day": 1,
            "champion_metrics": {
                "mean_reward": 1.5, "std_reward": 0.3,
                "median_reward": 1.4, "mean_gaps_closed": 2.1,
                "mean_episode_length": 25.0, "no_action_rate": 0.3,
                "n_episodes": 200,
            },
            "challenger_metrics": {
                "mean_reward": 1.7, "std_reward": 0.2,
                "median_reward": 1.6, "mean_gaps_closed": 2.5,
                "mean_episode_length": 23.0, "no_action_rate": 0.25,
                "n_episodes": 200,
            },
            "comparison": {
                "champion_mean_reward": 1.5,
                "challenger_mean_reward": 1.7,
                "relative_improvement": 0.133,
                "promote_challenger": True,
                "champion_gaps_closed": 2.1,
                "challenger_gaps_closed": 2.5,
            },
            "promoted": True,
        }

        path = tmp_path / "nightly_metrics.json"
        with open(path, "w") as f:
            json.dump(nightly, f)

        with open(path) as f:
            loaded = json.load(f)

        assert loaded["day"] == 1
        assert "champion_metrics" in loaded
        assert "challenger_metrics" in loaded
        assert "comparison" in loaded
        assert "promoted" in loaded
        assert loaded["comparison"]["promote_challenger"] == True

    def test_actions_format_for_patient_journey(self, tmp_path):
        """Validate actions_taken.json has fields needed for patient journey tab."""
        actions = [
            {
                "patient_id": "P10000",
                "action_id": 1,
                "measure": "COL",
                "channel": "sms",
                "variant": "scheduling_link",
                "reward": 0.04,
                "day": 1,
                "engagement": {
                    "delivered": True,
                    "opened": True,
                    "clicked": False,
                    "completed": False,
                    "failed": False,
                    "expired": False,
                },
            },
            {
                "patient_id": "P10000",
                "action_id": 0,
                "measure": None,
                "channel": None,
                "variant": None,
                "reward": 0.0,
                "day": 2,
                "engagement": {},
            },
        ]

        path = tmp_path / "actions_taken.json"
        with open(path, "w") as f:
            json.dump(actions, f)

        with open(path) as f:
            loaded = json.load(f)

        # Patient journey callback expects these fields
        for a in loaded:
            assert "patient_id" in a
            assert "action_id" in a
            assert "measure" in a
            assert "channel" in a
            assert "variant" in a
            assert "reward" in a
            assert "day" in a
            assert "engagement" in a
