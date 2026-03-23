"""
Shared test fixtures for the RL Agent Foundations test suite.
"""
import pytest
import json
import os
import shutil
import numpy as np
import torch

from config import (
    GENERATED_DATA_DIR, SIMULATION_DATA_DIR, NUM_ACTIONS, STATE_DIM,
    HEDIS_MEASURES, CHANNELS, ACTION_CATALOG, ACTION_BY_ID,
    MEASURE_WEIGHTS, CHECKPOINTS_DIR,
)
from datagen.patients import generate_patients
from datagen.state_features import generate_state_features
from datagen.historical_activity import generate_historical_activity
from datagen.gap_closure import generate_gap_closure
from datagen.action_eligibility import generate_action_eligibility


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def small_patients(rng):
    return generate_patients(50, rng=rng)


@pytest.fixture(scope="session")
def small_snapshots(small_patients, rng):
    return generate_state_features(small_patients, rng=rng)


@pytest.fixture(scope="session")
def small_historical(small_patients, small_snapshots, rng):
    return generate_historical_activity(small_patients, small_snapshots, n_records=500, rng=rng)


@pytest.fixture(scope="session")
def small_gap_closure(small_snapshots, rng):
    return generate_gap_closure(small_snapshots, rng=rng)


@pytest.fixture(scope="session")
def small_eligibility(small_snapshots, rng):
    return generate_action_eligibility(small_snapshots, rng=rng)


@pytest.fixture(scope="session")
def small_datasets(small_snapshots, small_historical, small_gap_closure, small_eligibility):
    return {
        "state_features": small_snapshots,
        "historical_activity": small_historical,
        "gap_closure": small_gap_closure,
        "action_eligibility": small_eligibility,
    }


@pytest.fixture
def test_data_dir(tmp_path):
    """Temporary data directory for tests that write files."""
    gen = tmp_path / "generated"
    sim = tmp_path / "simulation"
    gen.mkdir()
    sim.mkdir()
    return tmp_path
