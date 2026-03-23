"""
Tests for data generation modules.
Validates schema, consistency, clinical ranges, and cross-dataset integrity.
"""
import pytest
import numpy as np
import json

from config import (
    HEDIS_MEASURES, CHANNELS, ACTION_CATALOG, NUM_ACTIONS,
    ACTION_BY_ID, MEASURE_WEIGHTS, TRIPLE_WEIGHTED,
)
from datagen.constants import (
    AGE_RANGE, BP_SYSTOLIC, BP_DIASTOLIC, A1C, BMI, PHQ9,
    GAP_CLOSURE_BASE_RATES, DELIVERY_RATES, OPEN_RATES, CLICK_RATES,
)


# =========================================================================
# Patient Generation
# =========================================================================
class TestPatientGeneration:
    def test_correct_count(self, small_patients):
        assert len(small_patients) == 50

    def test_unique_ids(self, small_patients):
        ids = [p["patient_id"] for p in small_patients]
        assert len(set(ids)) == 50

    def test_id_format(self, small_patients):
        for p in small_patients:
            assert p["patient_id"].startswith("P")
            assert len(p["patient_id"]) == 6  # P + 5 digits

    def test_age_range(self, small_patients):
        for p in small_patients:
            assert AGE_RANGE[0] <= p["age"] <= AGE_RANGE[1]

    def test_sex_values(self, small_patients):
        for p in small_patients:
            assert p["sex"] in ("M", "F")

    def test_boolean_fields(self, small_patients):
        for p in small_patients:
            assert isinstance(p["dual_eligible"], bool)
            assert isinstance(p["lis_status"], bool)
            assert isinstance(p["snp_flag"], bool)

    def test_conditions_are_dict(self, small_patients):
        for p in small_patients:
            assert isinstance(p["conditions"], dict)
            for cond, val in p["conditions"].items():
                assert isinstance(val, bool), f"Condition {cond} should be bool, got {type(val)}"

    def test_conditions_have_expected_keys(self, small_patients):
        expected = {"diabetes", "hypertension", "hyperlipidemia", "depression",
                    "ckd", "chd", "copd", "chf"}
        for p in small_patients:
            assert set(p["conditions"].keys()) == expected

    def test_zip3_format(self, small_patients):
        for p in small_patients:
            assert len(p["zip3"]) == 3
            assert p["zip3"].isdigit()


# =========================================================================
# State Features
# =========================================================================
class TestStateFeatures:
    def test_correct_count(self, small_snapshots):
        assert len(small_snapshots) == 50

    def test_patient_id_matches(self, small_patients, small_snapshots):
        patient_ids = {p["patient_id"] for p in small_patients}
        snapshot_ids = {s["patient_id"] for s in small_snapshots}
        assert patient_ids == snapshot_ids

    def test_demographics_present(self, small_snapshots):
        for s in small_snapshots:
            d = s["demographics"]
            assert "age" in d
            assert "sex" in d
            assert "zip3" in d
            assert "dual_eligible" in d
            assert "lis_status" in d
            assert "snp_flag" in d

    def test_clinical_ranges(self, small_snapshots):
        for s in small_snapshots:
            c = s["clinical"]
            assert BP_SYSTOLIC["min"] <= c["bp_systolic_last"] <= BP_SYSTOLIC["max"]
            assert BP_DIASTOLIC["min"] <= c["bp_diastolic_last"] <= BP_DIASTOLIC["max"]
            assert A1C["min"] <= c["a1c_last"] <= A1C["max"]
            assert BMI["min"] <= c["bmi"] <= BMI["max"]
            assert PHQ9["min"] <= c["phq9_score"] <= PHQ9["max"]
            assert 0 <= c["ckd_stage"] <= 5

    def test_open_closed_gaps_disjoint(self, small_snapshots):
        for s in small_snapshots:
            open_set = set(s["open_gaps"])
            closed_set = set(s["closed_gaps"])
            assert open_set.isdisjoint(closed_set), \
                f"Patient {s['patient_id']} has overlapping gaps: {open_set & closed_set}"

    def test_gaps_are_valid_measures(self, small_snapshots):
        valid = set(HEDIS_MEASURES)
        for s in small_snapshots:
            for g in s["open_gaps"] + s["closed_gaps"]:
                assert g in valid, f"Invalid measure {g}"

    def test_medication_fill_rates_range(self, small_snapshots):
        for s in small_snapshots:
            for med, rate in s["medication_fill_rates"].items():
                assert 0.0 <= rate <= 1.0, f"Fill rate {med}={rate} out of range"

    def test_engagement_fields_present(self, small_snapshots):
        required_keys = {
            "sms_consent", "email_available", "portal_registered", "app_installed",
            "preferred_channel", "total_contacts_90d", "sms_response_rate",
            "email_open_rate", "days_since_last_contact",
        }
        for s in small_snapshots:
            assert required_keys.issubset(s["engagement"].keys())

    def test_engagement_rates_range(self, small_snapshots):
        for s in small_snapshots:
            e = s["engagement"]
            for key in ["sms_response_rate", "email_open_rate",
                        "portal_engagement_rate", "app_engagement_rate",
                        "ivr_completion_rate"]:
                assert 0.0 <= e[key] <= 1.0, f"{key}={e[key]} out of range"

    def test_risk_scores_range(self, small_snapshots):
        for s in small_snapshots:
            for key, val in s["risk_scores"].items():
                assert 0.0 <= val <= 5.0, f"Risk score {key}={val} out of range"

    def test_condition_specific_gap_eligibility(self, small_snapshots):
        """Patients without diabetes should not have diabetes-specific gaps."""
        for s in small_snapshots:
            conditions = s["clinical"]["conditions"]
            diabetes_measures = {"EED", "HBD", "KED", "MDS", "BPD"}
            if not conditions.get("diabetes", False):
                for m in diabetes_measures:
                    assert m not in s["open_gaps"] and m not in s["closed_gaps"], \
                        f"Non-diabetic {s['patient_id']} has diabetes measure {m}"

    def test_bcs_only_female(self, small_snapshots):
        for s in small_snapshots:
            if s["demographics"]["sex"] == "M":
                assert "BCS" not in s["open_gaps"] and "BCS" not in s["closed_gaps"]


# =========================================================================
# Historical Activity
# =========================================================================
class TestHistoricalActivity:
    def test_record_count(self, small_historical):
        assert len(small_historical) == 500

    def test_records_sorted_by_date(self, small_historical):
        dates = [r["date"] for r in small_historical]
        assert dates == sorted(dates)

    def test_required_fields_present(self, small_historical):
        required = {"record_id", "patient_id", "date", "action_id", "measure",
                    "channel", "variant", "outcome", "context"}
        for r in small_historical:
            assert required.issubset(r.keys()), f"Missing keys: {required - r.keys()}"

    def test_action_ids_valid(self, small_historical):
        for r in small_historical:
            assert 0 <= r["action_id"] < NUM_ACTIONS, \
                f"Invalid action_id {r['action_id']}"

    def test_action_id_matches_measure_channel(self, small_historical):
        for r in small_historical:
            action = ACTION_BY_ID[r["action_id"]]
            assert action.measure == r["measure"]
            assert action.channel == r["channel"]

    def test_outcome_fields(self, small_historical):
        for r in small_historical:
            o = r["outcome"]
            assert isinstance(o["delivered"], bool)
            assert isinstance(o["opened"], bool)
            assert isinstance(o["clicked"], bool)
            assert isinstance(o["gap_closed_within_30d"], bool)
            assert isinstance(o["gap_closed_within_90d"], bool)

    def test_outcome_logical_ordering(self, small_historical):
        """Can't click without opening, can't open without delivering."""
        for r in small_historical:
            o = r["outcome"]
            if o["clicked"]:
                assert o["opened"], "Clicked but not opened"
            if o["opened"]:
                assert o["delivered"], "Opened but not delivered"

    def test_gap_closure_consistency(self, small_historical):
        """30d closure implies 90d closure."""
        for r in small_historical:
            o = r["outcome"]
            if o["gap_closed_within_30d"]:
                assert o["gap_closed_within_90d"], "Closed in 30d but not 90d"

    def test_days_to_closure_consistency(self, small_historical):
        for r in small_historical:
            o = r["outcome"]
            if o["gap_closed_within_30d"]:
                assert o["days_to_closure"] is not None
                assert 1 <= o["days_to_closure"] <= 30
            elif o["gap_closed_within_90d"]:
                assert o["days_to_closure"] is not None
                assert 31 <= o["days_to_closure"] <= 90
            else:
                assert o["days_to_closure"] is None

    def test_measures_are_valid(self, small_historical):
        valid = set(HEDIS_MEASURES)
        for r in small_historical:
            assert r["measure"] in valid

    def test_channels_are_valid(self, small_historical):
        valid = set(CHANNELS)
        for r in small_historical:
            assert r["channel"] in valid

    def test_context_fields(self, small_historical):
        for r in small_historical:
            c = r["context"]
            assert c["prior_attempts_this_measure"] >= 0
            assert c["days_since_last_contact"] >= 1
            assert c["member_tenure_months"] >= 6


# =========================================================================
# Gap Closure
# =========================================================================
class TestGapClosure:
    def test_has_records(self, small_gap_closure):
        assert len(small_gap_closure) > 0

    def test_required_fields(self, small_gap_closure):
        required = {"patient_id", "measure_year", "measure", "timeline",
                    "final_status", "closure_date"}
        for r in small_gap_closure:
            assert required.issubset(r.keys())

    def test_timeline_has_entries(self, small_gap_closure):
        for r in small_gap_closure:
            assert len(r["timeline"]) > 0

    def test_timeline_entries_schema(self, small_gap_closure):
        for r in small_gap_closure:
            for entry in r["timeline"]:
                assert "date" in entry
                assert "gap_open" in entry
                assert isinstance(entry["gap_open"], bool)

    def test_final_status_matches_timeline(self, small_gap_closure):
        for r in small_gap_closure:
            last_entry = r["timeline"][-1]
            if r["final_status"] == "closed":
                assert not last_entry["gap_open"]
            elif r["final_status"] == "open":
                assert last_entry["gap_open"]

    def test_closure_date_present_when_closed(self, small_gap_closure):
        for r in small_gap_closure:
            if r["final_status"] == "closed":
                assert r["closure_date"] is not None

    def test_measures_valid(self, small_gap_closure):
        valid = set(HEDIS_MEASURES)
        for r in small_gap_closure:
            assert r["measure"] in valid

    def test_measure_year(self, small_gap_closure):
        for r in small_gap_closure:
            assert r["measure_year"] == 2026


# =========================================================================
# Action Eligibility
# =========================================================================
class TestActionEligibility:
    def test_correct_count(self, small_eligibility):
        assert len(small_eligibility) == 50

    def test_action_mask_length(self, small_eligibility):
        for e in small_eligibility:
            assert len(e["action_mask"]) == NUM_ACTIONS

    def test_no_action_always_available(self, small_eligibility):
        for e in small_eligibility:
            assert e["action_mask"][0] is True, \
                f"no_action should always be available for {e['patient_id']}"

    def test_global_constraints_present(self, small_eligibility):
        for e in small_eligibility:
            gc = e["global_constraints"]
            assert "opt_out" in gc
            assert "grievance_hold" in gc
            assert "suppression_active" in gc
            assert "max_contacts_per_week" in gc
            assert "contacts_this_week" in gc

    def test_opt_out_blocks_all_actions(self, small_eligibility):
        for e in small_eligibility:
            if e["global_constraints"]["opt_out"]:
                assert sum(e["action_mask"]) == 1, \
                    "Opt-out patient should only have no_action"

    def test_grievance_hold_blocks_all_actions(self, small_eligibility):
        for e in small_eligibility:
            if e["global_constraints"]["grievance_hold"]:
                assert sum(e["action_mask"]) == 1

    def test_contact_limit_blocks_all_actions(self, small_eligibility):
        for e in small_eligibility:
            gc = e["global_constraints"]
            if gc["contacts_this_week"] >= gc["max_contacts_per_week"]:
                if not gc["opt_out"] and not gc["grievance_hold"] and not gc["suppression_active"]:
                    assert sum(e["action_mask"]) == 1

    def test_sms_blocked_without_consent(self, small_snapshots, small_eligibility):
        snap_map = {s["patient_id"]: s for s in small_snapshots}
        for e in small_eligibility:
            snap = snap_map[e["patient_id"]]
            if not snap["engagement"]["sms_consent"] and not any(
                e["global_constraints"][k] for k in ["opt_out", "grievance_hold", "suppression_active"]
            ):
                # All SMS actions should be blocked
                for action in ACTION_CATALOG[1:]:
                    if action.channel == "sms" and e["action_mask"][action.action_id]:
                        pytest.fail(
                            f"SMS action {action.action_id} enabled for {e['patient_id']} without SMS consent"
                        )

    def test_app_blocked_without_install(self, small_snapshots, small_eligibility):
        snap_map = {s["patient_id"]: s for s in small_snapshots}
        for e in small_eligibility:
            snap = snap_map[e["patient_id"]]
            if not snap["engagement"]["app_installed"] and not any(
                e["global_constraints"][k] for k in ["opt_out", "grievance_hold", "suppression_active"]
            ):
                for action in ACTION_CATALOG[1:]:
                    if action.channel == "app" and e["action_mask"][action.action_id]:
                        pytest.fail(
                            f"App action {action.action_id} enabled for {e['patient_id']} without app installed"
                        )

    def test_closed_gap_actions_blocked(self, small_snapshots, small_eligibility):
        """Actions for already-closed gaps should be masked out."""
        snap_map = {s["patient_id"]: s for s in small_snapshots}
        for e in small_eligibility:
            snap = snap_map[e["patient_id"]]
            open_gaps = set(snap.get("open_gaps", []))
            if any(e["global_constraints"][k] for k in ["opt_out", "grievance_hold", "suppression_active"]):
                continue
            for action in ACTION_CATALOG[1:]:
                if action.measure not in open_gaps and e["action_mask"][action.action_id]:
                    pytest.fail(
                        f"Action {action.action_id} ({action.measure}) enabled for "
                        f"{e['patient_id']} but gap is not open"
                    )


# =========================================================================
# Cross-Dataset Consistency
# =========================================================================
class TestCrossDatasetConsistency:
    def test_patient_ids_consistent(self, small_snapshots, small_eligibility):
        snap_ids = {s["patient_id"] for s in small_snapshots}
        elig_ids = {e["patient_id"] for e in small_eligibility}
        assert snap_ids == elig_ids

    def test_historical_references_valid_patients(self, small_patients, small_historical):
        patient_ids = {p["patient_id"] for p in small_patients}
        for r in small_historical:
            assert r["patient_id"] in patient_ids

    def test_gap_closure_references_valid_patients(self, small_snapshots, small_gap_closure):
        patient_ids = {s["patient_id"] for s in small_snapshots}
        for r in small_gap_closure:
            assert r["patient_id"] in patient_ids

    def test_gap_closure_measures_match_patient_gaps(self, small_snapshots, small_gap_closure):
        """Each gap closure record's measure should be in the patient's eligible gaps."""
        snap_map = {s["patient_id"]: s for s in small_snapshots}
        for r in small_gap_closure:
            snap = snap_map[r["patient_id"]]
            all_gaps = set(snap["open_gaps"]) | set(snap["closed_gaps"])
            assert r["measure"] in all_gaps, \
                f"Gap closure for {r['measure']} but patient {r['patient_id']} " \
                f"doesn't have this gap (has {all_gaps})"


# =========================================================================
# Action Catalog Integrity
# =========================================================================
class TestActionCatalog:
    def test_action_ids_sequential(self):
        for i, action in enumerate(ACTION_CATALOG):
            assert action.action_id == i

    def test_no_action_is_index_zero(self):
        assert ACTION_CATALOG[0].measure == "NO_ACTION"
        assert ACTION_CATALOG[0].channel == "none"

    def test_all_measures_represented(self):
        measures_in_catalog = {a.measure for a in ACTION_CATALOG if a.measure != "NO_ACTION"}
        assert measures_in_catalog == set(HEDIS_MEASURES)

    def test_all_channels_represented(self):
        channels_in_catalog = {a.channel for a in ACTION_CATALOG if a.channel != "none"}
        assert channels_in_catalog == set(CHANNELS)

    def test_triple_weighted_measures_correct(self):
        assert TRIPLE_WEIGHTED == {"MAC", "MRA", "MDS", "DMC02", "TRC_M"}
        for m in TRIPLE_WEIGHTED:
            assert MEASURE_WEIGHTS[m] == 3.0

    def test_single_weighted_measures(self):
        for m in HEDIS_MEASURES:
            if m not in TRIPLE_WEIGHTED:
                assert MEASURE_WEIGHTS[m] == 1.0

    def test_action_descriptions_non_empty(self):
        for a in ACTION_CATALOG:
            assert len(a.description) > 0

    def test_action_by_id_complete(self):
        assert len(ACTION_BY_ID) == NUM_ACTIONS
        for i in range(NUM_ACTIONS):
            assert i in ACTION_BY_ID


# =========================================================================
# Data Serialization / Output Format Validation
# =========================================================================
class TestDataSerialization:
    def test_state_features_json_serializable(self, small_snapshots):
        """State features must be fully JSON-serializable for file storage."""
        serialized = json.dumps(small_snapshots, default=str)
        deserialized = json.loads(serialized)
        assert len(deserialized) == len(small_snapshots)

    def test_historical_activity_json_serializable(self, small_historical):
        serialized = json.dumps(small_historical, default=str)
        deserialized = json.loads(serialized)
        assert len(deserialized) == len(small_historical)

    def test_gap_closure_json_serializable(self, small_gap_closure):
        serialized = json.dumps(small_gap_closure, default=str)
        deserialized = json.loads(serialized)
        assert len(deserialized) == len(small_gap_closure)

    def test_eligibility_json_serializable(self, small_eligibility):
        serialized = json.dumps(small_eligibility, default=str)
        deserialized = json.loads(serialized)
        assert len(deserialized) == len(small_eligibility)

    def test_roundtrip_preserves_types(self, small_snapshots):
        """Verify JSON roundtrip preserves critical types."""
        serialized = json.dumps(small_snapshots[0], default=str)
        d = json.loads(serialized)
        assert isinstance(d["demographics"]["age"], int)
        assert isinstance(d["clinical"]["bp_systolic_last"], float)
        assert isinstance(d["open_gaps"], list)
        assert isinstance(d["engagement"]["sms_consent"], bool)
