"""Unit tests for resource-dominance execution profiles."""

from __future__ import annotations

from tools.test_plan.execution_profiles import (
    compile_execution_profile,
    profile_for_lane,
    validate_lane_against_profile,
)


def test_process_resource_dominates_semantic_defaults() -> None:
    profile = compile_execution_profile(
        semantic_owner="composition",
        resources={"complete-runtime", "process-group"},
        default_workers=4,
    )
    assert profile.name == "process-isolated"
    assert profile.process_supervision is True
    assert profile.workers == 2


def test_sqlite_lane_matches_execution_profile() -> None:
    errors = validate_lane_against_profile(
        name="storage",
        required_environment=("sqlite",),
        workers=0,
        distribution="none",
        timeout_seconds=120,
    )
    assert errors == []
    profile = profile_for_lane(
        name="storage",
        required_environment=("sqlite",),
        workers=0,
        distribution="none",
        timeout_seconds=120,
    )
    assert profile.name == "sqlite-serial"
    assert profile.sqlite_serial is True
    assert profile.setup_affinity == "sqlite"


def test_worker_mismatch_fails_profile_validation() -> None:
    errors = validate_lane_against_profile(
        name="storage",
        required_environment=("sqlite",),
        workers=4,
        distribution="worksteal",
        timeout_seconds=120,
    )
    assert errors
