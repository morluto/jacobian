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


def test_sqlite_serialization_dominates_process_scheduling() -> None:
    profile = compile_execution_profile(
        semantic_owner="storage",
        resources={"sqlite", "process-group"},
        default_workers=4,
    )

    assert profile.name == "sqlite-serial"
    assert profile.workers == 0
    assert profile.distribution == "none"
    assert profile.sqlite_serial is True
    assert profile.process_supervision is True


def test_lean_dominates_all_other_resources() -> None:
    profile = compile_execution_profile(
        semantic_owner="formal",
        resources={"lean", "sqlite", "process-group", "mcp"},
    )

    assert profile.name == "lean-serial"
    assert profile.timeout_seconds == 300
    assert profile.process_supervision is True
    assert profile.sqlite_serial is False


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


def test_lane_validation_uses_sqlite_over_process_dominance() -> None:
    errors = validate_lane_against_profile(
        name="storage-process",
        required_environment=("sqlite", "process-group"),
        workers=0,
        distribution="none",
        timeout_seconds=120,
    )

    assert errors == []
