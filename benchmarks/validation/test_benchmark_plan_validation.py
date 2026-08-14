"""Fail-closed tests for the canonical benchmark plan object."""

from __future__ import annotations

from typing import Any

import pytest
from tools.benchmark_plan.validation import (
    BenchmarkPlanValidationError,
    validate_plan,
)

SHA = "0" * 40
DIGEST = "sha256:" + "a" * 64


def _plan() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "event": "pull_request",
        "base_sha": SHA,
        "head_sha": "1" * 40,
        "changed_paths_digest": DIGEST,
        "planner_digest": DIGEST,
        "topology_digest": DIGEST,
        "mode": "changed",
        "run_check": True,
        "record_schema": True,
        "prospective_digest": True,
        "inventory": False,
        "host_matrix": [
            {
                "name": "task",
                "selector": "benchmarks/validation/test_task.py",
                "keyword": "",
                "splits": 0,
                "group": 0,
                "predicted_seconds": 5.0,
            }
        ],
        "oracle_scope": "changed-tasks",
        "oracle_matrix": [
            {
                "dataset": "suite",
                "shard": "task",
                "tasks": ["task"],
                "task_digests": [{"task": "task", "digest": DIGEST}],
                "predicted_seconds": 30.0,
            }
        ],
        "reasons": ["executable task change"],
    }


def _skipped_plan() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "event": "pull_request",
        "base_sha": "",
        "head_sha": "",
        "changed_paths_digest": "",
        "planner_digest": DIGEST,
        "topology_digest": "",
        "mode": "none",
        "run_check": False,
        "record_schema": False,
        "prospective_digest": False,
        "inventory": False,
        "host_matrix": [],
        "oracle_scope": "none",
        "oracle_matrix": [],
        "reasons": [],
    }


def _reject(plan: dict[str, Any]) -> str:
    with pytest.raises(BenchmarkPlanValidationError) as caught:
        validate_plan(plan)
    return str(caught.value)


def test_valid_benchmark_plan_is_accepted() -> None:
    validate_plan(_plan())


def test_skipped_plan_is_accepted_with_empty_topology_and_no_lanes() -> None:
    validate_plan(_skipped_plan())


def test_unknown_plan_version_is_rejected() -> None:
    plan = _plan()
    plan["schema_version"] = 3

    assert "version" in _reject(plan)


def test_invalid_event_is_rejected() -> None:
    plan = _plan()
    plan["event"] = "release"

    assert "event" in _reject(plan)


def test_malformed_base_sha_is_rejected() -> None:
    plan = _plan()
    plan["base_sha"] = "not-a-sha"

    assert "base-sha" in _reject(plan)


def test_planner_digest_must_be_a_sha256() -> None:
    plan = _plan()
    plan["planner_digest"] = "sha256:short"

    assert "planner-digest" in _reject(plan)


def test_skipped_plan_must_not_carry_a_topology_digest() -> None:
    plan = _skipped_plan()
    plan["topology_digest"] = DIGEST

    assert "topology-digest" in _reject(plan)


def test_running_checks_require_a_topology_digest() -> None:
    plan = _plan()
    plan["topology_digest"] = ""

    assert "topology-digest" in _reject(plan)


def test_inventory_lane_requires_integration_or_full_mode() -> None:
    plan = _plan()
    plan["inventory"] = True
    plan["mode"] = "changed"

    assert "inventory" in _reject(plan)


def test_inventory_lane_is_accepted_in_integration_mode() -> None:
    plan = _plan()
    plan["inventory"] = True
    plan["mode"] = "integration"
    plan["event"] = "merge_group"

    validate_plan(plan)


def test_prospective_digest_lane_requires_checks() -> None:
    plan = _skipped_plan()
    plan["prospective_digest"] = True

    assert "prospective-digest" in _reject(plan)


def test_record_schema_lane_must_run_when_checks_run() -> None:
    plan = _plan()
    plan["record_schema"] = False
    plan["prospective_digest"] = False

    assert "record-schema" in _reject(plan)


def test_oracle_plan_requires_a_nonempty_matrix() -> None:
    plan = _plan()
    plan["oracle_matrix"] = []

    assert "disabled" in _reject(plan).lower()


def test_disabled_oracle_must_have_none_scope_and_empty_matrix() -> None:
    plan = _plan()
    plan["oracle_scope"] = "none"

    assert "scope" in _reject(plan)


def test_host_validation_selector_cannot_escape_validation_tree() -> None:
    plan = _plan()
    plan["host_matrix"][0]["selector"] = "benchmarks/validation/../datasets"

    assert "invalid selector" in _reject(plan)


def test_host_validation_requires_every_declared_shard() -> None:
    plan = _plan()
    plan["host_matrix"][0].update({"splits": 2, "group": 1})

    assert "incomplete" in _reject(plan)


def test_duplicate_dataset_task_pair_is_rejected() -> None:
    plan = _plan()
    entry = plan["oracle_matrix"][0]
    plan["oracle_matrix"] = [entry, entry]

    assert "duplicate" in _reject(plan)


def test_missing_plan_key_is_rejected() -> None:
    plan = _plan()
    del plan["planner_digest"]

    assert "keys differ" in _reject(plan)


def test_extra_plan_key_is_rejected() -> None:
    plan = _plan()
    plan["unexpected_key"] = True

    assert "keys differ" in _reject(plan)


def test_running_checks_require_reasons() -> None:
    plan = _plan()
    plan["reasons"] = []

    assert "reasons" in _reject(plan)
