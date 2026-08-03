"""Fail-closed tests for the benchmark workflow plan contract."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
VALIDATOR = ROOT / ".github" / "scripts" / "validate-benchmark-plan"

SHA = "0" * 40
DIGEST = "sha256:" + "a" * 64


def _run(plan: dict[str, str]) -> subprocess.CompletedProcess[str]:
    payload = "\n".join(f"{key}={value}" for key, value in plan.items()) + "\n"
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        input=payload,
        text=True,
        capture_output=True,
        check=False,
    )


def _plan() -> dict[str, str]:
    return {
        "benchmark-plan-version": "1",
        "benchmark-plan-event": "pull_request",
        "benchmark-plan-base-sha": SHA,
        "benchmark-plan-head-sha": "1" * 40,
        "benchmark-planner-digest": DIGEST,
        "benchmark-topology-digest": DIGEST,
        "benchmark-plan-mode": "changed",
        "run-benchmark-check": "true",
        "run-benchmark-record-schema": "true",
        "run-benchmark-prospective-digest": "true",
        "run-benchmark-inventory": "false",
        "run-benchmark-oracle": "true",
        "benchmark-oracle-scope": "changed-tasks",
        "benchmark-oracle-matrix": json.dumps(
            [
                {
                    "dataset": "suite",
                    "shard": "task",
                    "tasks": ["task"],
                    "task_digests": [{"task": "task", "digest": DIGEST}],
                }
            ]
        ),
        "benchmark-plan-reasons": json.dumps(["executable task change"]),
    }


def test_valid_benchmark_plan_is_accepted() -> None:
    result = _run(_plan())

    assert result.returncode == 0, result.stderr


def test_skipped_plan_is_accepted_with_empty_topology_and_no_lanes() -> None:
    plan = {
        "benchmark-plan-version": "1",
        "benchmark-plan-event": "pull_request",
        "benchmark-plan-base-sha": "",
        "benchmark-plan-head-sha": "",
        "benchmark-planner-digest": DIGEST,
        "benchmark-topology-digest": "",
        "benchmark-plan-mode": "none",
        "run-benchmark-check": "false",
        "run-benchmark-record-schema": "false",
        "run-benchmark-prospective-digest": "false",
        "run-benchmark-inventory": "false",
        "run-benchmark-oracle": "false",
        "benchmark-oracle-scope": "none",
        "benchmark-oracle-matrix": "[]",
        "benchmark-plan-reasons": "[]",
    }

    result = _run(plan)

    assert result.returncode == 0, result.stderr


def test_unknown_plan_version_is_rejected() -> None:
    plan = _plan()
    plan["benchmark-plan-version"] = "2"

    result = _run(plan)

    assert result.returncode != 0
    assert "version" in result.stderr


def test_invalid_event_is_rejected() -> None:
    plan = _plan()
    plan["benchmark-plan-event"] = "release"

    result = _run(plan)

    assert result.returncode != 0
    assert "event" in result.stderr


def test_malformed_base_sha_is_rejected() -> None:
    plan = _plan()
    plan["benchmark-plan-base-sha"] = "not-a-sha"

    result = _run(plan)

    assert result.returncode != 0
    assert "base-sha" in result.stderr


def test_planner_digest_must_be_a_sha256() -> None:
    plan = _plan()
    plan["benchmark-planner-digest"] = "sha256:short"

    result = _run(plan)

    assert result.returncode != 0
    assert "planner-digest" in result.stderr


def test_skipped_plan_must_not_carry_a_topology_digest() -> None:
    plan = _plan()
    plan["run-benchmark-check"] = "false"
    plan["run-benchmark-record-schema"] = "false"
    plan["run-benchmark-prospective-digest"] = "false"
    plan["run-benchmark-inventory"] = "false"
    plan["run-benchmark-oracle"] = "false"
    plan["benchmark-oracle-scope"] = "none"
    plan["benchmark-oracle-matrix"] = "[]"
    plan["benchmark-plan-reasons"] = "[]"
    plan["benchmark-plan-mode"] = "none"
    plan["benchmark-topology-digest"] = DIGEST

    result = _run(plan)

    assert result.returncode != 0
    assert "topology-digest" in result.stderr


def test_running_checks_require_a_topology_digest() -> None:
    plan = _plan()
    plan["benchmark-topology-digest"] = ""

    result = _run(plan)

    assert result.returncode != 0
    assert "topology-digest" in result.stderr


def test_inventory_lane_requires_integration_or_full_mode() -> None:
    plan = _plan()
    plan["run-benchmark-inventory"] = "true"
    plan["benchmark-plan-mode"] = "changed"

    result = _run(plan)

    assert result.returncode != 0
    assert "inventory" in result.stderr


def test_inventory_lane_is_accepted_in_integration_mode() -> None:
    plan = _plan()
    plan["run-benchmark-inventory"] = "true"
    plan["benchmark-plan-mode"] = "integration"
    plan["benchmark-plan-event"] = "merge_group"

    result = _run(plan)

    assert result.returncode == 0, result.stderr


def test_prospective_digest_lane_requires_checks() -> None:
    plan = _plan()
    plan["run-benchmark-check"] = "false"
    plan["run-benchmark-record-schema"] = "false"
    plan["run-benchmark-oracle"] = "false"
    plan["benchmark-oracle-scope"] = "none"
    plan["benchmark-oracle-matrix"] = "[]"
    plan["benchmark-plan-reasons"] = "[]"
    plan["benchmark-plan-mode"] = "none"
    plan["benchmark-topology-digest"] = ""
    # leave prospective-digest true while checks are false
    plan["run-benchmark-prospective-digest"] = "true"

    result = _run(plan)

    assert result.returncode != 0
    assert "prospective-digest" in result.stderr


def test_record_schema_lane_must_run_when_checks_run() -> None:
    plan = _plan()
    plan["run-benchmark-record-schema"] = "false"
    plan["run-benchmark-prospective-digest"] = "false"

    result = _run(plan)

    assert result.returncode != 0
    assert "record-schema" in result.stderr


def test_prospective_digest_lane_requires_record_schema() -> None:
    plan = _plan()
    plan["run-benchmark-record-schema"] = "false"

    result = _run(plan)

    assert result.returncode != 0
    assert "record/schema" in result.stderr


def test_oracle_plan_requires_a_nonempty_matrix() -> None:
    plan = _plan()
    plan["benchmark-oracle-matrix"] = "[]"

    result = _run(plan)

    assert result.returncode != 0
    assert "non-empty matrix" in result.stderr


def test_disabled_oracle_must_have_none_scope_and_empty_matrix() -> None:
    plan = _plan()
    plan["run-benchmark-oracle"] = "false"
    plan["benchmark-oracle-scope"] = "changed-tasks"

    result = _run(plan)

    assert result.returncode != 0
    assert "disabled" in result.stderr.lower()


def test_duplicate_dataset_task_pair_is_rejected() -> None:
    plan = _plan()
    entry = json.loads(plan["benchmark-oracle-matrix"])[0]
    plan["benchmark-oracle-matrix"] = json.dumps([entry, entry])

    result = _run(plan)

    assert result.returncode != 0
    assert "duplicate" in result.stderr


def test_missing_plan_key_is_rejected() -> None:
    plan = _plan()
    del plan["benchmark-planner-digest"]

    result = _run(plan)

    assert result.returncode != 0
    assert "keys differ" in result.stderr


def test_extra_plan_key_is_rejected() -> None:
    plan = _plan()
    plan["benchmark-unexpected-key"] = "true"

    result = _run(plan)

    assert result.returncode != 0
    assert "keys differ" in result.stderr


def test_running_checks_require_reasons() -> None:
    plan = _plan()
    plan["benchmark-plan-reasons"] = "[]"

    result = _run(plan)

    assert result.returncode != 0
    assert "reasons" in result.stderr
