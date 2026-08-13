"""Regressions for bounded benchmark-plan validation."""

from __future__ import annotations

import json

import pytest
from tools.benchmark_plan.validation import (
    MAX_MATRIX_JOBS,
    BenchmarkPlanValidationError,
    validate_plan,
)

DIGEST = "sha256:" + "a" * 64


def _plan() -> dict[str, str]:
    return {
        "benchmark-plan-version": "2",
        "benchmark-plan-event": "pull_request",
        "benchmark-plan-base-sha": "0" * 40,
        "benchmark-plan-head-sha": "1" * 40,
        "benchmark-planner-digest": DIGEST,
        "benchmark-topology-digest": DIGEST,
        "benchmark-plan-mode": "changed",
        "run-benchmark-check": "true",
        "run-benchmark-record-schema": "true",
        "run-benchmark-prospective-digest": "true",
        "run-benchmark-inventory": "false",
        "run-benchmark-host-validation": "true",
        "benchmark-host-validation-matrix": json.dumps(
            [
                {
                    "name": "task",
                    "selector": "benchmarks/validation/test_task.py",
                    "keyword": "",
                    "splits": 0,
                    "group": 0,
                    "predicted_seconds": 5.0,
                }
            ]
        ),
        "run-benchmark-oracle": "true",
        "benchmark-oracle-scope": "changed-tasks",
        "benchmark-oracle-matrix": json.dumps(
            [
                {
                    "dataset": "suite",
                    "shard": "task",
                    "tasks": ["task"],
                    "task_digests": [{"task": "task", "digest": DIGEST}],
                    "predicted_seconds": 30.0,
                }
            ]
        ),
        "benchmark-plan-reasons": json.dumps(["executable task change"]),
    }


def test_oracle_task_cannot_appear_in_multiple_shards() -> None:
    plan = _plan()
    first = json.loads(plan["benchmark-oracle-matrix"])[0]
    second = {**first, "shard": "other"}
    plan["benchmark-oracle-matrix"] = json.dumps([first, second])

    with pytest.raises(BenchmarkPlanValidationError, match="suite/task"):
        validate_plan(plan)


def test_host_job_names_are_globally_unique() -> None:
    plan = _plan()
    first = json.loads(plan["benchmark-host-validation-matrix"])[0]
    second = {
        **first,
        "selector": "benchmarks/validation/test_other.py",
    }
    plan["benchmark-host-validation-matrix"] = json.dumps([first, second])

    with pytest.raises(BenchmarkPlanValidationError, match="matrix name"):
        validate_plan(plan)


@pytest.mark.parametrize("field", ["name", "selector", "keyword"])
def test_host_text_fields_reject_non_strings_without_tracebacks(field: str) -> None:
    plan = _plan()
    matrix = json.loads(plan["benchmark-host-validation-matrix"])
    matrix[0][field] = None
    plan["benchmark-host-validation-matrix"] = json.dumps(matrix)

    with pytest.raises(BenchmarkPlanValidationError, match=f"invalid {field}"):
        validate_plan(plan)


def test_host_split_count_is_bounded_before_group_expansion() -> None:
    plan = _plan()
    matrix = json.loads(plan["benchmark-host-validation-matrix"])
    matrix[0].update({"splits": MAX_MATRIX_JOBS + 1, "group": 1})
    plan["benchmark-host-validation-matrix"] = json.dumps(matrix)

    with pytest.raises(BenchmarkPlanValidationError, match="invalid splits"):
        validate_plan(plan)
