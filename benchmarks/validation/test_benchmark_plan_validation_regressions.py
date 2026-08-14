"""Regressions for bounded benchmark-plan validation."""

from __future__ import annotations

from typing import Any

import pytest
from tools.benchmark_plan.validation import (
    MAX_MATRIX_JOBS,
    BenchmarkPlanValidationError,
    validate_plan,
)

DIGEST = "sha256:" + "a" * 64


def _plan() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "event": "pull_request",
        "base_sha": "0" * 40,
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


def test_oracle_task_cannot_appear_in_multiple_shards() -> None:
    plan = _plan()
    first = plan["oracle_matrix"][0]
    plan["oracle_matrix"] = [first, {**first, "shard": "other"}]

    with pytest.raises(BenchmarkPlanValidationError, match="suite/task"):
        validate_plan(plan)


def test_host_job_names_are_globally_unique() -> None:
    plan = _plan()
    first = plan["host_matrix"][0]
    plan["host_matrix"] = [
        first,
        {**first, "selector": "benchmarks/validation/test_other.py"},
    ]

    with pytest.raises(BenchmarkPlanValidationError, match="matrix name"):
        validate_plan(plan)


@pytest.mark.parametrize("field", ["name", "selector", "keyword"])
def test_host_text_fields_reject_non_strings_without_tracebacks(field: str) -> None:
    plan = _plan()
    plan["host_matrix"][0][field] = None

    with pytest.raises(BenchmarkPlanValidationError, match=f"invalid {field}"):
        validate_plan(plan)


def test_host_split_count_is_bounded_before_group_expansion() -> None:
    plan = _plan()
    plan["host_matrix"][0].update({"splits": MAX_MATRIX_JOBS + 1, "group": 1})

    with pytest.raises(BenchmarkPlanValidationError, match="invalid splits"):
        validate_plan(plan)
