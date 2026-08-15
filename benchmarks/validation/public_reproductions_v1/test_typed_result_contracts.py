"""Strict result-type regressions shared by public reproductions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.public_reproductions_v1._fixtures import (
    _prepare_case,
    _write_json,
)
from benchmarks.validation.public_reproductions_v1._verifier import _run_verifier


@pytest.mark.parametrize(
    ("task_name", "field", "invalid"),
    (
        ("smith-rank-deficient", "rank", "1"),
        ("lean-transition", "goal_count", "2"),
    ),
)
def test_public_reproductions_reject_schema_invalid_integer_coercion(
    tmp_path: Path, task_name: str, field: str, invalid: str
) -> None:
    task, app, logs = _prepare_case(tmp_path, task_name, "integer-coercion")
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"][field] = invalid
    _write_json(submission_path, submission)

    assert _run_verifier(task, app, logs).reward == 0.0


@pytest.mark.parametrize(
    ("task_name", "path", "invalid"),
    (
        ("smith-rank-deficient", ("invariant_factors", 0), "2"),
        ("smith-rectangular", ("invariant_factors", 1), "6"),
    ),
    ids=(
        "smith-rank-deficient-path3-2",
        "smith-rectangular-path4-6",
    ),
)
def test_exact_numeric_results_reject_string_coercion(
    tmp_path: Path,
    task_name: str,
    path: tuple[str | int, ...],
    invalid: object,
) -> None:
    task, app, logs = _prepare_case(tmp_path, task_name, "numeric-coercion")
    assert _run_verifier(task, app, logs).reward == pytest.approx(1.0)

    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    parent = submission["result"]
    for part in path[:-1]:
        parent = parent[part]
    parent[path[-1]] = invalid
    _write_json(submission_path, submission)

    rejected = _run_verifier(task, app, logs)
    assert rejected.details["correctness"] == pytest.approx(0.0)
    assert rejected.reward == pytest.approx(0.0)
