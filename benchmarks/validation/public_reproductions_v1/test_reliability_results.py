"""Reliability-task result and input-binding regressions."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from benchmarks.validation._verifier_child import run_verifier_in_child
from benchmarks.validation.public_reproductions_v1._fixtures import (
    _prepare_case,
    _write_json,
)
from benchmarks.validation.public_reproductions_v1._verifier import _run_verifier


def test_reliability_recomputes_input_and_rejects_coerced_state_count(
    tmp_path: Path,
) -> None:
    task, app, logs = _prepare_case(
        tmp_path, "reliability-triangle-fair", "input-binding"
    )
    copied_task = tmp_path / "reliability-task"
    shutil.copytree(task, copied_task)
    expected_path = copied_task / "tests" / "expected.json"
    expected = json.loads(expected_path.read_text())
    expected["expected_probability"] = {"num": 0, "den": 1}
    _write_json(expected_path, expected)

    assert _run_verifier(copied_task, app, logs).reward == pytest.approx(1.0)

    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["states"] = "8"
    _write_json(submission_path, submission)

    assert _run_verifier(copied_task, app, logs).reward == 0.0


@pytest.mark.parametrize(
    ("task_name", "probability"),
    (
        ("reliability-series-path", {"num": 2, "den": 6}),
        ("reliability-single-edge", {"num": 2, "den": 6}),
        ("reliability-triangle-fair", {"num": 10, "den": 16}),
    ),
)
def test_reliability_accepts_equivalent_unreduced_probability(
    tmp_path: Path, task_name: str, probability: dict[str, int]
) -> None:
    task, app, logs = _prepare_case(tmp_path, task_name, "equivalent-rational")
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["probability"] = probability
    _write_json(submission_path, submission)

    accepted = _run_verifier(task, app, logs)
    assert accepted.details["correctness"] == pytest.approx(1.0)
    assert accepted.reward == pytest.approx(1.0)


def test_reliability_rejects_oversized_fraction_without_crashing(
    tmp_path: Path,
) -> None:
    task, app, logs = _prepare_case(
        tmp_path, "reliability-triangle-fair", "oversized-fraction"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["probability"] = {"num": 10**4000, "den": 1}
    _write_json(submission_path, submission)

    rejected = run_verifier_in_child(task=task, app=app, logs=logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0
    assert (logs / "reward.json").is_file()


@pytest.mark.parametrize(
    ("task_name", "path", "invalid"),
    (
        ("reliability-series-path", ("probability", "num"), "1"),
        ("reliability-single-edge", ("probability", "num"), "1"),
        ("reliability-triangle-fair", ("probability", "num"), "5"),
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
