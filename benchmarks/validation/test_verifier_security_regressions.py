from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from benchmarks.validation._verifier_child import run_verifier_in_child
from benchmarks.validation.public_reproductions_v1._verifier import _run_verifier

ROOT = Path(__file__).parents[2]
DATASETS = ROOT / "benchmarks" / "datasets"


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _solution_case(tmp_path: Path, dataset: str, task_name: str):
    task = DATASETS / dataset / task_name
    app = tmp_path / task_name / "app"
    logs = tmp_path / task_name / "logs"
    (app / "evidence").mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(task / "environment" / "input.json", app / "input.json")
    shutil.copy2(task / "solution" / "submission.json", app / "submission.json")
    shutil.copy2(task / "solution" / "answer.txt", app / "evidence" / "answer.txt")
    return task, app, logs


def test_reliability_recomputes_input_and_rejects_coerced_state_count(
    tmp_path: Path,
) -> None:
    task, app, logs = _solution_case(
        tmp_path, "public-reproductions-v1", "reliability-triangle-fair"
    )
    expected_path = task / "tests" / "expected.json"
    expected = json.loads(expected_path.read_text())
    expected["expected_probability"] = {"num": "0", "den": "1"}
    copied_task = tmp_path / "reliability-task"
    shutil.copytree(task, copied_task)
    _write_json(copied_task / "tests" / "expected.json", expected)

    accepted = _run_verifier(copied_task, app, logs)
    assert accepted.reward == pytest.approx(1.0)

    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["states"] = "8"
    _write_json(submission_path, submission)
    rejected = _run_verifier(copied_task, app, logs)
    assert rejected.reward == 0.0


def test_reliability_accepts_equivalent_unreduced_probability(
    tmp_path: Path,
) -> None:
    task, app, logs = _solution_case(
        tmp_path, "public-reproductions-v1", "reliability-triangle-fair"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["probability"] = {"num": "10", "den": "16"}
    _write_json(submission_path, submission)

    accepted = _run_verifier(task, app, logs)
    assert accepted.details["correctness"] == pytest.approx(1.0)
    assert accepted.reward == pytest.approx(1.0)


def test_reliability_rejects_oversized_fraction_without_crashing(
    tmp_path: Path,
) -> None:
    task, app, logs = _solution_case(
        tmp_path, "public-reproductions-v1", "reliability-triangle-fair"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["probability"] = {"num": "9" * 5000, "den": "1"}
    _write_json(submission_path, submission)

    rejected = run_verifier_in_child(task=task, app=app, logs=logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0
    assert (logs / "reward.json").is_file()


def test_symmetry_recomputes_orbits_and_rejects_nested_endpoint_bypass(
    tmp_path: Path,
) -> None:
    task, app, logs = _solution_case(
        tmp_path, "public-reproductions-v1", "symmetry-colored-reflection"
    )
    copied_task = tmp_path / "symmetry-task"
    shutil.copytree(task, copied_task)
    expected_path = copied_task / "tests" / "expected.json"
    expected = json.loads(expected_path.read_text())
    expected["expected_edge_orbits"] = []
    _write_json(expected_path, expected)

    accepted = _run_verifier(copied_task, app, logs)
    assert accepted.reward == pytest.approx(1.0)

    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["edge_orbits"] = [[[["a"], ["b"]], [["b"], ["c"]]]]
    _write_json(submission_path, submission)
    rejected = _run_verifier(copied_task, app, logs)
    assert rejected.reward == 0.0


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
    task, app, logs = _solution_case(tmp_path, "public-reproductions-v1", task_name)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"][field] = invalid
    _write_json(submission_path, submission)
    rejected = _run_verifier(task, app, logs)
    assert rejected.reward == 0.0
