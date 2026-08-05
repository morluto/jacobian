from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from benchmarks.validation._verifier_child import run_verifier_in_child
from benchmarks.validation.agent_workflow_v1 import support

ROOT = Path(__file__).parents[2]
DATASETS = ROOT / "benchmarks" / "datasets"
PROVIDER_TASKS = ("cddlib", "cgal", "gudhi", "lean-repl", "nauty", "regina")


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


@pytest.mark.parametrize("task_name", PROVIDER_TASKS)
def test_provider_verifier_images_include_bound_frozen_input(task_name: str) -> None:
    task = DATASETS / "provider-feasibility-v1" / task_name
    assert (task / "tests" / "input.json").read_bytes() == (
        task / "environment" / "input.json"
    ).read_bytes()
    assert (
        "COPY expected.json input.json " in (task / "tests" / "Dockerfile").read_text()
    )


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

    accepted = support._run_verifier(copied_task, app, logs)
    assert accepted["reward"] == pytest.approx(1.0)

    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["states"] = "8"
    _write_json(submission_path, submission)
    rejected = support._run_verifier(copied_task, app, logs)
    assert rejected["reward"] == 0.0


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

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == pytest.approx(1.0)
    assert accepted["reward"] == pytest.approx(1.0)


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
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0
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

    accepted = support._run_verifier(copied_task, app, logs)
    assert accepted["reward"] == pytest.approx(1.0)

    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["edge_orbits"] = [[[["a"], ["b"]], [["b"], ["c"]]]]
    _write_json(submission_path, submission)
    rejected = support._run_verifier(copied_task, app, logs)
    assert rejected["reward"] == 0.0


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
    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0


def _lean_case(tmp_path: Path, tasks: list[dict]):
    task = DATASETS / "provider-feasibility-v1" / "lean-repl"
    app = tmp_path / "lean-repl" / "app"
    logs = tmp_path / "lean-repl" / "logs"
    (app / "evidence").mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(task / "environment" / "input.json", app / "input.json")
    report = {
        "protocol": "leanprover-community/repl",
        "task_count": 2,
        "completed_count": 2,
        "parameter_error_count": 0,
        "return_code": 0,
        "tasks": tasks,
    }
    report_path = app / "evidence" / "provider-report.json"
    _write_json(report_path, report)
    expected = json.loads((task / "tests" / "expected.json").read_text())
    submission = {
        "task_id": expected["task_id"],
        "conclusion": "FEASIBLE",
        "result": {
            "provider": expected["provider"],
            "contract": expected["contract"],
            "status": "COMPLETED",
            "pin_sha256": expected["pin_sha256"],
        },
        "claimed_assurance": "COMPUTED",
        "scope": "pinned bounded provider reproduction",
        "completeness": "COMPLETE",
        "evidence": [
            {
                "path": "evidence/provider-report.json",
                "sha256": _digest(report_path),
            }
        ],
        "limitations": [],
    }
    _write_json(app / "submission.json", submission)
    return task, app, logs


def test_lean_repl_rejects_vacuous_empty_task_report(tmp_path: Path) -> None:
    task, app, logs = _lean_case(tmp_path, [])
    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0


def test_lean_repl_rejects_unhashable_task_id_without_crashing(
    tmp_path: Path,
) -> None:
    tasks = [
        {
            "task_id": ["CONJUNCTION-DECOMPOSITION"],
            "completed": True,
            "decomposition_observed": True,
            "tactics": [{"tactic": "constructor", "goal_count": 0, "error_count": 0}],
        },
        {
            "task_id": "LOCAL-PREMISE-APPLICATION",
            "completed": True,
            "decomposition_observed": True,
            "tactics": [{"tactic": "exact h hP", "goal_count": 0, "error_count": 0}],
        },
    ]
    task, app, logs = _lean_case(tmp_path, tasks)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0


def test_lean_repl_rejects_boolean_error_count(tmp_path: Path) -> None:
    tasks = [
        {
            "task_id": task_id,
            "completed": True,
            "decomposition_observed": True,
            "tactics": [{"tactic": tactic, "goal_count": 0, "error_count": False}],
        }
        for task_id, tactic in (
            ("CONJUNCTION-DECOMPOSITION", "constructor"),
            ("LOCAL-PREMISE-APPLICATION", "exact h hP"),
        )
    ]
    task, app, logs = _lean_case(tmp_path, tasks)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0


def test_lean_repl_derives_completion_from_final_goal_count(
    tmp_path: Path,
) -> None:
    tasks = [
        {
            "task_id": task_id,
            "completed": True,
            "decomposition_observed": True,
            "tactics": [{"tactic": tactic, "goal_count": 999, "error_count": 0}],
        }
        for task_id, tactic in (
            ("CONJUNCTION-DECOMPOSITION", "constructor"),
            ("LOCAL-PREMISE-APPLICATION", "exact h hP"),
        )
    ]
    task, app, logs = _lean_case(tmp_path, tasks)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0


def test_lean_repl_rejects_one_step_constructor_trace(tmp_path: Path) -> None:
    tasks = [
        {
            "task_id": "CONJUNCTION-DECOMPOSITION",
            "completed": True,
            "decomposition_observed": True,
            "tactics": [{"tactic": "constructor", "goal_count": 0, "error_count": 0}],
        },
        {
            "task_id": "LOCAL-PREMISE-APPLICATION",
            "completed": True,
            "decomposition_observed": True,
            "tactics": [{"tactic": "exact h hP", "goal_count": 0, "error_count": 0}],
        },
    ]
    task, app, logs = _lean_case(tmp_path, tasks)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0


def test_lean_repl_accepts_complete_distinct_task_traces(tmp_path: Path) -> None:
    tasks = [
        {
            "task_id": "CONJUNCTION-DECOMPOSITION",
            "completed": True,
            "decomposition_observed": True,
            "tactics": [
                {"tactic": "constructor", "goal_count": 2, "error_count": 0},
                {"tactic": "exact hP", "goal_count": 1, "error_count": 0},
                {"tactic": "exact hQ", "goal_count": 0, "error_count": 0},
            ],
        },
        {
            "task_id": "LOCAL-PREMISE-APPLICATION",
            "completed": True,
            "decomposition_observed": True,
            "tactics": [{"tactic": "exact h hP", "goal_count": 0, "error_count": 0}],
        },
    ]
    task, app, logs = _lean_case(tmp_path, tasks)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["reward"] == pytest.approx(1.0)
