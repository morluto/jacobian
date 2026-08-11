"""Generic assurance and protocol diagnostic contract tests.

Cross-task invariants for assurance ceilings, scope-assurance independence,
unhashable assurance handling, and task-local contract metadata validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import _metadata, support


def test_rational_solution_rejects_unsupported_verified_claim(
    tmp_path: Path,
) -> None:
    computed = support._run_verifier(
        *support._prepare_case(tmp_path, support.RATIONAL_TASK, "computed")
    )
    assert computed.details["correctness"] == 1.0
    assert computed.reward == pytest.approx(1.0)
    assert computed.details["false_certification"] is False

    for scenario, expected_correctness, false_certification in (
        ("missing", 1.0, True),
        ("invalid", 0.0, False),
    ):
        result = support._run_verifier(
            *support._prepare_case(tmp_path, support.RATIONAL_TASK, scenario)
        )
        assert result.details["correctness"] == expected_correctness
        assert result.reward == 0.0
        assert result.details["false_certification"] is false_certification


@pytest.mark.parametrize("task_name", support.RESOURCE_DERIVED_TASKS)
def test_resource_derived_oracles_and_assurance_boundary(
    tmp_path: Path,
    task_name: str,
) -> None:
    computed = support._run_verifier(
        *support._prepare_case(tmp_path, task_name, "computed")
    )
    assert computed.details["correctness"] == 1.0
    assert computed.reward == pytest.approx(1.0)
    assert computed.details["false_certification"] is False

    unsupported = support._run_verifier(
        *support._prepare_case(tmp_path, task_name, "missing")
    )
    assert unsupported.details["correctness"] == 1.0
    assert unsupported.reward == 0.0
    assert unsupported.details["false_certification"] is True


@pytest.mark.parametrize("task_name", support.VERIFIER_TASKS)
def test_verifiers_reject_unhashable_assurance(
    tmp_path: Path,
    task_name: str,
) -> None:
    task, app, logs = support._prepare_case(tmp_path, task_name, "computed")
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["claimed_assurance"] = []
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    expected_scope = 1.0 if support.is_scope_independent_assurance(task_name) else 0.0
    assert rejected.details["scope_accuracy"] == expected_scope
    assert rejected.reward == 0.0
    assert rejected.details["false_certification"] is False


@pytest.mark.parametrize(
    "task_name",
    [
        name
        for name in support.VERIFIER_TASKS
        if support.is_scope_independent_assurance(name)
    ],
)
def test_scope_independent_verifiers_preserve_scope_for_unsupported_assurance(
    tmp_path: Path,
    task_name: str,
) -> None:
    task, app, logs = support._prepare_case(tmp_path, task_name, "computed")
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["claimed_assurance"] = "CHECKED"
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["scope_accuracy"] == 1.0
    assert rejected.details["assurance_calibration"] == 0.0
    assert rejected.reward == 0.0


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"schema_version": "1", "input_binding_decoupled": "yes"}, "boolean"),
        ({"schema_version": "2"}, "schema_version"),
        ({"schema_version": "1", "unknown": True}, "unknown"),
    ],
)
def test_task_contract_metadata_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
    message: str,
) -> None:
    tests = tmp_path / "sample" / "tests"
    tests.mkdir(parents=True)
    support._write_json(tests / "verifier_contract.json", payload)
    monkeypatch.setattr(_metadata, "TASKS", tmp_path)

    with pytest.raises(ValueError, match=message):
        support.load_task_contract_metadata("sample")
