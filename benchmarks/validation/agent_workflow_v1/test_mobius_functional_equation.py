from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.agent_workflow_v1 import support

TASK = "mobius-functional-equation"


def _prepare_mobius_case(tmp_path: Path):
    task, app, logs = support._prepare_case(
        tmp_path, "mobius-functional-equation", "computed"
    )
    (app / "evidence" / "functional-equation-certificate.json").write_bytes(
        (task / "solution" / "functional-equation-certificate.json").read_bytes()
    )
    return task, app, logs


def test_mobius_functional_equation_accepts_exact_orbit(tmp_path: Path) -> None:
    task, app, logs = _prepare_mobius_case(tmp_path)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_mobius_functional_equation_rejects_corrupted_orbit_value(
    tmp_path: Path,
) -> None:
    task, app, logs = _prepare_mobius_case(tmp_path)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["solution_values"][1]["numerator"][0] += 1
    evidence = {
        "schema_version": "1",
        "task_id": submission["task_id"],
        "result": submission["result"],
        "limitations": submission["limitations"],
    }
    evidence_path = app / "evidence" / "functional-equation-certificate.json"
    support._write_json(evidence_path, evidence)
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_mobius_functional_equation_rejects_singular_matrix_claim(
    tmp_path: Path,
) -> None:
    task, app, logs = _prepare_mobius_case(tmp_path)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["coefficient_matrix"][2] = [0, 1, 1]
    evidence = {
        "schema_version": "1",
        "task_id": submission["task_id"],
        "result": submission["result"],
        "limitations": submission["limitations"],
    }
    evidence_path = app / "evidence" / "functional-equation-certificate.json"
    support._write_json(evidence_path, evidence)
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_mobius_functional_equation_rejects_scalar_orbit(tmp_path: Path) -> None:
    """A scalar orbit must be rejected without crashing the verifier."""
    task, app, logs = _prepare_mobius_case(tmp_path)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["orbit"] = 0
    evidence = {
        "schema_version": "1",
        "task_id": submission["task_id"],
        "result": submission["result"],
        "limitations": submission["limitations"],
    }
    evidence_path = app / "evidence" / "functional-equation-certificate.json"
    support._write_json(evidence_path, evidence)
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_mobius_functional_equation_rejects_short_orbit(tmp_path: Path) -> None:
    """A short orbit list must be rejected without crashing the verifier."""
    task, app, logs = _prepare_mobius_case(tmp_path)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["orbit"] = submission["result"]["orbit"][:1]
    evidence = {
        "schema_version": "1",
        "task_id": submission["task_id"],
        "result": submission["result"],
        "limitations": submission["limitations"],
    }
    evidence_path = app / "evidence" / "functional-equation-certificate.json"
    support._write_json(evidence_path, evidence)
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_mobius_functional_equation_rejects_boolean_matrix(tmp_path: Path) -> None:
    """Booleans in the coefficient matrix must be rejected even though they
    compare equal to the expected integer matrix."""
    task, app, logs = _prepare_mobius_case(tmp_path)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["coefficient_matrix"] = [
        [True, True, False],
        [False, True, True],
        [True, False, True],
    ]
    evidence = {
        "schema_version": "1",
        "task_id": submission["task_id"],
        "result": submission["result"],
        "limitations": submission["limitations"],
    }
    evidence_path = app / "evidence" / "functional-equation-certificate.json"
    support._write_json(evidence_path, evidence)
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0
