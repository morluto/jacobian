from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.agent_workflow_v1 import support

TASK = "nonlinear-recurrence-crossing-certificate"


def _prepare_nonlinear_recurrence_case(tmp_path: Path):
    task, app, logs = support._prepare_case(
        tmp_path, "nonlinear-recurrence-crossing-certificate", "computed"
    )
    source = task / "solution" / "nonlinear-recurrence-certificate.json"
    target = app / "evidence" / "nonlinear-recurrence-certificate.json"
    target.write_bytes(source.read_bytes())
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["evidence"][0]["sha256"] = support._digest(target)
    support._write_json(submission_path, submission)
    return task, app, logs


def _bind_nonlinear_recurrence_evidence(app: Path, submission: dict) -> None:
    path = app / "evidence" / "nonlinear-recurrence-certificate.json"
    evidence = json.loads(path.read_text())
    evidence["result"] = submission["result"]
    support._write_json(path, evidence)
    submission["evidence"][0]["sha256"] = support._digest(path)


def test_nonlinear_recurrence_accepts_reordered_terminal_bounds(
    tmp_path: Path,
) -> None:
    task, app, logs = _prepare_nonlinear_recurrence_case(tmp_path)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["terminal_bounds"].reverse()
    _bind_nonlinear_recurrence_evidence(app, submission)
    support._write_json(submission_path, submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["reward"] == pytest.approx(1.0)


def test_nonlinear_recurrence_rejects_insufficient_phase_budget(
    tmp_path: Path,
) -> None:
    task, app, logs = _prepare_nonlinear_recurrence_case(tmp_path)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["phase_transitions"] = 1789
    _bind_nonlinear_recurrence_evidence(app, submission)
    support._write_json(submission_path, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_nonlinear_recurrence_rejects_decimalized_decrement(
    tmp_path: Path,
) -> None:
    task, app, logs = _prepare_nonlinear_recurrence_case(tmp_path)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["decrement_lower_bound"] = {
        "numerator": 175,
        "denominator": 100,
    }
    _bind_nonlinear_recurrence_evidence(app, submission)
    support._write_json(submission_path, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_nonlinear_recurrence_accepts_alternative_threshold_certificate(
    tmp_path: Path,
) -> None:
    """A certificate with threshold 3, decrement 5/3, and 1880 transitions
    is mathematically valid and must be accepted rather than compared
    against the Oracle's hard-coded threshold 4.
    """
    task, app, logs = _prepare_nonlinear_recurrence_case(tmp_path)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    result = submission["result"]
    result["threshold"] = {"numerator": 3, "denominator": 1}
    result["decrement_lower_bound"] = {"numerator": 5, "denominator": 3}
    result["phase_transitions"] = 1880
    result["threshold_index_upper"] = 1881
    result["negative_index_upper"] = 1884
    result["terminal_bounds"] = [
        {
            "input_lower": {"numerator": 0, "denominator": 1},
            "input_upper": {"numerator": 7, "denominator": 4},
            "output_lower": "NEGATIVE_INFINITY",
            "output_upper": {"numerator": 33, "denominator": 28},
        },
        {
            "input_lower": {"numerator": 0, "denominator": 1},
            "input_upper": {"numerator": 1, "denominator": 1},
            "output_lower": "NEGATIVE_INFINITY",
            "output_upper": {"numerator": 0, "denominator": 1},
        },
        {
            "input_lower": {"numerator": 1, "denominator": 1},
            "input_upper": {"numerator": 33, "denominator": 28},
            "output_lower": {"numerator": 0, "denominator": 1},
            "output_upper": {"numerator": 305, "denominator": 924},
        },
    ]
    _bind_nonlinear_recurrence_evidence(app, submission)
    support._write_json(submission_path, submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_nonlinear_recurrence_rejects_boolean_coefficients(
    tmp_path: Path,
) -> None:
    """Boolean values must not spoof integer coefficients."""
    task, app, logs = _prepare_nonlinear_recurrence_case(tmp_path)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["potential_identity_coefficients"] = [True, -2, True]
    _bind_nonlinear_recurrence_evidence(app, submission)
    support._write_json(submission_path, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_nonlinear_recurrence_rejects_float_index_fields(
    tmp_path: Path,
) -> None:
    """Float-valued index fields must be rejected even when numerically equal."""
    task, app, logs = _prepare_nonlinear_recurrence_case(tmp_path)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["threshold_index_upper"] = 1791.0
    submission["result"]["negative_index_upper"] = 1794.0
    _bind_nonlinear_recurrence_evidence(app, submission)
    support._write_json(submission_path, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0
