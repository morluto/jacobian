from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "real-rooted-sign-polynomials"


def _prepare_real_rooted_sign_case(tmp_path: Path):
    task, app, logs = support._prepare_case(
        tmp_path, "real-rooted-sign-polynomials", "computed"
    )
    (app / "evidence" / "classification-certificate.json").write_bytes(
        (task / "solution" / "classification-certificate.json").read_bytes()
    )
    return task, app, logs


def test_real_rooted_sign_polynomials_accepts_complete_audit(tmp_path: Path) -> None:
    task, app, logs = _prepare_real_rooted_sign_case(tmp_path)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_real_rooted_sign_polynomials_rejects_public_list_without_full_audit(
    tmp_path: Path,
) -> None:
    task, app, logs = _prepare_real_rooted_sign_case(tmp_path)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["candidate_audit"] = [
        case
        for case in submission["result"]["candidate_audit"]
        if case["all_roots_real"]
    ]
    evidence = {
        "schema_version": "1",
        "task_id": submission["task_id"],
        "result": submission["result"],
        "limitations": submission["limitations"],
    }
    evidence_path = app / "evidence" / "classification-certificate.json"
    support._write_json(evidence_path, evidence)
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_real_rooted_sign_polynomials_rejects_corrupted_discriminant(
    tmp_path: Path,
) -> None:
    task, app, logs = _prepare_real_rooted_sign_case(tmp_path)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["candidate_audit"][12]["discriminant"] = 0
    evidence = {
        "schema_version": "1",
        "task_id": submission["task_id"],
        "result": submission["result"],
        "limitations": submission["limitations"],
    }
    evidence_path = app / "evidence" / "classification-certificate.json"
    support._write_json(evidence_path, evidence)
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_real_rooted_sign_polynomials_rejects_checked_assurance_above_ceiling(
    tmp_path: Path,
) -> None:
    """CHECKED is above the COMPUTED ceiling and must force reward to zero."""
    task, app, logs = _prepare_real_rooted_sign_case(tmp_path)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["claimed_assurance"] = "CHECKED"
    evidence = {
        "schema_version": "1",
        "task_id": submission["task_id"],
        "result": submission["result"],
        "limitations": submission["limitations"],
    }
    evidence_path = app / "evidence" / "classification-certificate.json"
    support._write_json(evidence_path, evidence)
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0


def test_real_rooted_sign_polynomials_rejects_booleans_in_integer_fields(
    tmp_path: Path,
) -> None:
    """Booleans must not be accepted where integers are required."""
    task, app, logs = _prepare_real_rooted_sign_case(tmp_path)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["root_product_square"] = True
    evidence = {
        "schema_version": "1",
        "task_id": submission["task_id"],
        "result": submission["result"],
        "limitations": submission["limitations"],
    }
    evidence_path = app / "evidence" / "classification-certificate.json"
    support._write_json(evidence_path, evidence)
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_real_rooted_sign_polynomials_rejects_symlinked_submission(
    tmp_path: Path,
) -> None:
    """A symlinked submission.json must be rejected."""
    task, app, logs = _prepare_real_rooted_sign_case(tmp_path)
    submission_path = app / "submission.json"
    external = tmp_path / "external_submission.json"
    submission_path.rename(external)
    submission_path.symlink_to(external)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_real_rooted_sign_polynomials_rejects_missing_evidence_envelope(
    tmp_path: Path,
) -> None:
    """Evidence that stores the result directly without the required envelope
    must be rejected so the agent-visible schema and instruction are honest."""
    task, app, logs = _prepare_real_rooted_sign_case(tmp_path)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    evidence_path = app / "evidence" / "classification-certificate.json"
    support._write_json(evidence_path, submission["result"])
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == 0.0
