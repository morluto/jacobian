from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import (
    _fixtures,
    _verifier,
)

TASK = "polynomial-root-localization-certificate"


def _case(tmp_path: Path):
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    evidence = task / "solution" / "root-bound-certificate.json"
    target = app / "evidence" / "root-bound-certificate.json"
    certificate = json.loads(evidence.read_text())
    certificate.pop("limitations", None)
    _fixtures._write_json(target, certificate)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["witness"][0]["sha256"] = _fixtures._digest(target)
    _fixtures._write_json(submission_path, submission)
    return task, app, logs


def _bind_evidence(app: Path, submission: dict) -> None:
    evidence_path = app / "evidence" / "root-bound-certificate.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["result"] = submission["result"]
    evidence.pop("limitations", None)
    _fixtures._write_json(evidence_path, evidence)
    submission["witness"][0]["sha256"] = _fixtures._digest(evidence_path)


def _rewrite(app: Path, submission: dict) -> None:
    _bind_evidence(app, submission)
    _fixtures._write_json(app / "submission.json", submission)


def test_accepts_permuted_weight_order(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["nonnegative_weights"].reverse()
    submission["result"]["root_identity_rhs"] = (
        submission["result"]["root_identity_rhs"][1:]
        + submission["result"]["root_identity_rhs"][:1]
    )
    _rewrite(app, submission)

    accepted = _verifier._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == pytest.approx(1.0)


def test_rejects_corrupted_symbolic_coefficient(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["one_minus_z_times_q"][2] = [0, -1, 0, 1]
    _rewrite(app, submission)

    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_boolean_in_weight_sum(tmp_path: Path) -> None:
    """Boolean values must not spoof integer entries in weight_sum."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["weight_sum"] = [True, 0, 0, 0]
    _rewrite(app, submission)

    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_boolean_in_controlled_powers(tmp_path: Path) -> None:
    """Boolean values must not spoof integer entries in controlled_powers."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["controlled_powers"] = [True, 2, 3, 4]
    _rewrite(app, submission)

    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_boolean_in_evidence_copy(tmp_path: Path) -> None:
    """Boolean ``true`` in the certificate must not pass as an exact result copy."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence_path = app / "evidence" / "root-bound-certificate.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["result"] = json.loads(
        json.dumps(submission["result"], separators=(",", ":")).replace(
            '"weight_sum":[1,0,0,0]', '"weight_sum":[true,0,0,0]'
        )
    )
    _fixtures._write_json(evidence_path, evidence)
    submission["witness"][0]["sha256"] = _fixtures._digest(evidence_path)
    _fixtures._write_json(app / "submission.json", submission)

    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.reward == 0.0
    assert rejected.reward == 0.0
