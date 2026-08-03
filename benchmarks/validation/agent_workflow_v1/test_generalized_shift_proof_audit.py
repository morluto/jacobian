from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from benchmarks.validation.agent_workflow_v1 import support

TASK = "generalized-shift-proof-audit"


def load_case(tmp_path: Path):
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    shutil.copy2(
        task / "solution" / "audit-certificate.json",
        app / "evidence" / "audit-certificate.json",
    )
    submission_path = app / "submission.json"
    return task, app, logs, submission_path, json.loads(submission_path.read_text())


def write_bound(app: Path, submission_path: Path, submission: dict) -> None:
    evidence = app / "evidence" / "audit-certificate.json"
    support._write_json(evidence, submission["result"])
    submission["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence.read_bytes()).hexdigest()
    )
    support._write_json(submission_path, submission)


def test_canonical_certificate_receives_full_reward(tmp_path: Path) -> None:
    task, app, logs, _, _ = load_case(tmp_path)
    result = support._run_verifier(task, app, logs)
    assert result["reward"] == pytest.approx(1.0)


def test_alternative_exact_certificates_are_accepted(tmp_path: Path) -> None:
    task, app, logs, submission_path, submission = load_case(tmp_path)
    submission["result"] = {
        "collision": {
            "first_index": -2,
            "second_index": 3,
            "alpha_first": 5,
            "alpha_second": 0,
        },
        "fourier_block": {
            "size": 9,
            "operator_norm_squared": {"numerator": 9, "denominator": 4},
        },
        "norm_direction": {
            "valid_relation": "OPERATOR_NORM_LE_HILBERT_SCHMIDT_NORM",
            "diagonal_entries": [
                {"numerator": -5, "denominator": 2},
                {"numerator": 1, "denominator": 3},
            ],
            "operator_norm_squared": {"numerator": 25, "denominator": 4},
            "hilbert_schmidt_norm_squared": {
                "numerator": 229,
                "denominator": 36,
            },
        },
        "radical_domain": {"m": 17, "radicand": -16, "real_status": "NOT_REAL"},
    }
    write_bound(app, submission_path, submission)
    result = support._run_verifier(task, app, logs)
    assert result["reward"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("section", "field", "bad_value"),
    [
        ("collision", "alpha_second", 1),
        ("fourier_block", "operator_norm_squared", {"numerator": 25, "denominator": 4}),
        (
            "norm_direction",
            "operator_norm_squared",
            {"numerator": 25, "denominator": 1},
        ),
        ("radical_domain", "radicand", -1),
    ],
)
def test_corrupted_defect_certificates_are_rejected(
    tmp_path: Path, section: str, field: str, bad_value: object
) -> None:
    task, app, logs, submission_path, submission = load_case(tmp_path)
    submission["result"][section][field] = bad_value
    write_bound(app, submission_path, submission)
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


def test_verified_claim_is_rejected(tmp_path: Path) -> None:
    task, app, logs, submission_path, submission = load_case(tmp_path)
    submission["claimed_assurance"] = "VERIFIED"
    support._write_json(submission_path, submission)
    result = support._run_verifier(task, app, logs)
    assert result["reward"] == 0.0
