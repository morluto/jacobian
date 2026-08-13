from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "exact-farkas-ldl-slice"


def _prepare_farkas_slice_case(tmp_path: Path):
    task, app, logs = support._prepare_case(
        tmp_path, "exact-farkas-ldl-slice", "computed"
    )
    (app / "evidence" / "farkas-slice-certificate.json").write_bytes(
        (task / "solution" / "farkas-slice-certificate.json").read_bytes()
    )
    return task, app, logs


def _bind_farkas_slice(app: Path, submission: dict) -> None:
    evidence = {
        "schema_version": "1",
        "task_id": submission["task_id"],
        "result": submission["result"],
        "limitations": submission["limitations"],
    }
    evidence_path = app / "evidence" / "farkas-slice-certificate.json"
    support._write_json(evidence_path, evidence)
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)


def test_exact_farkas_slice_accepts_sylvester_certificate(tmp_path: Path) -> None:
    task, app, logs = _prepare_farkas_slice_case(tmp_path)
    accepted = support._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == pytest.approx(1.0)


def test_exact_farkas_slice_accepts_ldl_certificate(tmp_path: Path) -> None:
    task, app, logs = _prepare_farkas_slice_case(tmp_path)
    frozen = json.loads((app / "input.json").read_text())
    matrix = [
        [Fraction(item["numerator"], item["denominator"]) for item in row]
        for row in frozen["matrix"]
    ]
    lower = [[Fraction(i == j) for j in range(4)] for i in range(4)]
    diagonal = [Fraction() for _ in range(4)]
    for i in range(4):
        diagonal[i] = matrix[i][i] - sum(
            lower[i][k] * lower[i][k] * diagonal[k] for k in range(i)
        )
        for j in range(i + 1, 4):
            lower[j][i] = (
                matrix[j][i]
                - sum(lower[j][k] * lower[i][k] * diagonal[k] for k in range(i))
            ) / diagonal[i]

    def encode(value: Fraction) -> dict:
        return {"numerator": value.numerator, "denominator": value.denominator}

    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["proof_mode"] = "LDL"
    submission["result"]["positive_definite_certificate"] = {
        "l": [[encode(item) for item in row] for row in lower],
        "d": [encode(item) for item in diagonal],
    }
    _bind_farkas_slice(app, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == pytest.approx(1.0)


def test_exact_farkas_slice_rejects_corrupted_minor(tmp_path: Path) -> None:
    task, app, logs = _prepare_farkas_slice_case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["positive_definite_certificate"][
        "leading_principal_determinants"
    ][2]["numerator"] += 1
    _bind_farkas_slice(app, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_exact_farkas_slice_rejects_full_certificate_overclaim(tmp_path: Path) -> None:
    task, app, logs = _prepare_farkas_slice_case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["scope"] = "FULL_SIX_BLOCK_FARKAS_CERTIFICATE"
    _bind_farkas_slice(app, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["scope_accuracy"] == 0.0
    assert rejected.reward == 0.0


def test_exact_farkas_slice_rejects_checked_assurance_above_ceiling(
    tmp_path: Path,
) -> None:
    """CHECKED is above the COMPUTED ceiling and must force reward to zero.

    Mathematical correctness remains independent of the assurance ceiling.
    """
    task, app, logs = _prepare_farkas_slice_case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["claimed_assurance"] = "CHECKED"
    _bind_farkas_slice(app, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 1.0
    assert rejected.details["assurance_calibration"] == 0.0
    assert rejected.reward == 0.0


def test_exact_farkas_slice_rejects_oversized_submission(tmp_path: Path) -> None:
    """An oversized submission.json must be rejected without crashing."""
    task, app, logs = _prepare_farkas_slice_case(tmp_path)
    (app / "submission.json").write_text('{"a": 1' + ", " * (2 * 1024 * 1024) + "}")

    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_exact_farkas_slice_rejects_missing_evidence_envelope(tmp_path: Path) -> None:
    """Evidence must include the required envelope around the result."""
    task, app, logs = _prepare_farkas_slice_case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence_path = app / "evidence" / "farkas-slice-certificate.json"
    support._write_json(evidence_path, submission["result"])
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["evidence_validity"] == 0.0
    assert rejected.reward == 0.0


def test_exact_farkas_slice_accepts_large_valid_evidence_padding(
    tmp_path: Path,
) -> None:
    """Legal JSON whitespace must not create a hidden evidence-size limit."""
    task, app, logs = _prepare_farkas_slice_case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence_path = app / "evidence" / "farkas-slice-certificate.json"
    evidence_path.write_text(
        " \n" * (600 * 1024)
        + json.dumps(
            {
                "schema_version": "1",
                "task_id": submission["task_id"],
                "result": submission["result"],
                "limitations": submission["limitations"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        + " " * (600 * 1024)
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted.details["evidence_validity"] == 1.0
    assert accepted.reward == pytest.approx(1.0)


@pytest.mark.parametrize("prefix,suffix", [("garbage", ""), ("", "garbage")])
def test_exact_farkas_slice_rejects_evidence_json_garbage(
    tmp_path: Path,
    prefix: str,
    suffix: str,
) -> None:
    """Streaming acceptance must still reject content outside the JSON value."""
    task, app, logs = _prepare_farkas_slice_case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence_path = app / "evidence" / "farkas-slice-certificate.json"
    payload = {
        "schema_version": "1",
        "task_id": submission["task_id"],
        "result": submission["result"],
        "limitations": submission["limitations"],
    }
    evidence_path.write_text(prefix + json.dumps(payload) + suffix)
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["evidence_validity"] == 0.0
    assert rejected.reward == 0.0
