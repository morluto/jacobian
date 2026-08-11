from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "reciprocal-polynomial-classification"


def test_reciprocal_polynomial_classification_accepts_oracle(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "reciprocal-polynomial-classification", "computed"
    )
    target = app / "evidence" / "classification-certificate.json"
    target.write_bytes((task / "solution" / target.name).read_bytes())
    accepted = support._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == pytest.approx(1.0)


def test_reciprocal_polynomial_classification_accepts_alternative_member(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "reciprocal-polynomial-classification", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    m = 7
    quotient = [
        {"exponent": 2 * index, "coefficient": (-1) ** index} for index in range(m)
    ]
    result = submission["result"]
    result.update(
        {
            "m": m,
            "polynomial_terms": [
                {"exponent": item["exponent"] + 1, "coefficient": item["coefficient"]}
                for item in quotient
            ],
            "quotient_terms": quotient,
            "reverse_terms": quotient,
            "reverse_quotient_constant": 1,
            "degree": 13,
            "quotient_degree": 12,
        }
    )
    target = app / "evidence" / "classification-certificate.json"
    certificate = json.loads(
        (task / "solution" / "classification-certificate.json").read_text()
    )
    certificate["result"] = result
    support._write_json(target, certificate)
    submission["evidence"][0]["sha256"] = support._digest(target)
    support._write_json(submission_path, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == pytest.approx(1.0)


def test_reciprocal_polynomial_classification_rejects_corrupted_coefficient(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "reciprocal-polynomial-classification", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["polynomial_terms"][3]["coefficient"] = 1
    target = app / "evidence" / "classification-certificate.json"
    certificate = json.loads(
        (task / "solution" / "classification-certificate.json").read_text()
    )
    certificate["result"] = submission["result"]
    support._write_json(target, certificate)
    submission["evidence"][0]["sha256"] = support._digest(target)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_reciprocal_polynomial_classification_accepts_schema_valid_integral_numbers(
    tmp_path: Path,
) -> None:
    """JSON Schema's ``integer`` type accepts integral floats like ``8.0``,
    so the verifier must accept them while still rejecting booleans.
    """
    task, app, logs = support._prepare_case(
        tmp_path, "reciprocal-polynomial-classification", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    result = submission["result"]
    result["m"] = float(result["m"])
    result["degree"] = float(result["degree"])
    result["quotient_degree"] = float(result["quotient_degree"])
    result["reverse_quotient_constant"] = float(result["reverse_quotient_constant"])
    for section in ("polynomial_terms", "quotient_terms", "reverse_terms"):
        for term in result[section]:
            term["exponent"] = float(term["exponent"])
            term["coefficient"] = float(term["coefficient"])
    target = app / "evidence" / "classification-certificate.json"
    certificate = json.loads(
        (task / "solution" / "classification-certificate.json").read_text()
    )
    certificate["result"] = result
    support._write_json(target, certificate)
    submission["evidence"][0]["sha256"] = support._digest(target)
    support._write_json(submission_path, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == pytest.approx(1.0)


def test_reciprocal_polynomial_classification_rejects_boolean_coefficient(
    tmp_path: Path,
) -> None:
    """Booleans are not schema-valid integers and must not spoof unit terms."""
    task, app, logs = support._prepare_case(
        tmp_path, "reciprocal-polynomial-classification", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["polynomial_terms"][0]["coefficient"] = True
    target = app / "evidence" / "classification-certificate.json"
    certificate = json.loads(
        (task / "solution" / "classification-certificate.json").read_text()
    )
    certificate["result"] = submission["result"]
    support._write_json(target, certificate)
    submission["evidence"][0]["sha256"] = support._digest(target)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_reciprocal_polynomial_classification_accepts_unsorted_terms(
    tmp_path: Path,
) -> None:
    """Neither the instruction nor the schema requires ascending exponent
    order, so a mathematically identical witness in a different order must
    receive full reward.
    """
    task, app, logs = support._prepare_case(
        tmp_path, "reciprocal-polynomial-classification", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    result = submission["result"]
    for section in ("polynomial_terms", "quotient_terms", "reverse_terms"):
        result[section].reverse()
    target = app / "evidence" / "classification-certificate.json"
    certificate = json.loads(
        (task / "solution" / "classification-certificate.json").read_text()
    )
    certificate["result"] = result
    support._write_json(target, certificate)
    submission["evidence"][0]["sha256"] = support._digest(target)
    support._write_json(submission_path, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == pytest.approx(1.0)


def test_reciprocal_polynomial_classification_rejects_checked_above_ceiling(
    tmp_path: Path,
) -> None:
    """CHECKED is above the task's COMPUTED assurance ceiling, so it is an
    unsupported certification that forces reward to zero rather than earning
    the partial reward a below-ceiling mismatch would receive.
    """
    task, app, logs = support._prepare_case(
        tmp_path, "reciprocal-polynomial-classification", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["claimed_assurance"] = "CHECKED"
    target = app / "evidence" / "classification-certificate.json"
    target.write_bytes(
        (task / "solution" / "classification-certificate.json").read_bytes()
    )
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected.reward == 0.0


def test_reciprocal_polynomial_classification_rejects_boolean_in_evidence_copy(
    tmp_path: Path,
) -> None:
    """When the evidence certificate replaces an integer coefficient with
    boolean ``true``, Python equality treats them as equal but the evidence
    does not exactly copy the result.  The verifier must reject this.
    """
    task, app, logs = support._prepare_case(
        tmp_path, "reciprocal-polynomial-classification", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    target = app / "evidence" / "classification-certificate.json"
    certificate = json.loads(
        (task / "solution" / "classification-certificate.json").read_text()
    )
    certificate["result"] = json.loads(
        json.dumps(submission["result"], separators=(",", ":")).replace(
            '"coefficient":1', '"coefficient":true'
        )
    )
    support._write_json(target, certificate)
    submission["evidence"][0]["sha256"] = support._digest(target)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["evidence_validity"] == 0.0
    assert rejected.reward == 0.0
