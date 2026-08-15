from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import (
    _fixtures,
    _metadata,
    _verifier,
)

TASK = "ga-action-local-finiteness-certificate"


def _case(tmp_path: Path):
    return _fixtures._prepare_case(tmp_path, TASK, "computed")


def _rewrite(app: Path, submission: dict) -> None:
    _fixtures._write_json(app / "submission.json", submission)


def _rational(value: Fraction) -> str:
    return str(value)


def test_oracle_passes(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    result = _verifier._run_verifier(task, app, logs)
    assert result.details["correctness"] == 1.0
    assert result.reward == pytest.approx(1.0)


def test_accepts_an_alternative_scaled_basis(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    result = submission["result"]
    scales = [Fraction(2), Fraction(3), Fraction(4), Fraction(5), Fraction(6)]
    original_coordinates = [Fraction(value) for value in result["f_coordinates"]]
    for index, poly in enumerate(result["basis"]):
        poly[0]["coefficient"] = _rational(scales[index])
    result["f_coordinates"] = [
        _rational(value / scale)
        for value, scale in zip(original_coordinates, scales, strict=True)
    ]
    for row, entries in enumerate(result["action_matrix"]):
        for column, poly in enumerate(entries):
            for term in poly:
                term["coefficient"] = _rational(
                    Fraction(term["coefficient"]) * scales[column] / scales[row]
                )
    _rewrite(app, submission)

    accepted = _verifier._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == pytest.approx(1.0)


@pytest.mark.parametrize(
    "corruption",
    ["singular_basis", "wrong_coordinates", "wrong_action"],
)
def test_rejects_corrupted_certificates(tmp_path: Path, corruption: str) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    if corruption == "singular_basis":
        submission["result"]["basis"][4] = submission["result"]["basis"][3]
    elif corruption == "wrong_coordinates":
        submission["result"]["f_coordinates"][0] = "8"
    else:
        submission["result"]["action_matrix"][0][4][0]["coefficient"] = "2"
    _rewrite(app, submission)

    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_visible_input_tampering(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    source = json.loads((app / "input.json").read_text())
    source["f"][0]["coefficient"] = "2"
    _fixtures._write_json(app / "input.json", source)

    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_accepts_unreduced_rational_coordinates(tmp_path: Path) -> None:
    """Coordinates are not sparse term lists; unreduced rationals are schema-valid."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    result = submission["result"]
    original = Fraction(result["f_coordinates"][0])
    unreduced = f"{original.numerator * 2}/{original.denominator * 2}"
    assert str(Fraction(unreduced)) == str(original)
    result["f_coordinates"][0] = unreduced
    _rewrite(app, submission)

    accepted = _verifier._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == pytest.approx(1.0)


def test_accepts_long_rational_coordinates(tmp_path: Path) -> None:
    """No undisclosed byte cap on rational coordinate strings."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    result = submission["result"]
    big = Fraction(10) ** 80
    scales = [big] + [Fraction(1)] * 4
    original_coordinates = [Fraction(value) for value in result["f_coordinates"]]
    for index, poly in enumerate(result["basis"]):
        poly[0]["coefficient"] = _rational(
            scales[index] * Fraction(poly[0]["coefficient"])
        )
    result["f_coordinates"] = [
        _rational(value / scale)
        for value, scale in zip(original_coordinates, scales, strict=True)
    ]
    assert len(result["f_coordinates"][0]) > 80
    for row, entries in enumerate(result["action_matrix"]):
        for column, poly in enumerate(entries):
            for term in poly:
                term["coefficient"] = _rational(
                    Fraction(term["coefficient"]) * scales[column] / scales[row]
                )
    _rewrite(app, submission)

    accepted = _verifier._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == pytest.approx(1.0)


def test_rejects_noncanonical_basis_coefficient(tmp_path: Path) -> None:
    """Sparse term lists must use canonical reduced rationals per the contract."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    result = submission["result"]
    original = Fraction(result["basis"][0][0]["coefficient"])
    result["basis"][0][0]["coefficient"] = (
        f"{original.numerator * 2}/{original.denominator * 2}"
    )
    _rewrite(app, submission)

    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_task_metadata_declares_input_binding_decoupled() -> None:
    """Input-binding decoupling is declared in task-local metadata, not a global registry."""
    assert _metadata.is_input_binding_decoupled(TASK) is True
    metadata = _metadata.load_task_contract_metadata(TASK)
    assert metadata.get("input_binding_decoupled") is True
