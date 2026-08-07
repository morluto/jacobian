from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "ga-action-local-finiteness-certificate"
VALID_EVIDENCE = (
    "A finite coefficient expansion alone does not prove preservation. "
    "The coaction and group law establish that the submitted subspace is "
    "invariant under the action.\n"
)


def _case(tmp_path: Path):
    return support._prepare_case(tmp_path, TASK, "computed")


def _rewrite(app: Path, submission: dict) -> None:
    evidence = app / "evidence/answer.txt"
    evidence.write_text(VALID_EVIDENCE)
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)


def test_accepts_equivalent_limitation_wording(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = [
        "This degree-four action is only a frozen certificate, not a proof of the general theorem."
    ]
    _rewrite(app, submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["limitation_accuracy"] == 1.0
    assert accepted["reward"] == 1.0


def _rational(value: Fraction) -> str:
    return str(value)


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

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_equivalent_explanatory_evidence_needs_no_private_marker(
    tmp_path: Path,
) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence/answer.txt"
    evidence.write_text(
        "A finite coefficient expansion by itself does not establish that the "
        "span is preserved. The coaction identity, equivalently the group law, "
        "shows that the submitted subspace is invariant under the action.\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["evidence_validity"] == 1.0
    assert accepted["reward"] == 1.0


def test_accepts_composition_identity_wording(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence/answer.txt"
    evidence.write_text(
        "A finite coefficient expansion alone does not establish preservation. "
        "The exact action matrix satisfies the composition identity, which "
        "makes the submitted span invariant under every parameter.\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["evidence_validity"] == 1.0
    assert accepted["reward"] == 1.0


def test_rejects_self_contradictory_explanation(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence/answer.txt"
    evidence.write_text(
        VALID_EVIDENCE
        + "However, the composition identity is false and the span is not invariant.\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == 0.0


def test_unrelated_evidence_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence/answer.txt"
    evidence.write_text("The certificate has many exact polynomial entries.\n")
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == 0.0


def test_unverified_assurance_is_accepted(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["claimed_assurance"] = "UNVERIFIED"
    _rewrite(app, submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["assurance_calibration"] == 1.0
    assert accepted["reward"] == 1.0


def test_false_limitation_is_rejected_with_diagnostic(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = ["The general theorem is formally verified."]
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["limitation_accuracy"] == 0.0
    assert rejected["reward"] == 0.0


@pytest.mark.parametrize(
    "corruption",
    ["singular_basis", "wrong_coordinates", "wrong_action", "false_assurance"],
)
def test_rejects_corrupted_certificates(tmp_path: Path, corruption: str) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    if corruption == "singular_basis":
        submission["result"]["basis"][4] = submission["result"]["basis"][3]
    elif corruption == "wrong_coordinates":
        submission["result"]["f_coordinates"][0] = "8"
    elif corruption == "wrong_action":
        submission["result"]["action_matrix"][0][4][0]["coefficient"] = "2"
    else:
        submission["claimed_assurance"] = "VERIFIED"
    _rewrite(app, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0


def test_rejects_visible_input_tampering(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    source = json.loads((app / "input.json").read_text())
    source["f"][0]["coefficient"] = "2"
    support._write_json(app / "input.json", source)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["input_binding"] == 0.0
    assert rejected["correctness"] == 1.0
    assert rejected["reward"] == 0.0


def test_envelope_failure_preserves_independent_diagnostics(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["evidence"][0]["path"] = "../answer.txt"
    support._write_json(app / "submission.json", submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["protocol_compliance"] == 0.0
    assert rejected["correctness"] == 1.0
    assert rejected["scope_accuracy"] == 1.0
    assert rejected["assurance_calibration"] == 1.0
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == 0.0


def test_unhashable_assurance_preserves_scope_diagnostic(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["claimed_assurance"] = []
    support._write_json(app / "submission.json", submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["scope_accuracy"] == 1.0
    assert rejected["assurance_calibration"] == 0.0
    assert rejected["false_certification"] is False
    assert rejected["reward"] == 0.0


def test_accepts_unreduced_rational_coordinates(tmp_path: Path) -> None:
    """Coordinates are not sparse term lists; unreduced rationals are schema-valid."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    result = submission["result"]
    original = Fraction(result["f_coordinates"][0])
    # Replace the first coordinate with an unreduced equivalent, e.g. 7 -> 14/2.
    unreduced = f"{original.numerator * 2}/{original.denominator * 2}"
    assert str(Fraction(unreduced)) == str(original)
    result["f_coordinates"][0] = unreduced
    _rewrite(app, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_accepts_long_rational_coordinates(tmp_path: Path) -> None:
    """No undisclosed byte cap on rational coordinate strings."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    result = submission["result"]
    # Scale basis[0] by a large number so the coordinate string exceeds 80 chars.
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

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


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

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_task_metadata_declares_input_binding_decoupled() -> None:
    """Input-binding decoupling is declared in task-local metadata, not a global registry."""
    assert support.is_input_binding_decoupled(TASK) is True
    metadata = support.load_task_contract_metadata(TASK)
    assert metadata.get("input_binding_decoupled") is True
