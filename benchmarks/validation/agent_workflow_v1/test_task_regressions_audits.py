from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.agent_workflow_v1 import support
from jsonschema import Draft202012Validator, ValidationError


def test_metric_tsp_scope_is_part_of_correctness(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "metric-tsp-proof-repair", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["scope"] = "wrong scope"
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["scope_accuracy"] == 0.0
    assert rejected["reward"] == 0.0


def test_gaussian_moment_audit_accepts_alternative_rational_parameter(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "gaussian-moment-generality-audit", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"].update(
        parameter_a="2",
        h_coefficients=["2", "1"],
        v_coefficients=["-1", "-3/4", "-1/8"],
        zeta={"numerator": ["0", "2"], "denominator": ["1", "-1"]},
    )
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text(
        "Finite moment checks cannot prove the result. The exact branch and "
        "formal square root cancel for every m>=1, giving the COMPUTED identity.\n"
        "RESULT_JSON: {}\n"
    )
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_gaussian_moment_audit_validates_v_via_correction_identity(
    tmp_path: Path,
) -> None:
    """The verifier must accept any v satisfying the correction-factor identity,
    not just a pinned closed-form expression.  a=3 produces a different v than
    a=1 or a=2, so passing here proves the identity check is what gates acceptance.
    """
    task, app, logs = support._prepare_case(
        tmp_path, "gaussian-moment-generality-audit", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"].update(
        parameter_a="3",
        h_coefficients=["3", "1"],
        v_coefficients=["-1", "-1/2", "-1/18"],
        zeta={"numerator": ["0", "3"], "denominator": ["1", "-1"]},
    )
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text(
        "Finite moment checks cannot prove the result. The exact branch and "
        "formal square root cancel for every m>=1, giving the COMPUTED identity.\n"
        "RESULT_JSON: {}\n"
    )
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_gaussian_moment_audit_rejects_v_violating_correction_identity(
    tmp_path: Path,
) -> None:
    """A v that does not satisfy 1-2t v(zeta)=(1-t)^(-2) must be rejected even
    when h and zeta are correct, proving the correction identity is the gate.
    """
    task, app, logs = support._prepare_case(
        tmp_path, "gaussian-moment-generality-audit", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["v_coefficients"] = ["-1", "-3/2", "0"]
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_gaussian_moment_audit_accepts_paraphrased_evidence(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "gaussian-moment-generality-audit", "computed"
    )
    evidence_path = app / "evidence" / "answer.txt"
    submission = json.loads((app / "submission.json").read_text())
    evidence_path.write_text(
        "Twelve finite checks are insufficient to prove the universal claim. "
        "The sqrt branch cancellation yields the all-exponents identity at "
        "COMPUTED assurance.\n"
        "RESULT_JSON: {}\n"
    )
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["evidence_validity"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_gaussian_moment_audit_accepts_paraphrased_limitations(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "gaussian-moment-generality-audit", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["limitations"] = [
        "This is not machine verification of the result.",
        "The 2D Gaussian Moments Conjecture is not settled here.",
    ]
    support._write_json(submission_path, submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["reward"] == pytest.approx(1.0)


def test_gaussian_moment_audit_rejects_evidence_without_result_marker(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "gaussian-moment-generality-audit", "computed"
    )
    evidence_path = app / "evidence" / "answer.txt"
    submission = json.loads((app / "submission.json").read_text())
    evidence_path.write_text(
        "Finite moment checks cannot justify the universal exponent claim. "
        "The formal square root cancels for every m>=1 at COMPUTED assurance.\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 1.0
    assert rejected["evidence_validity"] == 0.0


def test_gaussian_moment_audit_rejects_oversized_evidence(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "gaussian-moment-generality-audit", "computed"
    )
    evidence_path = app / "evidence" / "answer.txt"
    submission = json.loads((app / "submission.json").read_text())
    evidence_path.write_text("x" * 2_097_152)
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0


def test_gaussian_moment_audit_rejects_oversized_coefficient_arrays(
    tmp_path: Path,
) -> None:
    """A zeta represented by numerator/denominator multiplied by a long dense
    common polynomial preserves the rational function but would make the
    verifier's polynomial multiplication quadratic in an attacker-controlled
    length; the bounded coefficient gate rejects it without that work.
    """
    task, app, logs = support._prepare_case(
        tmp_path, "gaussian-moment-generality-audit", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    # t/(1-t) multiplied by (1+z)^17 worth of dense common factor on both sides.
    submission["result"]["zeta"] = {
        "numerator": ["0"] + ["1"] * 17,
        "denominator": ["1"] + ["0"] * 16 + ["-1"],
    }
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_metric_tsp_evidence_requires_calculations(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "metric-tsp-proof-repair", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text(
        "MST Euler shortcut optimal approximation\nRESULT_JSON: {}\n"
    )
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 1.0
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == pytest.approx(0.9)


def test_metric_tsp_accepts_factor_two_claim(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "metric-tsp-proof-repair", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["corrected_claim"] = "factor-2 approximation"
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0


def test_divisibility_accepts_schema_valid_integral_numbers(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "divisibility-construction-witness", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"] = {
        key: float(value) for key, value in submission["result"].items()
    }
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0


def test_modular_obstruction_requires_the_certified_modulus(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "modular-cubic-obstruction", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["modulus"] = 14
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_natural_subtraction_schema_requires_both_basis_entries() -> None:
    task = support._task("natural-subtraction-proof-repair")
    schema = json.loads((task / "environment" / "submission_schema.json").read_text())
    submission = json.loads((task / "solution" / "submission.json").read_text())
    submission["result"]["basis_order"] = []
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(submission)

    submission = json.loads((task / "solution" / "submission.json").read_text())
    submission["result"]["multipliers"] = ["1"]
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(submission)


def test_complex_power_sum_accepts_reversed_branch_order(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "complex-power-sum-elimination", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["branches"].reverse()
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (
            ("recurrence", "power_sums", "5", 3),
            {"numerator": 19, "denominator": 1},
        ),
        (
            ("branches", 0, "target", "sqrt17"),
            {"numerator": 3, "denominator": 1},
        ),
        (("branches", 0, "denominator_norms", "s_minus_12"), 31),
        (("branches",), []),
    ],
)
def test_complex_power_sum_rejects_corrupted_certificates(
    tmp_path: Path,
    path: tuple[object, ...],
    replacement: object,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "complex-power-sum-elimination", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    target = submission["result"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_autoformalization_rejects_positive_lean_compile_claim(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "autoformalization-semantic-audit", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text(
        "dimension dot product coordinate\n"
        "Both Lean declarations compile.\n"
        "RESULT_JSON: {}\n"
    )
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 1.0
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == pytest.approx(0.9)


def test_inverse_distance_audit_accepts_alternative_rational_direction(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "inverse-distance-remainder-audit", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["directional_witnesses"][0] = {
        "direction": [
            {"numerator": 4, "denominator": 5},
            {"numerator": 3, "denominator": 5},
        ],
        "quadratic_coefficient": {"numerator": 23, "denominator": 50},
        "sign": "POSITIVE",
        "normalized_residual_limit": "quadratic_coefficient",
    }
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (
            ("second_order_term", "dot_square_coefficient"),
            {"numerator": 1, "denominator": 1},
        ),
        (
            ("directional_witnesses", 0, "quadratic_coefficient"),
            {"numerator": 2, "denominator": 1},
        ),
        (
            ("response_audit", "defects"),
            ["CUBIC_REMAINDER_FALSE"],
        ),
    ],
)
def test_inverse_distance_audit_rejects_corrupted_certificates(
    tmp_path: Path,
    path: tuple[object, ...],
    replacement: object,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "inverse-distance-remainder-audit", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    target = submission["result"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0
