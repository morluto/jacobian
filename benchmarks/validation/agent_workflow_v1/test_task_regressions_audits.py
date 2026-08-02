from __future__ import annotations

import json
from fractions import Fraction
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


def test_polynomial_divisibility_derives_parameter_from_gcd(
    tmp_path: Path,
) -> None:
    """The verifier must derive the parameter from the recomputed linear gcd,
    not from a hard-coded integer.  Submitting a wrong parameter with the
    correct gcd/remainder must be rejected.
    """
    task, app, logs = support._prepare_case(
        tmp_path, "polynomial-divisibility-uniqueness", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["parameter"] = 3
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_polynomial_divisibility_derivation_helper_is_not_hardcoded() -> None:
    """Thread PRRT_kwDOThEfjc6VuwnB-derive: the parameter-derivation logic must
    compute the unique integer root of a controlled linear gcd, not match a
    hard-coded value.  Exercise ``derive_unique_parameter`` directly with linear
    gcds whose integral root is *not* the canonical 2, so a regressed
    implementation that only accepts ``parameter == 2`` would fail these cases.
    """
    import importlib
    import sys

    tests_dir = support.TASKS / "polynomial-divisibility-uniqueness" / "tests"
    sys.path.insert(0, str(tests_dir))
    try:
        verifier = importlib.import_module("verifier")
        # c0 + c1*a with root -c0/c1; pick roots that are not 2.
        assert verifier.derive_unique_parameter([Fraction(3), Fraction(1)]) == -3
        assert verifier.derive_unique_parameter([Fraction(-5), Fraction(1)]) == 5
        assert verifier.derive_unique_parameter([Fraction(0), Fraction(1)]) == 0
        # Non-integral rational root cannot license a UNIQUE_PARAMETER claim.
        assert verifier.derive_unique_parameter([Fraction(1), Fraction(2)]) is None
        # Nonlinear and constant gcds cannot certify uniqueness.
        assert (
            verifier.derive_unique_parameter([Fraction(6), Fraction(-5), Fraction(1)])
            is None
        )
        assert verifier.derive_unique_parameter([Fraction(2), Fraction(0)]) is None
    finally:
        sys.path.remove(str(tests_dir))
        sys.modules.pop("verifier", None)
        sys.modules.pop("verifier_support", None)


def test_polynomial_divisibility_rejects_nonlinear_gcd(
    tmp_path: Path,
) -> None:
    """If the submitted gcd is not linear, the verifier must reject the
    UNIQUE_PARAMETER conclusion because a nonlinear gcd does not establish
    a unique root.
    """
    task, app, logs = support._prepare_case(
        tmp_path, "polynomial-divisibility-uniqueness", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    # Replace the linear gcd [-2, 1] with (a-2)(a-3) = a^2 - 5a + 6, a
    # degree-2 polynomial that *contains* the correct root a=2 but also has
    # the extraneous root a=3, breaking uniqueness.  A weaker verifier that
    # only checks the submitted gcd vanishes at the reported parameter would
    # accept this; the correct verifier rejects it because the recomputed gcd
    # is linear and the submitted gcd is not equal to it.
    submission["result"]["common_gcd"] = [6, -5, 1]
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_polynomial_divisibility_accepts_canonical_solution(
    tmp_path: Path,
) -> None:
    """The canonical solution must still earn full reward after the fix,
    proving the verifier derives a=2 from the recomputed gcd a-2 rather than
    from a hard-coded expectation.
    """
    task, app, logs = support._prepare_case(
        tmp_path, "polynomial-divisibility-uniqueness", "computed"
    )
    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_polynomial_divisibility_rejects_extra_result_keys(
    tmp_path: Path,
) -> None:
    """Thread PRRT_kwDOThEfjc6VuwnA: reject result objects outside the
    advertised schema (additionalProperties: false).
    """
    task, app, logs = support._prepare_case(
        tmp_path, "polynomial-divisibility-uniqueness", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["extra_field"] = 0
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_polynomial_divisibility_rejects_missing_result_keys(
    tmp_path: Path,
) -> None:
    """Thread PRRT_kwDOThEfjc6VuwnA: the verifier requires the exact five
    result keys (additionalProperties: false), so a result object missing a
    required key must be rejected even when the remaining values are correct.
    """
    task, app, logs = support._prepare_case(
        tmp_path, "polynomial-divisibility-uniqueness", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    del submission["result"]["quotient"]
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_polynomial_divisibility_rejects_empty_evidence(
    tmp_path: Path,
) -> None:
    """Thread PRRT_kwDOThEfjc6Vu4rD: validate evidence content before awarding
    full credit.  An empty answer.txt with a matching digest must not pass.
    """
    task, app, logs = support._prepare_case(
        tmp_path, "polynomial-divisibility-uniqueness", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text("\n")
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 1.0
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == 0.0


def test_polynomial_divisibility_rejects_evidence_with_wrong_result_marker(
    tmp_path: Path,
) -> None:
    """Thread PRRT_kwDOThEfjc6Vu4rD: evidence whose RESULT_JSON marker does not
    match the submitted result must be rejected.
    """
    task, app, logs = support._prepare_case(
        tmp_path, "polynomial-divisibility-uniqueness", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text('Some derivation text.\nRESULT_JSON: {"parameter": 99}\n')
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 1.0
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == 0.0


def test_polynomial_divisibility_rejects_symlinked_submission(
    tmp_path: Path,
) -> None:
    """Thread PRRT_kwDOThEfjc6Vu4rE: reject symlinked submission artifacts."""
    task, app, logs = support._prepare_case(
        tmp_path, "polynomial-divisibility-uniqueness", "computed"
    )
    submission_path = app / "submission.json"
    alias = app / "evidence" / "alias.json"
    alias.write_text(submission_path.read_text())
    submission_path.unlink()
    submission_path.symlink_to(alias)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_polynomial_divisibility_reports_input_integrity_separately(
    tmp_path: Path,
) -> None:
    """Thread PRRT_kwDOThEfjc6Vu4rF: report input integrity separately from
    correctness.  Tampering with input.json must not zero the mathematical
    correctness signal or the evidence-validity signal; only the aggregate
    reward is gated on input integrity.
    """
    task, app, logs = support._prepare_case(
        tmp_path, "polynomial-divisibility-uniqueness", "computed"
    )
    input_path = app / "input.json"
    original = input_path.read_bytes()
    tampered = json.loads(original)
    tampered["extra_key"] = "tampered"
    support._write_json(input_path, tampered)

    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 1.0
    assert result["evidence_validity"] == 1.0
    assert result["input_integrity"] == 0.0
    assert result["reward"] == 0.0


def test_polynomial_divisibility_rejects_duplicate_evidence(
    tmp_path: Path,
) -> None:
    """Thread PRRT_kwDOThEfjc6VuwnB-evidence: enforce the maxItems: 1 evidence
    contract.  Repeating the same valid evidence descriptor must not pass even
    though each copy individually binds the expected evidence file.
    """
    task, app, logs = support._prepare_case(
        tmp_path, "polynomial-divisibility-uniqueness", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["evidence"].append(dict(submission["evidence"][0]))
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 1.0
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == 0.0


def test_polynomial_divisibility_rejects_deeply_nested_json(
    tmp_path: Path,
) -> None:
    """Thread PRRT_kwDOThEfjc6VuwnB-recursion: a deeply nested submission.json
    that triggers RecursionError in json.loads must score zero instead of
    crashing the verifier without writing reward.json.
    """
    task, app, logs = support._prepare_case(
        tmp_path, "polynomial-divisibility-uniqueness", "computed"
    )
    submission_path = app / "submission.json"
    # Write deeply nested JSON as a raw string to avoid RecursionError in
    # json.dumps during test setup; the verifier's json.loads will hit it.
    submission_path.write_text("[" * 10_000 + "0" + "]" * 10_000)

    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 0.0
    assert result["evidence_validity"] == 0.0
    assert result["reward"] == 0.0
