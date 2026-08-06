from __future__ import annotations

import hashlib
import json
import math
import shutil
from fractions import Fraction
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "rank-one-determinant-limit"


def encode(q: Fraction) -> dict[str, int]:
    return {"numerator": q.numerator, "denominator": q.denominator}


def exact_sample(n: int) -> dict[str, object]:
    product = math.prod(i**3 - i for i in range(2, n + 1))
    reciprocal = sum((Fraction(1, i**3 - i) for i in range(2, n + 1)), Fraction())
    return {
        "n": n,
        "diagonal_product": product,
        "reciprocal_sum": encode(reciprocal),
        "determinant_constant": product,
        "determinant_linear": int(-product * reciprocal),
        "lambda": encode(1 / reciprocal),
    }


def load_case(tmp_path: Path):
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    shutil.copy2(
        task / "solution" / "determinant-certificate.json",
        app / "evidence" / "determinant-certificate.json",
    )
    submission_path = app / "submission.json"
    return task, app, logs, submission_path, json.loads(submission_path.read_text())


def write_bound(app: Path, submission_path: Path, submission: dict) -> None:
    evidence = app / "evidence" / "determinant-certificate.json"
    support._write_json(evidence, submission["result"])
    submission["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence.read_bytes()).hexdigest()
    )
    support._write_json(submission_path, submission)


def test_canonical_oracle_receives_full_reward(tmp_path: Path) -> None:
    task, app, logs, _, _ = load_case(tmp_path)
    assert support._run_verifier(task, app, logs)["reward"] == pytest.approx(1.0)


def test_alternative_sample_sizes_and_swapped_tail_factors(tmp_path: Path) -> None:
    task, app, logs, submission_path, submission = load_case(tmp_path)
    submission["result"]["samples"] = [
        exact_sample(n) for n in (4, 6, 7, 9, 10, 12, 18)
    ]
    submission["result"]["tail_gap"]["affine_shifts"] = [2, -1]
    write_bound(app, submission_path, submission)
    assert support._run_verifier(task, app, logs)["reward"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("section", "mutation"),
    [
        (
            "partial_fraction_coefficients",
            lambda value: value.__setitem__(0, encode(Fraction(2, 3))),
        ),
        ("samples", lambda value: value[0].__setitem__("determinant_linear", -2)),
        ("tail_gap", lambda value: value.__setitem__("numerator", 7)),
        ("limit", lambda value: value.update(encode(Fraction(5, 1)))),
    ],
)
def test_corrupted_symbolic_certificates_are_rejected(
    tmp_path: Path, section: str, mutation
) -> None:
    task, app, logs, submission_path, submission = load_case(tmp_path)
    mutation(submission["result"][section])
    write_bound(app, submission_path, submission)
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


def test_verified_claim_is_rejected(tmp_path: Path) -> None:
    task, app, logs, submission_path, submission = load_case(tmp_path)
    submission["claimed_assurance"] = "VERIFIED"
    support._write_json(submission_path, submission)
    result = support._run_verifier(task, app, logs)
    assert result["false_certification"] is True
    assert result["reward"] == 0.0


def test_checked_claim_above_computed_ceiling_is_rejected(tmp_path: Path) -> None:
    task, app, logs, submission_path, submission = load_case(tmp_path)
    submission["claimed_assurance"] = "CHECKED"
    support._write_json(submission_path, submission)
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


def test_scalar_sample_elements_rejected_without_crash(tmp_path: Path) -> None:
    """A malformed submission with scalar sample elements is rejected cleanly
    instead of crashing with AttributeError before reward.json is written.
    """
    task, app, logs, submission_path, submission = load_case(tmp_path)
    submission["result"]["samples"] = [1, 2, 3, 4, 5, 6]
    write_bound(app, submission_path, submission)
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


def test_empty_limitations_rejected(tmp_path: Path) -> None:
    """An otherwise valid submission with empty limitations does not receive
    full reward because the frozen limitations are part of scope scoring.
    """
    task, app, logs, submission_path, submission = load_case(tmp_path)
    submission["limitations"] = []
    write_bound(app, submission_path, submission)
    result = support._run_verifier(task, app, logs)
    assert result["scope_accuracy"] == 0.0
    assert result["reward"] < 1.0


def test_wrong_limitations_rejected(tmp_path: Path) -> None:
    """Arbitrary limitation strings do not satisfy the frozen limitations."""
    task, app, logs, submission_path, submission = load_case(tmp_path)
    submission["limitations"] = ["some other limitation"]
    write_bound(app, submission_path, submission)
    result = support._run_verifier(task, app, logs)
    assert result["scope_accuracy"] == 0.0
    assert result["reward"] < 1.0


def test_float_determinant_coefficients_rejected(tmp_path: Path) -> None:
    """Determinant coefficients serialized as floats are rejected even though
    Python numeric equality would accept them.
    """
    task, app, logs, submission_path, submission = load_case(tmp_path)
    sample = submission["result"]["samples"][0]
    product = sample["diagonal_product"]
    sample["diagonal_product"] = float(product)
    sample["determinant_constant"] = float(product)
    sample["determinant_linear"] = float(sample["determinant_linear"])
    write_bound(app, submission_path, submission)
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


def test_unreduced_rational_coefficients_accepted(tmp_path: Path) -> None:
    """Equivalent unreduced rationals (e.g. 2/12 for 1/6) are accepted as the
    same Fraction value.
    """
    task, app, logs, submission_path, submission = load_case(tmp_path)
    # The limit is 4/1; submit as 8/2 (unreduced) — but the limit must equal 4.
    # Instead, modify a reciprocal_sum that has a non-trivial denominator.
    # Find a sample whose reciprocal_sum has a reducible representation.
    sample = submission["result"]["samples"][0]
    n = sample["n"]
    reciprocal = sum((Fraction(1, i**3 - i) for i in range(2, n + 1)), Fraction())
    # Double both numerator and denominator to create an unreduced form.
    unreduced = {
        "numerator": reciprocal.numerator * 2,
        "denominator": reciprocal.denominator * 2,
    }
    sample["reciprocal_sum"] = unreduced
    write_bound(app, submission_path, submission)
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 1.0
    assert result["reward"] == pytest.approx(1.0)
