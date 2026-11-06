"""Useful degree-bound constructions and pre-replay refusals."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest
from benchmarks.validation._verifier_child import run_verifier_in_child
from benchmarks.validation.mathematical_benchmarks_v1._fixtures import _prepare_case
from jsonschema import Draft202012Validator


def _rational(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _from_roots(roots: list[Fraction], leading: int) -> list[dict[str, int]]:
    coefficients = [Fraction(leading)]
    for root in roots:
        product = [Fraction(0)] * (len(coefficients) + 1)
        for index, coefficient in enumerate(coefficients):
            product[index] += coefficient
            product[index + 1] -= root * coefficient
        coefficients = product
    return [_rational(value) for value in coefficients]


@pytest.mark.parametrize(("degree", "accepted"), [(32, True), (33, False)])
def test_degree_envelope_retains_a_large_exact_counterexample(
    tmp_path: Path, degree: int, *, accepted: bool
) -> None:
    task, app, logs = _prepare_case(
        tmp_path, "polynomial-tail-counterexample", "degree"
    )
    extra = [Fraction(-n) for n in range(2, degree - 1)]
    proots = [Fraction(-1), Fraction(-1, 3), Fraction(0), *extra]
    qroots = [Fraction(-1), Fraction(0), *extra]
    # P-Q = x(x+1)(2x-1/3) * product(x+n). It is zero at 0 and
    # negative at 1/100; all roots are distinct and nonpositive.
    submission = {
        "result": {
            "p_coefficients": _from_roots(proots, 2),
            "q_coefficients": _from_roots(qroots, 1),
            "p_roots": [_rational(x) for x in reversed(proots)],
            "q_roots": [_rational(x) for x in reversed(qroots)],
            "x1": _rational(Fraction(0)),
            "x2": _rational(Fraction(1, 100)),
        }
    }
    schema = json.loads((task / "environment/submission_schema.json").read_text())
    assert Draft202012Validator(schema).is_valid(submission) is accepted
    (app / "submission.json").write_text(json.dumps(submission))
    assert run_verifier_in_child(task=task, app=app, logs=logs).reward == float(
        accepted
    )


def test_oversized_scalar_and_wrong_monotonicity_claim_score_zero(
    tmp_path: Path,
) -> None:
    for scenario, x2 in [
        ("oversized", {"numerator": 1, "denominator": 1 << 256}),
        ("wrong", {"numerator": 1, "denominator": 1}),
    ]:
        task, app, logs = _prepare_case(
            tmp_path / scenario, "polynomial-tail-counterexample", scenario
        )
        path = app / "submission.json"
        submission = json.loads(path.read_text())
        submission["result"]["x2"] = x2
        path.write_text(json.dumps(submission))
        assert run_verifier_in_child(task=task, app=app, logs=logs).reward == 0.0
