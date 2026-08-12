from __future__ import annotations

from fractions import Fraction

import jacobian.exact_domain_checkers as exact_domain_checkers
from jacobian.canonical import format_canonical_integer
from jacobian.domains.polynomial.checkers import POLYNOMIAL_EXACT_REPLAY_CHECKERS
from jacobian_checkers.jacobian_syzygy import (
    _matrix_digest,
    _rational,
    _wire_rational,
)


def test_materialized_syzygy_verifier_is_domain_owned() -> None:
    declaration = next(
        declaration
        for declaration in POLYNOMIAL_EXACT_REPLAY_CHECKERS
        if declaration.capability_id
        == "polynomial.jacobian_syzygy.coefficients.materialize"
    )

    assert declaration.verification_capability_id == (
        "polynomial.jacobian_syzygy.coefficients.verify"
    )
    assert declaration.function == "check_materialized_graded_jacobian_syzygy"


def test_syzygy_checker_support_uses_actual_repeated_factor_support() -> None:
    payload = {
        "linear_factors": [
            {
                "label": str(index),
                "coefficients": [
                    {"num": "1", "den": "1"},
                    {"num": "0", "den": "1"},
                    {"num": "0", "den": "1"},
                ],
            }
            for index in range(6)
        ],
        "max_degree": 8,
    }

    assert exact_domain_checkers._checker_supports(
        "polynomial.jacobian_syzygy.coefficients.materialize", payload
    )


def test_syzygy_checker_support_accepts_immediate_expanded_kernel() -> None:
    payload = {
        "polynomial": {
            "polynomial": {
                "terms": [
                    {
                        "coefficient": {"num": "1", "den": "1"},
                        "exponents": [16 - index, index, 0],
                    }
                    for index in range(17)
                ]
            }
        },
        "max_degree": 8,
    }

    assert exact_domain_checkers._checker_supports(
        "polynomial.jacobian_syzygy.coefficients.materialize", payload
    )


def test_syzygy_checker_support_rejects_excessive_replay() -> None:
    payload = {
        "linear_factors": [
            {
                "label": str(index),
                "coefficients": [
                    {"num": "1", "den": "1"},
                    {"num": "1", "den": "1"},
                    {"num": "1", "den": "1"},
                ],
            }
            for index in range(16)
        ],
        "max_degree": 8,
    }

    assert not exact_domain_checkers._checker_supports(
        "polynomial.jacobian_syzygy.coefficients.materialize", payload
    )


def test_syzygy_checker_accepts_canonical_coefficients_above_decimal_limit() -> None:
    numerator = 10**4_500 + 7
    value = {
        "num": format_canonical_integer(numerator),
        "den": "3",
    }

    parsed = _rational(value)

    assert parsed == Fraction(numerator, 3)
    assert _wire_rational(parsed) == value


def test_syzygy_matrix_digest_formats_coefficients_above_decimal_limit() -> None:
    numerator = 10**4_500 + 7

    digest = _matrix_digest(
        multiplier_degree=0,
        source_basis=((0, 0, 0),),
        target_basis=((0, 0, 0),),
        entries=((0, 0, Fraction(numerator, 3)),),
    )

    assert digest.startswith("sha256:")
