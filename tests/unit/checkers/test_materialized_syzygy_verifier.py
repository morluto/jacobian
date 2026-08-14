from __future__ import annotations

from fractions import Fraction

from jacobian.canonical import format_canonical_integer
from jacobian.domains.polynomial.checkers import (
    POLYNOMIAL_AUTHORIZED_CHECKERS,
    _materialized_syzygy_supports,
)
from jacobian_checkers.jacobian_syzygy import (
    _matrix_digest,
    _rational,
    _wire_rational,
)


def test_materialized_syzygy_verifier_is_domain_owned() -> None:
    declaration = next(
        declaration
        for declaration in POLYNOMIAL_AUTHORIZED_CHECKERS
        if declaration.operation_id
        == "polynomial.jacobian_syzygy.coefficients.materialize"
    )

    assert declaration.verification_operation_id == (
        "polynomial.jacobian_syzygy.coefficients.verify"
    )
    assert declaration.function == "check_materialized_graded_jacobian_syzygy"


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


def test_materialized_syzygy_budget_accounts_for_coefficient_digits() -> None:
    terms = [
        {
            "coefficient": {"num": "1" * 32_768, "den": "1"},
            "exponents": [16 - index, index, 0],
        }
        for index in range(7)
    ]

    assert not _materialized_syzygy_supports(
        {
            "polynomial": {"polynomial": {"terms": terms}},
            "max_degree": 8,
        }
    )
