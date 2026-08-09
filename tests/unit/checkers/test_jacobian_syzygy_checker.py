from __future__ import annotations

from fractions import Fraction

from jacobian.canonical import format_canonical_integer
from jacobian_checkers.jacobian_syzygy import (
    _matrix_digest,
    _rational,
    _wire_rational,
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
