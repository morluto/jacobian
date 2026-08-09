from __future__ import annotations

from fractions import Fraction

from jacobian.canonical import format_canonical_integer
from jacobian_checkers.jacobian_syzygy import _rational, _wire_rational


def test_syzygy_checker_accepts_canonical_coefficients_above_decimal_limit() -> None:
    numerator = 10**4_500 + 7
    value = {
        "num": format_canonical_integer(numerator),
        "den": "3",
    }

    parsed = _rational(value)

    assert parsed == Fraction(numerator, 3)
    assert _wire_rational(parsed) == value
