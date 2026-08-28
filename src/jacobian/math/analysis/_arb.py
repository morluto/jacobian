"""Shared Arb conversion helpers for analysis enclosure families."""

from __future__ import annotations

from typing import Any

from jacobian.canonical import format_canonical_integer
from jacobian.math.analysis._models import (
    MAX_DYADIC_EXPONENT,
    ExactDyadic,
)
from jacobian.math.analysis.intervals import ClosedRationalInterval


def dyadic_endpoints(
    lower_mantissa: Any,
    lower_exponent: Any,
    upper_mantissa: Any,
    upper_exponent: Any,
) -> tuple[ExactDyadic, ExactDyadic] | None:
    """Serialize Arb endpoints only when their exponents fit the wire contract."""

    if (
        abs(lower_exponent) > MAX_DYADIC_EXPONENT
        or abs(upper_exponent) > MAX_DYADIC_EXPONENT
    ):
        return None
    return (
        ExactDyadic(
            mantissa=format_canonical_integer(int(lower_mantissa)),
            exponent=int(lower_exponent),
        ),
        ExactDyadic(
            mantissa=format_canonical_integer(int(upper_mantissa)),
            exponent=int(upper_exponent),
        ),
    )


def _normalize_dyadic_pair(mantissa: int, exponent: int) -> tuple[int, int]:
    if mantissa == 0:
        return 0, 0
    while mantissa % 2 == 0:
        mantissa //= 2
        exponent += 1
    return mantissa, exponent


def _add_dyadic_pairs(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    common_exponent = min(left[1], right[1])
    return _normalize_dyadic_pair(
        (left[0] << (left[1] - common_exponent))
        + (right[0] << (right[1] - common_exponent)),
        common_exponent,
    )


def _negate_dyadic_pair(value: tuple[int, int]) -> tuple[int, int]:
    return -value[0], value[1]


def _halve_dyadic_pair(value: tuple[int, int]) -> tuple[int, int]:
    return _normalize_dyadic_pair(value[0], value[1] - 1)


def _exact_arb_dyadic_pair(value: Any) -> tuple[int, int]:
    mantissa, exponent = value.man_exp()
    return _normalize_dyadic_pair(int(mantissa), int(exponent))


def arb_source_interval(interval: ClosedRationalInterval) -> Any:
    """Build one Arb ball that contains the exact rational source interval.

    Arb radii have a fixed implementation precision. Anchoring a one-sided
    interval at its endpoint nearest zero preserves a proved sign even when
    the radius is rounded upward by python-flint's public constructor.
    """

    from flint import arb, fmpq

    lower_ratio = interval.lower.as_integer_ratio()
    upper_ratio = interval.upper.as_integer_ratio()
    if lower_ratio == upper_ratio:
        return arb(fmpq(*lower_ratio))

    lower = _exact_arb_dyadic_pair(arb(fmpq(*lower_ratio)).lower())
    upper = _exact_arb_dyadic_pair(arb(fmpq(*upper_ratio)).upper())
    half_width = _halve_dyadic_pair(
        _add_dyadic_pairs(upper, _negate_dyadic_pair(lower))
    )
    actual_radius = _exact_arb_dyadic_pair(arb((0, 0), half_width).upper())
    if interval.lower.as_fraction() >= 0:
        midpoint = _add_dyadic_pairs(lower, actual_radius)
    elif interval.upper.as_fraction() <= 0:
        midpoint = _add_dyadic_pairs(upper, _negate_dyadic_pair(actual_radius))
    else:
        midpoint = _halve_dyadic_pair(_add_dyadic_pairs(lower, upper))
    return arb(midpoint, half_width)
