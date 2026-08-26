"""Shared Arb conversion helpers for analysis enclosure families."""

from __future__ import annotations

from typing import Any

from jacobian.canonical import format_canonical_integer
from jacobian.math.analysis._models import MAX_DYADIC_EXPONENT, ExactDyadic


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


__all__ = ["dyadic_endpoints"]
