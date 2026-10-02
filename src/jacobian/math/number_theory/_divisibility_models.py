"""Typed contracts owned by integer divisibility operations."""

from __future__ import annotations

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.math.number_theory._integer_input import IntegerInput
from jacobian.math.number_theory._models import BoundedInteger


class IntegerPairRequest(StrictModel):
    """Two canonical integers supplied to a symmetric binary operation."""

    left: IntegerInput = Field(
        description=(
            "Exact integer with at most 256 decimal digits, as a canonical decimal "
            "string or a safe JSON integer (absolute value at most 9007199254740991)."
        )
    )
    right: IntegerInput = Field(
        description=(
            "Exact integer with at most 256 decimal digits, as a canonical decimal "
            "string or a safe JSON integer (absolute value at most 9007199254740991)."
        )
    )


class DivisibilityRequest(StrictModel):
    """A divisor and dividend supplied to a divisibility predicate."""

    divisor: BoundedInteger
    dividend: BoundedInteger


class ValuationRequest(StrictModel):
    """One integer and a prime base supplied to a p-adic valuation."""

    value: BoundedInteger
    prime: BoundedInteger


class ExtendedGcdResult(StrictModel):
    """A gcd together with exact Bezout coefficients."""

    gcd: BoundedInteger
    left_coefficient: BoundedInteger
    right_coefficient: BoundedInteger


__all__ = [
    "DivisibilityRequest",
    "ExtendedGcdResult",
    "IntegerPairRequest",
    "ValuationRequest",
]
