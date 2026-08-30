"""Typed contracts for finite divisibility poset construction."""

from __future__ import annotations

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.sets._models import (
    FiniteIntegerSet,
)

MAX_DIVISIBILITY_SET_SIZE = 500


class DivisibilityPosetRequest(StrictModel):
    """Construct the proper-divisibility poset of a finite set of positive integers."""

    values: FiniteIntegerSet = Field(
        description=(
            "Finite set of positive canonical decimal integers. The poset "
            "has a < b exactly when a divides b and a != b."
        ),
        examples=[{"elements": ["2", "3", "6"]}],
    )


__all__ = [
    "MAX_DIVISIBILITY_SET_SIZE",
    "DivisibilityPosetRequest",
]
