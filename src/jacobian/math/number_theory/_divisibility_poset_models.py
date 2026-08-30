"""Typed contracts for finite divisibility poset construction."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

MAX_DIVISIBILITY_SET_SIZE = 500


class DivisibilityPosetRequest(StrictModel):
    """Construct the proper-divisibility poset of a finite set of positive integers."""

    values: tuple[str, ...] = Field(
        min_length=1,
        max_length=MAX_DIVISIBILITY_SET_SIZE,
        description=(
            "Finite set of positive canonical decimal integers. The poset "
            "has a < b exactly when a divides b and a != b."
        ),
        examples=["2", "3", "6"],
    )


class DivisibilityPosetResult(StrictModel):
    """The canonical proper-divisibility poset."""

    values: tuple[str, ...] = Field(min_length=1)
    strict_order_pairs: tuple[tuple[str, str], ...] = Field(default=())

    @model_validator(mode="after")
    def require_canonical_pairs(self) -> Self:
        for a, b in self.strict_order_pairs:
            if a == b:
                raise PydanticCustomError(
                    "divisibility_poset.no_reflexive",
                    "strict order pairs must not be reflexive",
                )
        return self


__all__ = [
    "MAX_DIVISIBILITY_SET_SIZE",
    "DivisibilityPosetRequest",
    "DivisibilityPosetResult",
]
