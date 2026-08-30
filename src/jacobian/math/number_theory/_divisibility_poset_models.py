"""Typed contracts for finite divisibility poset construction."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.combinatorics.posets.core._models import (
    MAX_POSET_ELEMENTS,
    ElementLabel,
)

MAX_DIVISIBILITY_SET_SIZE = 500


class DivisibilityPosetRequest(StrictModel):
    """Construct the proper-divisibility poset of a finite set of positive integers."""

    values: tuple[ElementLabel, ...] = Field(
        min_length=1,
        max_length=min(MAX_DIVISIBILITY_SET_SIZE, MAX_POSET_ELEMENTS),
        description=(
            "Finite set of positive canonical decimal integers. The poset "
            "has a < b exactly when a divides b and a != b."
        ),
        examples=["2", "3", "6"],
    )

    @model_validator(mode="after")
    def require_positive_unique_values(self) -> Self:
        if any(parse_canonical_integer(value) <= 0 for value in self.values):
            raise PydanticCustomError(
                "divisibility_poset.positive_values",
                "values must be positive canonical integers",
            )
        if len(set(self.values)) != len(self.values):
            raise PydanticCustomError(
                "divisibility_poset.values_unique",
                "values must be distinct",
            )
        return self


__all__ = [
    "MAX_DIVISIBILITY_SET_SIZE",
    "DivisibilityPosetRequest",
]
