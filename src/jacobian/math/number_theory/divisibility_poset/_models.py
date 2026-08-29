"""Typed contracts for the divisibility-poset operation."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.sets._models import (
    FiniteIntegerSet,
)
from jacobian.math.combinatorics.posets.core._models import (
    MAX_POSET_ELEMENTS,
    ElementLabel,
    FinitePoset,
)

MAX_DIVISIBILITY_POSET_ELEMENTS = MAX_POSET_ELEMENTS  # 64


class DivisibilityPosetRequest(StrictModel):
    """A bounded finite set of distinct positive integers for poset construction."""

    source_set: FiniteIntegerSet

    @model_validator(mode="after")
    def require_positive_and_bounded(self) -> Self:
        from jacobian.canonical import parse_canonical_integer

        if len(self.source_set.elements) > MAX_DIVISIBILITY_POSET_ELEMENTS:
            raise PydanticCustomError(
                "number_theory.divisibility_poset.carrier_too_large",
                f"source set must have at most "
                f"{MAX_DIVISIBILITY_POSET_ELEMENTS} elements",
            )
        for value in self.source_set.elements:
            if parse_canonical_integer(value) <= 0:
                raise PydanticCustomError(
                    "number_theory.divisibility_poset.non_positive",
                    "all source set elements must be positive integers",
                )
        return self


class ElementSource(StrictModel):
    """One poset element label mapped to its source canonical integer."""

    label: ElementLabel
    source_integer: CanonicalInteger


class IntegerDivisibilityPosetResult(StrictModel):
    """Canonical finite poset under proper divisibility with source mapping."""

    source_set: FiniteIntegerSet
    poset: FinitePoset
    element_sources: tuple[ElementSource, ...] = Field(
        max_length=MAX_DIVISIBILITY_POSET_ELEMENTS
    )


__all__ = [
    "MAX_DIVISIBILITY_POSET_ELEMENTS",
    "DivisibilityPosetRequest",
    "ElementSource",
    "IntegerDivisibilityPosetResult",
]
