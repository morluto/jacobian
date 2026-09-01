"""Typed contracts for the divisibility-poset operation."""

from __future__ import annotations

from pydantic import Field

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
MAX_DIVISIBILITY_POSET_DIGIT_WORK = 10_000_000_000


def _divisibility_poset_admission_error(
    source_set: FiniteIntegerSet,
) -> tuple[str, str] | None:
    if len(source_set.elements) > MAX_DIVISIBILITY_POSET_ELEMENTS:
        return (
            "carrier_too_large",
            f"source set must have at most {MAX_DIVISIBILITY_POSET_ELEMENTS} elements",
        )
    if any(value == "0" or value.startswith("-") for value in source_set.elements):
        return ("non_positive", "all source set elements must be positive integers")
    max_digits = max((len(value) for value in source_set.elements), default=1)
    pair_count = len(source_set.elements) ** 2
    if pair_count * max_digits**2 > MAX_DIVISIBILITY_POSET_DIGIT_WORK:
        return (
            "digit_work_exceeded",
            "divisibility arithmetic exceeds the admitted digit work budget",
        )
    return None


class DivisibilityPosetRequest(StrictModel):
    """A bounded finite set of distinct positive integers for poset construction."""

    source_set: FiniteIntegerSet


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
    "_divisibility_poset_admission_error",
]
