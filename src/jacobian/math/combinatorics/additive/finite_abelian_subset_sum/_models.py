"""Typed carriers for finite abelian subset-sum profiles."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, model_validator

from jacobian._exact import DecimalIntegerEncoding
from jacobian._models import StrictModel
from jacobian.math.groups.finite_abelian import (
    FiniteAbelianGroupElement,
    FiniteAbelianProductGroup,
)

MAX_FINITE_ABELIAN_SUBSET_SUM_ITEMS = 4095
MAX_FINITE_ABELIAN_SUBSET_SUM_ORDER = 4096
MAX_FINITE_ABELIAN_SUBSET_SUM_TRANSITIONS = 4_000_000
MAX_FINITE_ABELIAN_SUBSET_SUM_RANKED_WORK = 16_777_216
MAX_FINITE_ABELIAN_SUBSET_SUM_COORDINATE_SLOTS = 16_777_216
MAX_FINITE_ABELIAN_SUBSET_SUM_OUTPUT_BITS = 67_108_864
MAX_FINITE_ABELIAN_SUBSET_SUM_MULTIPLICITY_BITS = MAX_FINITE_ABELIAN_SUBSET_SUM_ITEMS
MAX_FINITE_ABELIAN_SUBSET_SUM_MULTIPLICITY_DIGITS = (
    MAX_FINITE_ABELIAN_SUBSET_SUM_MULTIPLICITY_BITS * 30_103 + 99_999
) // 100_000 + 1
SubsetMultiplicity = Annotated[
    int,
    DecimalIntegerEncoding(max_digits=MAX_FINITE_ABELIAN_SUBSET_SUM_MULTIPLICITY_DIGITS),
]


class FiniteAbelianSubsetSumRequest(StrictModel):
    group: FiniteAbelianProductGroup
    sequence: tuple[FiniteAbelianGroupElement, ...] = Field(
        max_length=MAX_FINITE_ABELIAN_SUBSET_SUM_ITEMS
    )

    @model_validator(mode="after")
    def require_group_bound(self) -> Self:
        if (
            self.group.order > MAX_FINITE_ABELIAN_SUBSET_SUM_ORDER
        ):
            raise ValueError(
                "finite abelian subset-sum group order must be between 2 and 4,096"
            )
        if len(self.sequence) > MAX_FINITE_ABELIAN_SUBSET_SUM_ITEMS:
            raise ValueError("finite abelian subset-sum sequence is too long")
        if any(element.group != self.group for element in self.sequence):
            raise ValueError("every sequence element must use the supplied group")
        return self


class FiniteAbelianSubsetSumRow(StrictModel):
    element: FiniteAbelianGroupElement
    multiplicity: SubsetMultiplicity = Field(ge=0)


class FiniteAbelianSubsetSumResult(StrictModel):
    group: FiniteAbelianProductGroup
    sequence: tuple[FiniteAbelianGroupElement, ...] = Field(
        max_length=MAX_FINITE_ABELIAN_SUBSET_SUM_ITEMS
    )
    rows: tuple[FiniteAbelianSubsetSumRow, ...] = Field(
        max_length=MAX_FINITE_ABELIAN_SUBSET_SUM_ORDER
    )
    support_size: int = Field(ge=0)
    covers_group: bool
    total_subsets: SubsetMultiplicity = Field(ge=1)

    @model_validator(mode="after")
    def require_complete_profile(self) -> Self:
        if self.group.order > MAX_FINITE_ABELIAN_SUBSET_SUM_ORDER:
            raise ValueError("finite abelian subset-sum result group exceeds its bound")
        if any(element.group != self.group for element in self.sequence):
            raise ValueError("profile sequence must use the supplied group")
        if (len(self.sequence) + self.group.order) * len(self.group.moduli) > (
            MAX_FINITE_ABELIAN_SUBSET_SUM_COORDINATE_SLOTS
        ):
            raise ValueError("finite abelian subset-sum result exceeds coordinate bound")
        coordinates = tuple(row.element.coordinates for row in self.rows)
        if len(self.rows) != self.group.order or coordinates != tuple(sorted(coordinates)) or len(set(coordinates)) != len(coordinates):
            raise ValueError(
                "finite abelian subset-sum rows must enumerate the group canonically"
            )
        if any(row.element.group != self.group for row in self.rows):
            raise ValueError("profile rows must use the supplied group")
        if self.support_size != sum(row.multiplicity > 0 for row in self.rows):
            raise ValueError(
                "support size must equal the positive-multiplicity row count"
            )
        if self.covers_group != (self.support_size == self.group.order):
            raise ValueError("coverage must equal full group support")
        if sum(row.multiplicity for row in self.rows) != self.total_subsets:
            raise ValueError("profile multiplicities must sum to total subsets")
        return self


__all__ = [
    "MAX_FINITE_ABELIAN_SUBSET_SUM_COORDINATE_SLOTS",
    "MAX_FINITE_ABELIAN_SUBSET_SUM_ITEMS",
    "MAX_FINITE_ABELIAN_SUBSET_SUM_MULTIPLICITY_BITS",
    "MAX_FINITE_ABELIAN_SUBSET_SUM_MULTIPLICITY_DIGITS",
    "MAX_FINITE_ABELIAN_SUBSET_SUM_ORDER",
    "MAX_FINITE_ABELIAN_SUBSET_SUM_OUTPUT_BITS",
    "MAX_FINITE_ABELIAN_SUBSET_SUM_RANKED_WORK",
    "MAX_FINITE_ABELIAN_SUBSET_SUM_TRANSITIONS",
    "FiniteAbelianSubsetSumRequest",
    "FiniteAbelianSubsetSumResult",
    "FiniteAbelianSubsetSumRow",
]
