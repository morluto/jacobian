"""Canonical values shared by additive-combinatorics operations."""

from __future__ import annotations

from itertools import pairwise
from typing import Annotated, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel

__all__ = ["IndexSubset", "IndexedIntegerSequence"]

NonnegativeIndex = Annotated[StrictInt, Field(ge=0)]


class IndexedIntegerSequence(StrictModel):
    """A finite materialized tuple of indexed integers.

    Tuple order defines the indices, so repeated values and zeros remain
    distinct selectable positions.  Operation-specific requests own their
    item-count, digit, and work limits.
    """

    values: tuple[CanonicalInteger, ...] = Field(
        description=(
            "Canonical signed decimal integers in index order; repeated values "
            "and zeros remain distinct positions."
        ),
        examples=[("2", "3")],
    )


class IndexSubset(StrictModel):
    """A finite subset of nonnegative indices in canonical increasing order."""

    indices: tuple[NonnegativeIndex, ...] = Field(
        description="Strictly increasing nonnegative indices.",
        examples=[(0, 2)],
    )

    @model_validator(mode="after")
    def require_strictly_increasing_indices(self) -> Self:
        if any(left >= right for left, right in pairwise(self.indices)):
            raise ValueError("subset indices must be strictly increasing")
        return self
