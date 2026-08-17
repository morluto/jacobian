"""Typed wire contracts for finite set-system discrepancy operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_GROUND_SET = 20
MAX_SETS = 100


class FiniteSetSystem(StrictModel):
    """A finite ground set [n] and a family of subsets over it.

    Each subset is a tuple of distinct element indices in 0..n-1. The
    ground set size ``n`` bounds the indices that may appear in any
    subset. An empty family is permitted.
    """

    ground_set_size: int = Field(ge=0, le=MAX_GROUND_SET, strict=True)
    sets: tuple[tuple[int, ...], ...] = Field(max_length=MAX_SETS)

    @model_validator(mode="after")
    def require_valid_sets(self) -> Self:
        for subset in self.sets:
            seen: set[int] = set()
            for element in subset:
                if not (0 <= element < self.ground_set_size):
                    raise ValueError("subset element must be in 0..ground_set_size-1")
                if element in seen:
                    raise ValueError("subset elements must be distinct")
                seen.add(element)
        return self


class DiscrepancyEvalRequest(StrictModel):
    """Evaluate the signed sums and maximum imbalance of a coloring."""

    set_system: FiniteSetSystem
    coloring: tuple[int, ...]

    @model_validator(mode="after")
    def require_valid_coloring(self) -> Self:
        if len(self.coloring) != self.set_system.ground_set_size:
            raise ValueError(
                "coloring length must equal ground_set_size",
            )
        for value in self.coloring:
            if value not in (-1, 1):
                raise ValueError("coloring values must be +1 or -1")
        return self


class DiscrepancyEvalResult(StrictModel):
    """The signed sum on every set and the maximum absolute imbalance."""

    signed_sums: tuple[int, ...]
    max_absolute_imbalance: int = Field(ge=0, strict=True)


class DiscrepancyOptimumRequest(StrictModel):
    """Search for a coloring minimizing the maximum discrepancy."""

    set_system: FiniteSetSystem


class DiscrepancyOptimumResult(StrictModel):
    """The optimum coloring found and its discrepancy."""

    optimal_coloring: tuple[int, ...]
    optimal_discrepancy: int = Field(ge=0, strict=True)
    exhaustive: bool


__all__ = [
    "MAX_GROUND_SET",
    "MAX_SETS",
    "DiscrepancyEvalRequest",
    "DiscrepancyEvalResult",
    "DiscrepancyOptimumRequest",
    "DiscrepancyOptimumResult",
    "FiniteSetSystem",
]
