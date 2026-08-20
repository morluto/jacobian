"""Typed wire contracts for finite topological space operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.finite_topology_spaces.values import (
    FiniteTopologicalMap,
    FiniteTopologicalSpace,
)


class SubsetRequest(StrictModel):
    """Operate on a subset of points."""

    space: FiniteTopologicalSpace
    subset: tuple[int, ...] = Field(default=())

    @model_validator(mode="after")
    def require_valid_indices(self) -> Self:
        n = len(self.space.points)
        for i in self.subset:
            if not 0 <= i < n:
                raise ValueError("subset index out of range")
        return self


class InteriorResult(StrictModel):
    interior: tuple[int, ...]


class ClosureResult(StrictModel):
    closure: tuple[int, ...]


class BoundaryResult(StrictModel):
    boundary: tuple[int, ...]


class ContinuousCheckRequest(StrictModel):
    """Check continuity of a point map."""

    point_map: FiniteTopologicalMap


class ContinuousCheckResult(StrictModel):
    is_continuous: bool


class KolmogorovQuotientRequest(StrictModel):
    space: FiniteTopologicalSpace


class KolmogorovQuotientResult(StrictModel):
    quotient_points: tuple[str, ...]
    quotient_preorder: tuple[tuple[int, ...], ...]


__all__ = [
    "BoundaryResult",
    "ClosureResult",
    "ContinuousCheckRequest",
    "ContinuousCheckResult",
    "InteriorResult",
    "KolmogorovQuotientRequest",
    "KolmogorovQuotientResult",
    "SubsetRequest",
]
