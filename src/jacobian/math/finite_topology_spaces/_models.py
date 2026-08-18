"""Typed wire contracts for finite topological space operations."""

from __future__ import annotations

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.math.finite_topology_spaces.values import (
    FiniteTopologicalMap,
    FiniteTopologicalSpace,
)


class SubsetRequest(StrictModel):
    """Operate on a subset of points."""

    space: FiniteTopologicalSpace
    subset: tuple[int, ...] = Field(default=())


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
