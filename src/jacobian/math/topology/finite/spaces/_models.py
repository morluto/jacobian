"""Typed wire contracts for finite topological space operations."""

from __future__ import annotations

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.math._labels import OpaqueLabel
from jacobian.math.topology.finite.spaces.values import (
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
    """The canonical quotient map; each target label is its first source representative."""

    quotient_map: FiniteTopologicalMap

    @property
    def quotient_points(self) -> tuple[tuple[OpaqueLabel, ...], ...]:
        return tuple(
            tuple(
                label
                for label, image in zip(
                    self.quotient_map.source.points, self.class_map, strict=True
                )
                if image == target
            )
            for target in range(len(self.quotient_map.target.points))
        )

    @property
    def quotient_preorder(self) -> tuple[tuple[int, ...], ...]:
        return self.quotient_map.target.preorder

    @property
    def class_map(self) -> tuple[int, ...]:
        return self.quotient_map.point_map


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
