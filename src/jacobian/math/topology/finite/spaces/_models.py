"""Typed wire contracts for finite topological space operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math._labels import OpaqueLabel
from jacobian.math.topology.finite.spaces.values import (
    FiniteTopologicalMap,
    FiniteTopologicalSpace,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by finite-topology contracts."""

    return PydanticCustomError(f"finite_topology_space.{reason}", message)


class SubsetRequest(StrictModel):
    """Operate on a subset of points."""

    space: FiniteTopologicalSpace
    subset: tuple[int, ...] = Field(default=())

    @model_validator(mode="after")
    def require_valid_indices(self) -> Self:
        n = len(self.space.points)
        for i in self.subset:
            if not 0 <= i < n:
                raise _validation_error(
                    "subset_index_out_of_range", "subset index out of range"
                )
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
    quotient_points: tuple[tuple[OpaqueLabel, ...], ...]
    quotient_preorder: tuple[tuple[int, ...], ...]
    class_map: tuple[int, ...]


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
