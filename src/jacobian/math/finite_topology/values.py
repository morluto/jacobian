"""Provider-independent values for exact finite topological spaces."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_TOPOLOGY_POINTS = 64
MAX_TOPOLOGY_OPENS = 1024


class FiniteTopology(StrictModel):
    """A finite topology on labelled points.

    A topology is specified by a set of points (0..n-1) and a collection of
    open subsets. The empty set and the full set are required. Arbitrary
    unions and finite intersections of opens must be opens (validated lazily
    via the specialization preorder representation).
    The ``open_sets`` must include the empty set ``()`` and the full set
    ``tuple(range(point_count))``.
    """

    point_count: int = Field(ge=1, le=MAX_TOPOLOGY_POINTS)
    open_sets: tuple[tuple[int, ...], ...] = Field(
        min_length=2, max_length=MAX_TOPOLOGY_OPENS
    )

    @model_validator(mode="after")
    def require_valid_topology(self) -> Self:
        for op in self.open_sets:
            for pt in op:
                if not 0 <= pt < self.point_count:
                    raise ValueError("open set point out of range")
        if () not in self.open_sets:
            raise ValueError("empty set must be an open set")
        full = tuple(range(self.point_count))
        if full not in self.open_sets:
            raise ValueError("full set must be an open set")
        seen: set[tuple[int, ...]] = set()
        for op in self.open_sets:
            canonical = tuple(sorted(set(op)))
            if canonical in seen:
                raise ValueError("duplicate open set")
            seen.add(canonical)
        return self


class PointMap(StrictModel):
    """A map from one finite topology to another."""

    domain_point_count: int = Field(ge=1, le=MAX_TOPOLOGY_POINTS)
    codomain_point_count: int = Field(ge=1, le=MAX_TOPOLOGY_POINTS)
    function: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid_map(self) -> Self:
        if len(self.function) != self.domain_point_count:
            raise ValueError("function must have domain_point_count entries")
        for target in self.function:
            if not 0 <= target < self.codomain_point_count:
                raise ValueError("function value out of range")
        return self


__all__ = [
    "MAX_TOPOLOGY_OPENS",
    "MAX_TOPOLOGY_POINTS",
    "FiniteTopology",
    "PointMap",
]
