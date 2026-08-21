"""Provider-independent exact values for finite topological spaces."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_TOPOLOGY_POINTS = 32
MAX_TOPOLOGY_OPENS = 1024


class FiniteTopology(StrictModel):
    """A topology on the labelled carrier ``0..point_count-1``."""

    point_count: int = Field(ge=1, le=MAX_TOPOLOGY_POINTS)
    open_sets: tuple[tuple[int, ...], ...] = Field(
        min_length=2, max_length=MAX_TOPOLOGY_OPENS
    )

    @model_validator(mode="after")
    def require_topology_axioms(self) -> Self:
        canonical: list[frozenset[int]] = []
        for open_set in self.open_sets:
            if tuple(sorted(set(open_set))) != open_set:
                raise ValueError("each open set must be sorted with distinct points")
            if any(not 0 <= point < self.point_count for point in open_set):
                raise ValueError("open set point is outside the carrier")
            canonical.append(frozenset(open_set))
        opens = set(canonical)
        if len(opens) != len(canonical):
            raise ValueError("open sets must be distinct")
        full = frozenset(range(self.point_count))
        if frozenset() not in opens or full not in opens:
            raise ValueError("empty and full sets must be open")
        for left_index, left in enumerate(canonical):
            for right in canonical[left_index:]:
                if left | right not in opens:
                    raise ValueError("open sets must be closed under unions")
                if left & right not in opens:
                    raise ValueError("open sets must be closed under intersections")
        return self


class PointMap(StrictModel):
    """A total map between labelled finite carriers."""

    domain_point_count: int = Field(ge=1, le=MAX_TOPOLOGY_POINTS)
    codomain_point_count: int = Field(ge=1, le=MAX_TOPOLOGY_POINTS)
    values: tuple[int, ...] = Field(min_length=1, max_length=MAX_TOPOLOGY_POINTS)

    @model_validator(mode="after")
    def require_total_bounded_map(self) -> Self:
        if len(self.values) != self.domain_point_count:
            raise ValueError("map must have one value per domain point")
        if any(not 0 <= target < self.codomain_point_count for target in self.values):
            raise ValueError("map value is outside the codomain carrier")
        return self


class BeatPointWitness(StrictModel):
    point: int = Field(ge=0, le=MAX_TOPOLOGY_POINTS - 1)
    witness: int = Field(ge=0, le=MAX_TOPOLOGY_POINTS - 1)


__all__ = [
    "MAX_TOPOLOGY_OPENS",
    "MAX_TOPOLOGY_POINTS",
    "BeatPointWitness",
    "FiniteTopology",
    "PointMap",
]
