"""Typed wire contracts for finite topology operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.finite_topology.values import (
    FiniteTopology,
    PointMap,
)


class SpecializationPreorderRequest(StrictModel):
    """Compute the specialization preorder of a finite topology."""

    topology: FiniteTopology


class SpecializationPreorderResult(StrictModel):
    """The specialization preorder as an adjacency relation.

    For points ``i, j``: ``preorder[i][j]`` is True iff ``j`` is in the
    closure of ``{i}`` (i.e., ``i <= j`` in the specialization order).
    """

    preorder: tuple[tuple[bool, ...], ...]


class ClosureRequest(StrictModel):
    """Compute the closure of a subset."""

    topology: FiniteTopology
    subset: tuple[int, ...] = Field(default=())

    @model_validator(mode="after")
    def require_valid_subset(self) -> Self:
        for pt in self.subset:
            if not 0 <= pt < self.topology.point_count:
                raise ValueError("subset point out of range")
        return self


class ClosureResult(StrictModel):
    """The closure of the subset."""

    closure: tuple[int, ...]


class InteriorRequest(StrictModel):
    """Compute the interior of a subset."""

    topology: FiniteTopology
    subset: tuple[int, ...]

    @model_validator(mode="after")
    def require_valid_subset(self) -> Self:
        if not self.subset:
            return self
        for pt in self.subset:
            if not 0 <= pt < self.topology.point_count:
                raise ValueError("subset point out of range")
        return self


class InteriorResult(StrictModel):
    """The interior of the subset."""

    interior: tuple[int, ...]


class ConnectedComponentsRequest(StrictModel):
    """Compute connected components of a finite topology."""

    topology: FiniteTopology


class ConnectedComponentsResult(StrictModel):
    """Connected components as a partition of the point set."""

    components: tuple[tuple[int, ...], ...]


class IsContinuousRequest(StrictModel):
    """Check if a point map between topologies is continuous."""

    domain: FiniteTopology
    codomain: FiniteTopology
    function: PointMap


class IsContinuousResult(StrictModel):
    """Whether the map is continuous."""

    is_continuous: bool


class BeatPointsRequest(StrictModel):
    """Find all beat points (up and down) of a T0 finite space."""

    topology: FiniteTopology


class BeatPointsResult(StrictModel):
    """The up and down beat points."""

    up_beat_points: tuple[int, ...]
    down_beat_points: tuple[int, ...]


__all__ = [
    "BeatPointsRequest",
    "BeatPointsResult",
    "ClosureRequest",
    "ClosureResult",
    "ConnectedComponentsRequest",
    "ConnectedComponentsResult",
    "InteriorRequest",
    "InteriorResult",
    "IsContinuousRequest",
    "IsContinuousResult",
    "SpecializationPreorderRequest",
    "SpecializationPreorderResult",
]
