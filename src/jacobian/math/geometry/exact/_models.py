"""Typed wire contracts for exact geometry point-configuration operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel

MAX_POINTS = 64
MAX_DIMENSION = 20


class LabelledRationalPoint(StrictModel):
    """A labelled rational point in bounded dimension."""

    label: str = Field(min_length=1, max_length=64)
    coordinates: tuple[CanonicalRational, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid_dimension(self) -> Self:
        if len(self.coordinates) > MAX_DIMENSION:
            raise ValueError("dimension exceeds bound")
        return self


class PointConfiguration(StrictModel):
    """A finite set of labelled rational points in a fixed dimension."""

    points: tuple[LabelledRationalPoint, ...] = Field(
        min_length=2,
        max_length=MAX_POINTS,
    )

    @model_validator(mode="after")
    def require_uniform_dimension(self) -> Self:
        if not self.points:
            return self
        dim = len(self.points[0].coordinates)
        for p in self.points[1:]:
            if len(p.coordinates) != dim:
                raise ValueError("all points must have the same dimension")
        labels = [p.label for p in self.points]
        if len(labels) != len(set(labels)):
            raise ValueError("point labels must be unique")
        return self


class DistanceProfileRequest(StrictModel):
    """Compute exact pairwise squared distances."""

    configuration: PointConfiguration


class DistanceMultiplicityEntry(StrictModel):
    """One squared distance and how many pairs have it."""

    squared_distance: CanonicalRational
    pair_count: int = Field(gt=0)


class DistanceProfileResult(StrictModel):
    """Complete distance multiplicity profile of a point configuration."""

    dimension: int = Field(ge=1)
    point_count: int = Field(ge=2)
    entries: tuple[DistanceMultiplicityEntry, ...]


class DistanceGraphRequest(StrictModel):
    """Build the graph induced by a selected squared distance."""

    configuration: PointConfiguration
    target_squared_distance: CanonicalRational = Field(
        description="Nonnegative squared Euclidean distance to select.",
    )

    @model_validator(mode="after")
    def require_nonnegative_target(self) -> Self:
        if self.target_squared_distance.as_fraction() < 0:
            raise ValueError("squared distance target must be nonnegative")
        return self


class DistanceGraphResult(StrictModel):
    """Graph whose edges connect pairs at the target squared distance."""

    vertex_count: int = Field(ge=2)
    edges: tuple[tuple[int, int], ...]


# ---------------------------------------------------------------------------
# Configuration-wide incidence search: collinear triples and concyclic quadruples
# ---------------------------------------------------------------------------


class CollinearTriplesRequest(StrictModel):
    """Search a planar configuration for collinear triples."""

    configuration: PointConfiguration

    @model_validator(mode="after")
    def require_planar(self) -> Self:
        if not self.configuration.points:
            return self
        if len(self.configuration.points[0].coordinates) != 2:
            raise ValueError("collinear-triple search requires a planar configuration")
        return self


class ConcyclicQuadruplesRequest(StrictModel):
    """Search a planar configuration for concyclic quadruples."""

    configuration: PointConfiguration

    @model_validator(mode="after")
    def require_planar(self) -> Self:
        if not self.configuration.points:
            return self
        if len(self.configuration.points[0].coordinates) != 2:
            raise ValueError(
                "concyclic-quadruple search requires a planar configuration"
            )
        return self


class IncidenceSearchResult(StrictModel):
    """Witnesses to a forbidden planar incidence configuration, or none."""

    dimension: int = Field(ge=2, le=2)
    point_count: int = Field(ge=3)
    holds: bool = Field(
        description="True iff at least one witness incidence exists.",
    )
    witnesses: tuple[tuple[int, ...], ...] = Field(default=())

    @model_validator(mode="after")
    def require_consistent_witnesses(self) -> Self:
        if self.holds and not self.witnesses:
            raise ValueError("a holds=True result must list at least one witness")
        if not self.holds and self.witnesses:
            raise ValueError("a holds=False result must list no witnesses")
        return self


__all__ = [
    "CollinearTriplesRequest",
    "ConcyclicQuadruplesRequest",
    "DistanceGraphRequest",
    "DistanceGraphResult",
    "DistanceMultiplicityEntry",
    "DistanceProfileRequest",
    "DistanceProfileResult",
    "IncidenceSearchResult",
    "LabelledRationalPoint",
    "PointConfiguration",
]
