"""Typed wire contracts for exact geometry point-configuration operations."""

from __future__ import annotations

from enum import StrEnum
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
# Circumradius profile
# ---------------------------------------------------------------------------


class CircumradiusProfileRequest(StrictModel):
    """Compute the exact circumradius profile of every unordered triple."""

    configuration: PointConfiguration

    @model_validator(mode="after")
    def require_planar(self) -> Self:
        if not self.configuration.points:
            return self
        if len(self.configuration.points[0].coordinates) != 2:
            raise ValueError("circumradius profile requires a planar configuration")
        return self


class CircumradiusTripleDisposition(StrEnum):
    """Mathematical disposition of one unordered triple."""

    NONDEGENERATE = "NONDEGENERATE"
    DEGENERATE = "DEGENERATE"


class CircumradiusProfileEntry(StrictModel):
    """One unordered triple and its exact squared circumradius."""

    triple: tuple[int, int, int] = Field(min_length=3, max_length=3)
    disposition: CircumradiusTripleDisposition
    squared_radius: CanonicalRational | None = Field(default=None)

    @model_validator(mode="after")
    def require_consistent_radius(self) -> Self:
        if self.disposition is CircumradiusTripleDisposition.NONDEGENERATE:
            if self.squared_radius is None:
                raise ValueError(
                    "a nondegenerate triple must report its squared radius"
                )
            if self.squared_radius.as_fraction() <= 0:
                raise ValueError("squared circumradius must be positive")
        elif self.disposition is CircumradiusTripleDisposition.DEGENERATE:
            if self.squared_radius is not None:
                raise ValueError("a degenerate triple must not report a squared radius")
        return self


class CircumradiusMultiplicityEntry(StrictModel):
    """One positive squared circumradius and how many triples share it."""

    squared_radius: CanonicalRational
    triple_count: int = Field(gt=0)


class CircumradiusProfileResult(StrictModel):
    """Complete exact circumradius data for a planar point configuration."""

    dimension: int = Field(ge=2, le=2)
    point_count: int = Field(ge=3)
    triples: tuple[CircumradiusProfileEntry, ...]
    multiplicities: tuple[CircumradiusMultiplicityEntry, ...]
    nondegenerate_count: int = Field(ge=0)
    degenerate_count: int = Field(ge=0)

    @model_validator(mode="after")
    def require_consistent_counts(self) -> Self:
        from itertools import combinations

        expected = len(list(combinations(range(self.point_count), 3)))
        if len(self.triples) != expected:
            raise ValueError("triples must cover every unordered triple exactly once")
        nondeg = sum(
            1
            for entry in self.triples
            if entry.disposition is CircumradiusTripleDisposition.NONDEGENERATE
        )
        if self.nondegenerate_count != nondeg:
            raise ValueError("nondegenerate_count must match the nondegenerate triples")
        if self.degenerate_count != expected - nondeg:
            raise ValueError("degenerate_count must match the degenerate triples")
        mult_total = sum(entry.triple_count for entry in self.multiplicities)
        if mult_total != nondeg:
            raise ValueError("multiplicity total must match the nondegenerate count")
        return self


__all__ = [
    "CircumradiusMultiplicityEntry",
    "CircumradiusProfileEntry",
    "CircumradiusProfileRequest",
    "CircumradiusProfileResult",
    "CircumradiusTripleDisposition",
    "DistanceGraphRequest",
    "DistanceGraphResult",
    "DistanceMultiplicityEntry",
    "DistanceProfileRequest",
    "DistanceProfileResult",
    "LabelledRationalPoint",
    "PointConfiguration",
]
