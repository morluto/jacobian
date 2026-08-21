"""Typed wire contracts for exact geometry point-configuration operations."""

from __future__ import annotations

from fractions import Fraction
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


__all__ = [
    "DistanceGraphRequest",
    "DistanceGraphResult",
    "DistanceMultiplicityEntry",
    "DistanceProfileRequest",
    "DistanceProfileResult",
    "LabelledRationalPoint",
    "PinnedLineDistanceRequest",
    "PinnedLineDistanceResult",
    "PinnedLineEntry",
    "PointConfiguration",
]


# ---------------------------------------------------------------------------
# Pinned line-distance profile
# ---------------------------------------------------------------------------


class PinnedLineDistanceRequest(StrictModel):
    """Compute distances from an anchor to all pair-spanned lines."""

    configuration: PointConfiguration
    anchor: tuple[CanonicalRational, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_planar_and_matching_anchor(self) -> Self:
        if not self.configuration.points:
            return self
        if len(self.configuration.points[0].coordinates) != 2:
            raise ValueError("pinned line-distance profile requires a planar configuration")
        if len(self.anchor) != 2:
            raise ValueError("the anchor must be a planar rational point")
        # A pair of coincident points does not span a line; require distinct
        # coordinates so every pair defines a geometric line.
        coords = {
            tuple(c.as_fraction() for c in pt.coordinates)
            for pt in self.configuration.points
        }
        if len(coords) != len(self.configuration.points):
            raise ValueError(
                "pinned line-distance profile requires distinct point coordinates",
            )
        return self


class PinnedLineEntry(StrictModel):
    """One pair-spanned line with its canonical equation and source pairs."""

    line_coefficients: tuple[CanonicalRational, ...] = Field(min_length=3, max_length=3)
    squared_distance: CanonicalRational
    pairs: tuple[tuple[int, int], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_sorted_pairs(self) -> Self:
        for i, j in self.pairs:
            if not i < j:
                raise ValueError("source pairs must be ordered (i < j)")
        if len(set(self.pairs)) != len(self.pairs):
            raise ValueError("source pairs must be unique")
        if self.squared_distance.as_fraction() < 0:
            raise ValueError("squared distance must be nonnegative")
        return self


class PinnedLineDistanceResult(StrictModel):
    """Complete pinned line-distance profile for a point configuration."""

    dimension: int = Field(ge=2, le=2)
    point_count: int = Field(ge=2)
    lines: tuple[PinnedLineEntry, ...]
    distance_multiplicities: tuple[tuple[CanonicalRational, int], ...]

    @model_validator(mode="after")
    def require_consistent_profile(self) -> Self:
        from itertools import combinations

        expected_pairs = len(list(combinations(range(self.point_count), 2)))
        total_pairs = sum(len(entry.pairs) for entry in self.lines)
        if total_pairs != expected_pairs:
            raise ValueError("every source pair must be assigned to exactly one line")
        # distance multiplicities must partition the lines by squared distance
        mult: dict[Fraction, int] = {}
        for entry in self.lines:
            mult[entry.squared_distance.as_fraction()] = mult.get(
                entry.squared_distance.as_fraction(), 0
            ) + 1
        reconstructed = tuple(
            (
                CanonicalRational.from_fraction(d),
                count,
            )
            for d, count in sorted(mult.items())
        )
        if reconstructed != self.distance_multiplicities:
            raise ValueError("distance multiplicities must partition the lines and be sorted")
        return self
