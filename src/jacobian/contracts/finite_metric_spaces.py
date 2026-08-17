"""Typed wire contracts for exact finite metric space operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.base import ContractModel

MAX_POINTS = 64


class PointLabel(ContractModel):
    """One point label in a finite metric space."""

    index: int = Field(ge=0, le=MAX_POINTS - 1)


class FiniteMetricSpace(ContractModel):
    """A finite metric space as an upper-triangular distance matrix."""

    point_count: int = Field(ge=2, le=MAX_POINTS)
    distances: tuple[tuple[int, ...], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid_distances(self) -> Self:
        self._require_square()
        self._require_metric_properties()
        return self

    def _require_square(self) -> None:
        if len(self.distances) != self.point_count:
            raise ValueError("distance matrix row count must match point_count")
        for row in self.distances:
            if len(row) != self.point_count:
                raise ValueError("distance matrix must be square")

    def _require_metric_properties(self) -> None:
        for i in range(self.point_count):
            if self.distances[i][i] != 0:
                raise ValueError("diagonal distances must be zero")
            for j in range(self.point_count):
                if self.distances[i][j] != self.distances[j][i]:
                    raise ValueError("distance matrix must be symmetric")
                if self.distances[i][j] < 0:
                    raise ValueError("distances must be nonnegative")


class MetricProfileRequest(ContractModel):
    """Compute distance profile, radius, diameter, centers, periphery."""

    metric_space: FiniteMetricSpace


class EccentricityResult(ContractModel):
    """One point's eccentricity."""

    point: int = Field(ge=0, le=MAX_POINTS - 1)
    eccentricity: int = Field(ge=0)


class MetricProfileResult(ContractModel):
    """Profile of a finite metric space."""

    diameter: int = Field(ge=0)
    radius: int = Field(ge=0)
    eccentricities: tuple[EccentricityResult, ...] = Field(min_length=2)
    centers: tuple[int, ...] = Field(min_length=1)
    periphery: tuple[int, ...] = Field(min_length=0)
    method: Literal["FLOYD_WARSHALL"] = "FLOYD_WARSHALL"


class BallRequest(ContractModel):
    """Compute the ball of given radius centered at a point."""

    metric_space: FiniteMetricSpace
    center: int = Field(ge=0, le=MAX_POINTS - 1)
    radius: int = Field(ge=0, le=10000)


class BallResult(ContractModel):
    """The ball (set of points within radius of center)."""

    center: int = Field(ge=0, le=MAX_POINTS - 1)
    radius: int = Field(ge=0, le=10000)
    points: tuple[int, ...] = Field(min_length=1)
    method: Literal["DIRECT_SCAN"] = "DIRECT_SCAN"


class GromovHyperbolicityRequest(ContractModel):
    """Compute the four-point Gromov hyperbolicity of a metric space."""

    metric_space: FiniteMetricSpace


class GromovHyperbolicityResult(ContractModel):
    """The four-point Gromov hyperbolicity (max delta over all quadruples)."""

    hyperbolicity: int = Field(ge=0)
    method: Literal["FOUR_POINT_BRUTE_FORCE"] = "FOUR_POINT_BRUTE_FORCE"


__all__ = [
    "BallRequest",
    "BallResult",
    "EccentricityResult",
    "FiniteMetricSpace",
    "GromovHyperbolicityRequest",
    "GromovHyperbolicityResult",
    "MetricProfileRequest",
    "MetricProfileResult",
    "PointLabel",
]
