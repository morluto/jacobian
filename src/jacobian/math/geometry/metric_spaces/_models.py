"""Typed wire contracts for exact finite metric space operations."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel

MAX_POINTS = 64
MAX_DISTANCE = (1 << 53) - 1

DistanceValue = Annotated[int, Field(ge=0, le=MAX_DISTANCE)]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by finite-metric-space contracts."""

    return PydanticCustomError(f"finite_metric_space.{reason}", message)


class FiniteMetricSpace(StrictModel):
    """A finite metric space given by its complete distance matrix."""

    point_count: int = Field(ge=2, le=MAX_POINTS)
    distances: tuple[tuple[DistanceValue, ...], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid_distances(self) -> Self:
        self._require_square()
        self._require_metric_properties()
        return self

    def _require_square(self) -> None:
        if len(self.distances) != self.point_count:
            raise _validation_error(
                "distance_row_count_mismatch",
                "distance matrix row count must match point_count",
            )
        for row in self.distances:
            if len(row) != self.point_count:
                raise _validation_error(
                    "distance_matrix_not_square", "distance matrix must be square"
                )

    def _require_metric_properties(self) -> None:
        for i in range(self.point_count):
            if self.distances[i][i] != 0:
                raise _validation_error(
                    "distance_diagonal_nonzero", "diagonal distances must be zero"
                )
            for j in range(self.point_count):
                if self.distances[i][j] != self.distances[j][i]:
                    raise _validation_error(
                        "distance_matrix_asymmetric",
                        "distance matrix must be symmetric",
                    )
        self._require_positive_separation()
        self._require_triangle_inequality()

    def _require_positive_separation(self) -> None:
        for i in range(self.point_count):
            for j in range(self.point_count):
                if i != j and self.distances[i][j] == 0:
                    raise _validation_error(
                        "distance_nonpositive_between_distinct_points",
                        "distinct points must have positive distance",
                    )

    def _require_triangle_inequality(self) -> None:
        for i in range(self.point_count):
            for j in range(self.point_count):
                for k in range(self.point_count):
                    if (
                        self.distances[i][j]
                        > self.distances[i][k] + self.distances[k][j]
                    ):
                        raise _validation_error(
                            "distance_triangle_inequality_violation",
                            "distances must satisfy the triangle inequality",
                        )


class MetricProfileRequest(StrictModel):
    """Compute distance profile, radius, diameter, centers, periphery."""

    metric_space: FiniteMetricSpace


class EccentricityResult(StrictModel):
    """One point's eccentricity."""

    point: int = Field(ge=0, le=MAX_POINTS - 1)
    eccentricity: int = Field(ge=0, le=MAX_DISTANCE)


class MetricProfileResult(StrictModel):
    """Profile of a finite metric space."""

    diameter: int = Field(ge=0, le=MAX_DISTANCE)
    radius: int = Field(ge=0, le=MAX_DISTANCE)
    eccentricities: tuple[EccentricityResult, ...] = Field(min_length=2)
    centers: tuple[int, ...] = Field(min_length=1)
    periphery: tuple[int, ...] = Field(min_length=0)


class BallRequest(StrictModel):
    """Compute the ball of given radius centered at a point."""

    metric_space: FiniteMetricSpace
    center: int = Field(ge=0, le=MAX_POINTS - 1)
    radius: int = Field(ge=0, le=MAX_DISTANCE)

    @model_validator(mode="after")
    def require_center_in_range(self) -> Self:
        if self.center >= self.metric_space.point_count:
            raise _validation_error(
                "ball_center_out_of_range",
                "ball center index must be within the metric space",
            )
        return self


class BallResult(StrictModel):
    """The ball (set of points within radius of center)."""

    center: int = Field(ge=0, le=MAX_POINTS - 1)
    radius: int = Field(ge=0, le=MAX_DISTANCE)
    points: tuple[int, ...] = Field(min_length=1)


class GromovHyperbolicityRequest(StrictModel):
    """Compute the four-point Gromov hyperbolicity of a metric space."""

    metric_space: FiniteMetricSpace


class GromovHyperbolicityResult(StrictModel):
    """The four-point Gromov hyperbolicity (max delta over all quadruples)."""

    hyperbolicity: CanonicalRational
