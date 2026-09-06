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


class MetricProfileRequest(StrictModel):
    """Compute distance profile, radius, diameter, centers, periphery."""

    metric_space: FiniteMetricSpace


class EccentricityResult(StrictModel):
    """One point's eccentricity."""

    point: int = Field(ge=0, le=MAX_POINTS - 1)
    eccentricity: int = Field(ge=0, le=MAX_DISTANCE)


class MetricProfileResult(StrictModel):
    """Profile of a finite metric space."""

    metric_space: FiniteMetricSpace
    diameter: int = Field(ge=0, le=MAX_DISTANCE)
    radius: int = Field(ge=0, le=MAX_DISTANCE)
    eccentricities: tuple[EccentricityResult, ...] = Field(min_length=2)
    centers: tuple[int, ...] = Field(min_length=1)
    periphery: tuple[int, ...] = Field(min_length=0)

    @model_validator(mode="after")
    def bind_point_axes(self) -> Self:
        expected = tuple(range(self.metric_space.point_count))
        if tuple(value.point for value in self.eccentricities) != expected:
            raise _validation_error(
                "eccentricity_axis_mismatch",
                "eccentricities must cover the metric-space point axis in order",
            )
        for name, values in (("centers", self.centers), ("periphery", self.periphery)):
            if tuple(sorted(set(values))) != values or any(
                point not in expected for point in values
            ):
                raise _validation_error(
                    f"{name}_axis_mismatch",
                    f"{name} must be sorted unique indices on the metric-space point axis",
                )
        return self


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

    metric_space: FiniteMetricSpace
    center: int = Field(ge=0, le=MAX_POINTS - 1)
    radius: int = Field(ge=0, le=MAX_DISTANCE)
    points: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def bind_point_axes(self) -> Self:
        point_count = self.metric_space.point_count
        if self.center >= point_count:
            raise _validation_error(
                "ball_center_out_of_range",
                "ball center index must be within the metric space",
            )
        if tuple(sorted(set(self.points))) != self.points or any(
            not 0 <= point < point_count for point in self.points
        ):
            raise _validation_error(
                "ball_points_axis_mismatch",
                "ball points must be sorted unique indices within the metric space",
            )
        return self


class GromovHyperbolicityRequest(StrictModel):
    """Compute the four-point Gromov hyperbolicity of a metric space."""

    metric_space: FiniteMetricSpace


class GromovHyperbolicityResult(StrictModel):
    """The four-point Gromov hyperbolicity (max delta over all quadruples)."""

    metric_space: FiniteMetricSpace
    hyperbolicity: CanonicalRational
