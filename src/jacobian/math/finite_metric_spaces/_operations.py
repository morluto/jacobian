"""Domain-owned finite metric space operations."""

from __future__ import annotations

from jacobian._exact import CanonicalRational
from jacobian.math.finite_metric_spaces import (
    ball,
    gromov_hyperbolicity,
    metric_profile,
)
from jacobian.math.finite_metric_spaces._models import (
    BallRequest,
    BallResult,
    EccentricityResult,
    FiniteMetricSpace,
    GromovHyperbolicityRequest,
    GromovHyperbolicityResult,
    MetricProfileRequest,
    MetricProfileResult,
)


def _distance_matrix(metric_space: FiniteMetricSpace) -> list[list[int]]:
    return [list(row) for row in metric_space.distances]


def compute_metric_profile(request: MetricProfileRequest) -> MetricProfileResult:
    distances = _distance_matrix(request.metric_space)
    profile = metric_profile(distances)
    n = len(distances)
    return MetricProfileResult(
        diameter=profile["diameter"],
        radius=profile["radius"],
        eccentricities=tuple(
            EccentricityResult(point=i, eccentricity=profile["eccentricities"][i])
            for i in range(n)
        ),
        centers=profile["centers"],
        periphery=profile["periphery"],
    )


def compute_ball(request: BallRequest) -> BallResult:
    distances = _distance_matrix(request.metric_space)
    points = ball(distances, request.center, request.radius)
    return BallResult(
        center=request.center,
        radius=request.radius,
        points=tuple(points),
    )


def compute_gromov_hyperbolicity(
    request: GromovHyperbolicityRequest,
) -> GromovHyperbolicityResult:
    distances = _distance_matrix(request.metric_space)
    result = gromov_hyperbolicity(distances)
    return GromovHyperbolicityResult(
        hyperbolicity=CanonicalRational.from_fraction(result)
    )
