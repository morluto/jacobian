"""Domain adapter for finite metric space operations."""

from __future__ import annotations

from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.finite_metric_spaces import (
    BallRequest,
    BallResult,
    EccentricityResult,
    GromovHyperbolicityRequest,
    GromovHyperbolicityResult,
    MetricProfileRequest,
    MetricProfileResult,
)
from jacobian.math.finite_metric_spaces import (
    ball,
    gromov_hyperbolicity,
    metric_profile,
)


def _distance_matrix(
    request: MetricProfileRequest | BallRequest | GromovHyperbolicityRequest,
) -> list[list[int]]:
    ms = request.metric_space
    return [list(row) for row in ms.distances]


def compute_metric_profile(request: MetricProfileRequest) -> MetricProfileResult:
    distances = _distance_matrix(request)
    result = metric_profile(distances)
    n = len(distances)
    return MetricProfileResult(
        diameter=result["diameter"],
        radius=result["radius"],
        eccentricities=tuple(
            EccentricityResult(point=i, eccentricity=result["eccentricities"][i])
            for i in range(n)
        ),
        centers=result["centers"],
        periphery=result["periphery"],
    )


def compute_ball(request: BallRequest) -> BallResult:
    distances = _distance_matrix(request)
    center = request.center
    radius = request.radius
    points = ball(distances, center, radius)
    return BallResult(
        center=center,
        radius=radius,
        points=tuple(points),
    )


def compute_gromov_hyperbolicity(
    request: GromovHyperbolicityRequest,
) -> GromovHyperbolicityResult:
    distances = _distance_matrix(request)
    result = gromov_hyperbolicity(distances)
    return GromovHyperbolicityResult(
        hyperbolicity=CanonicalRational.from_fraction(result)
    )
