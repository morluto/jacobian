"""Domain-owned finite metric space operations."""

from __future__ import annotations

from jacobian.math.geometry.metric_spaces import (
    ball,
    gromov_hyperbolicity,
    metric_profile,
)
from jacobian.math.geometry.metric_spaces._models import (
    BallRequest,
    BallResult,
    GromovHyperbolicityRequest,
    GromovHyperbolicityResult,
    MetricProfileRequest,
    MetricProfileResult,
)


def compute_metric_profile(request: MetricProfileRequest) -> MetricProfileResult:
    return metric_profile(request.metric_space)


def compute_ball(request: BallRequest) -> BallResult:
    return ball(request.metric_space, request.center, request.radius)


def compute_gromov_hyperbolicity(
    request: GromovHyperbolicityRequest,
) -> GromovHyperbolicityResult:
    return gromov_hyperbolicity(request.metric_space)
