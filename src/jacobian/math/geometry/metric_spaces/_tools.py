"""Finite metric space operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.geometry.metric_spaces._models import (
    BallRequest,
    BallResult,
    GromovHyperbolicityRequest,
    GromovHyperbolicityResult,
    MetricProfileRequest,
    MetricProfileResult,
)
from jacobian.math.geometry.metric_spaces.operations import (
    ball,
    gromov_hyperbolicity,
    metric_profile,
)


def _metric_profile(request: MetricProfileRequest) -> MetricProfileResult:
    return metric_profile(request.metric_space)


def _ball(request: BallRequest) -> BallResult:
    return ball(request.metric_space, request.center, request.radius)


def _gromov_hyperbolicity(
    request: GromovHyperbolicityRequest,
) -> GromovHyperbolicityResult:
    return gromov_hyperbolicity(request.metric_space)


_METRIC_SPACE = {
    "metric_space": {
        "point_count": 3,
        "distances": [[0, 1, 2], [1, 0, 1], [2, 1, 0]],
    }
}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="metric_space.profile.compute",
        title="Compute diameter, radius, eccentricities, centers, and periphery",
        description="Compute the exact metric profile of a finite metric space: "
        "diameter (max eccentricity), radius (min eccentricity), "
        "eccentricities for all points, centers, and periphery.",
        request_type=MetricProfileRequest,
        result_type=MetricProfileResult,
        run=_metric_profile,
        tags=("metric", "profile", "exact"),
        examples=(
            OperationExample(
                name="path_graph",
                description="Profile of a path metric space with 3 points.",
                input=_METRIC_SPACE,
            ),
        ),
    ),
    MathTool(
        operation_id="metric_space.ball.compute",
        title="Compute the ball of a given radius centered at a point",
        description="Return the set of all points within the given radius of a specified "
        "center point in a finite metric space.",
        request_type=BallRequest,
        result_type=BallResult,
        run=_ball,
        tags=("metric", "ball", "exact"),
        examples=(
            OperationExample(
                name="ball_1",
                description="Ball of radius 1 centered at point 0 in a 3-point space.",
                input={
                    "metric_space": _METRIC_SPACE["metric_space"],
                    "center": 0,
                    "radius": 1,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="metric_space.gromov_hyperbolicity.compute",
        title="Compute the four-point Gromov hyperbolicity",
        description="Compute the exact four-point Gromov hyperbolicity of a finite "
        "metric space by brute-force enumeration over all quadruples.",
        request_type=GromovHyperbolicityRequest,
        result_type=GromovHyperbolicityResult,
        run=_gromov_hyperbolicity,
        tags=("metric", "hyperbolicity", "exact"),
        examples=(
            OperationExample(
                name="path_graph",
                description="Gromov hyperbolicity of a 3-point path metric.",
                input=_METRIC_SPACE,
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
