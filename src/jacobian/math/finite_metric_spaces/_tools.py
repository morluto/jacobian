"""Finite metric space operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.finite_metric_spaces._models import (
    BallRequest,
    BallResult,
    GromovHyperbolicityRequest,
    GromovHyperbolicityResult,
    MetricProfileRequest,
    MetricProfileResult,
)
from jacobian.math.finite_metric_spaces._operations import (
    compute_ball,
    compute_gromov_hyperbolicity,
    compute_metric_profile,
)


def fms_operation[RequestT: StrictModel, ResultT: StrictModel](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
    version: str = "1",
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version=version,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


_METRIC_SPACE = {
    "metric_space": {
        "point_count": 3,
        "distances": [[0, 1, 2], [1, 0, 1], [2, 1, 0]],
    }
}


FINITE_METRIC_SPACE_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    fms_operation(
        "metric_space.profile.compute",
        "Compute diameter, radius, eccentricities, centers, and periphery",
        "Compute the exact metric profile of a finite metric space: "
        "diameter (max eccentricity), radius (min eccentricity), "
        "eccentricities for all points, centers, and periphery.",
        MetricProfileRequest,
        MetricProfileResult,
        compute_metric_profile,
        "metric",
        "profile",
        "exact",
        examples=(
            example(
                "path_graph",
                "Profile of a path metric space with 3 points.",
                _METRIC_SPACE,
            ),
        ),
        version="2",
    ),
    fms_operation(
        "metric_space.ball.compute",
        "Compute the ball of a given radius centered at a point",
        "Return the set of all points within the given radius of a "
        "specified center point in a finite metric space.",
        BallRequest,
        BallResult,
        compute_ball,
        "metric",
        "ball",
        "exact",
        examples=(
            example(
                "ball_1",
                "Ball of radius 1 centered at point 0 in a 3-point space.",
                {
                    "metric_space": _METRIC_SPACE["metric_space"],
                    "center": 0,
                    "radius": 1,
                },
            ),
        ),
    ),
    fms_operation(
        "metric_space.gromov_hyperbolicity.compute",
        "Compute the four-point Gromov hyperbolicity",
        "Compute the exact four-point Gromov hyperbolicity of a finite "
        "metric space by brute-force enumeration over all quadruples.",
        GromovHyperbolicityRequest,
        GromovHyperbolicityResult,
        compute_gromov_hyperbolicity,
        "metric",
        "hyperbolicity",
        "exact",
        examples=(
            example(
                "path_graph",
                "Gromov hyperbolicity of a 3-point path metric.",
                _METRIC_SPACE,
            ),
        ),
    ),
)

TOOLS = FINITE_METRIC_SPACE_OPERATIONS

__all__ = ["TOOLS"]
