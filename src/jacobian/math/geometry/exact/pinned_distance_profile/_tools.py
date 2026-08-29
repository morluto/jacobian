"""Typed declarations for the pinned distance profile operation."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.geometry.exact.pinned_distance_profile._models import (
    PinnedDistanceProfileRequest,
    PinnedDistanceProfileResult,
)
from jacobian.math.geometry.exact.pinned_distance_profile.operations import (
    compute_pinned_distance_profile,
)


def _compute(request: PinnedDistanceProfileRequest) -> PinnedDistanceProfileResult:
    return compute_pinned_distance_profile(request.configuration)


TOOLS: MathTools = (
    MathTool(
        operation_id="geometry.points.pinned_distance_support_profile.compute",
        title="Compute exact pinned-distance support profiles",
        description=(
            "For one finite labelled rational point configuration, return for "
            "every source point its complete sorted partition of all other "
            "source labels by exact squared Euclidean distance."
        ),
        request_type=PinnedDistanceProfileRequest,
        result_type=PinnedDistanceProfileResult,
        run=_compute,
        tags=("geometry", "distance", "exact"),
        examples=(
            example(
                "three_points",
                "Pinned distance profile of three collinear points.",
                {
                    "configuration": {
                        "points": [
                            {"label": "a", "coordinates": [{"num": "0", "den": "1"}, {"num": "0", "den": "1"}]},
                            {"label": "b", "coordinates": [{"num": "1", "den": "1"}, {"num": "0", "den": "1"}]},
                            {"label": "c", "coordinates": [{"num": "2", "den": "1"}, {"num": "0", "den": "1"}]},
                        ],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
