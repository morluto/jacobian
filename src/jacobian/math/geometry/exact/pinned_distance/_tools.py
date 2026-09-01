"""Pinned distance support profile operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.geometry.exact.pinned_distance._models import (
    PinnedDistanceSupportProfileRequest,
    PinnedDistanceSupportProfileResult,
)
from jacobian.math.geometry.exact.pinned_distance.operations import (
    compute_pinned_distance_support_profile,
)


def compute_pinned_distance_support_profile_op(
    request: PinnedDistanceSupportProfileRequest,
) -> PinnedDistanceSupportProfileResult:
    return compute_pinned_distance_support_profile(request.configuration)


TOOLS: MathTools = (
    MathTool(
        operation_id="geometry.points.pinned_distance_support_profile.compute",
        title="Compute the pinned distance support profile of a point configuration",
        description=(
            "For each source point in a finite labelled rational point "
            "configuration, return the complete sorted partition of all "
            "other source labels by exact squared Euclidean distance."
        ),
        request_type=PinnedDistanceSupportProfileRequest,
        result_type=PinnedDistanceSupportProfileResult,
        run=compute_pinned_distance_support_profile_op,
        tags=("geometry", "exact"),
        examples=(
            OperationExample(
                name="unit_square",
                description="Unit square with 4 points.",
                input={
                    "configuration": {
                        "points": [
                            {
                                "label": "a",
                                "coordinates": [
                                    {"num": "0", "den": "1"},
                                    {"num": "0", "den": "1"},
                                ],
                            },
                            {
                                "label": "b",
                                "coordinates": [
                                    {"num": "1", "den": "1"},
                                    {"num": "0", "den": "1"},
                                ],
                            },
                            {
                                "label": "c",
                                "coordinates": [
                                    {"num": "0", "den": "1"},
                                    {"num": "1", "den": "1"},
                                ],
                            },
                            {
                                "label": "d",
                                "coordinates": [
                                    {"num": "1", "den": "1"},
                                    {"num": "1", "den": "1"},
                                ],
                            },
                        ],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
